import os
import logging
import requests
import string
import feedparser
import trafilatura
from typing import Dict, Any
from dotenv import load_dotenv

# --- LangChain imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# --- Environment & Logging ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
# Silence internal noise
logging.getLogger('trafilatura').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL = "intfloat/e5-small-v2"
INDEX_NAME = os.getenv("INDEX_NAME")

class RAGPipeline:
    """
    Equity Research RAG Pipeline
    - Local embeddings (e5-small-v2)
    - Pinecone for historical context
    - Google News RSS Summaries (Top 10) - FAST & STABLE
    - OpenRouter LLM with multi-fallback
    """

    def __init__(self):
        # 1️⃣ Embeddings
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2️⃣ Vector Store
        self.vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )

        # 3️⃣ Retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})

        # 4️⃣ LLMs
        self.primary_llm = ChatOpenAI(
            model="google/gemini-2.0-flash-exp:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.3,
        )

        self.fallback_llms = [
            ChatOpenAI(
                model="mistralai/mistral-small-3.1-24b-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.3,
            ),
        ]
        self.local_llm = ChatOllama(model="phi3", temperature=0.3)

        # 5️⃣ Build chain
        self.chain = self._build_chain()
        logger.info("✅ RAG Chain initialized.")


    # Helpers


    def _extract_keywords(self, query: str) -> str:
        stop_words = {
            "why", "is", "the", "what", "how", "when",
            "did", "does", "a", "an", "in", "on", "of", "for",
            "update", "with", "stock", "today", "news", "price"
        }
        clean = query.translate(str.maketrans("", "", string.punctuation))
        return " ".join(w for w in clean.split() if w.lower() not in stop_words)

    def _get_robust_stock_news(self, ticker_or_company: str) -> str:
        """
        Fetches Top 10 Google News RSS Summaries.
        Zero scraping overhead = Maximum speed & stability.
        """
        if not ticker_or_company.strip():
            return ""

        # Google News RSS URL
        rss_url = f"https://news.google.com/rss/search?q={ticker_or_company}+stock+news+when:1d&hl=en-US&gl=US&ceid=US:en"
        
        feed = None
        try:
            # We must download the XML ourselves because feedparser gets blocked by Google
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            resp = requests.get(rss_url, headers=headers, timeout=5)
            feed = feedparser.parse(resp.content)
        except Exception as e:
            logger.error(f"RSS Fetch Error: {e}")
            return "Error fetching live news feed."

        rag_output_lines = []
        
        if not feed or not feed.entries:
            return "No relevant news found via RSS."
        
        # --- PROCESS TOP 10 SUMMARIES ---
        limit = 10
        logger.info(f"Found {len(feed.entries)} articles. Extracting top {limit} summaries...")

        for entry in feed.entries[:limit]:
            title = entry.title
            source = entry.source.title if hasattr(entry, 'source') else "Unknown"
            published = entry.published if hasattr(entry, 'published') else "Unknown"
            
            # Just grab the summary provided by Google
            summary_raw = entry.summary if hasattr(entry, 'summary') else ""
            
            # Clean HTML tags from the summary
            content = ""
            try:
                summary_clean = trafilatura.extract(summary_raw)
                content = summary_clean if summary_clean else summary_raw
            except:
                content = summary_raw

            rag_output_lines.append(f"SOURCE: {source} ({published})\nTITLE: {title}\nSUMMARY: {content}\n")

        return "\n".join(rag_output_lines)

    def _fetch_live_news(self, inputs: Dict[str, Any]) -> str:
        query_keywords = self._extract_keywords(inputs["question"])
        if len(query_keywords.split()) < 1:
            return ""
        logger.info(f"Fetching news for: {query_keywords}")
        return self._get_robust_stock_news(query_keywords)

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _get_embedding_query(self, inputs: Dict[str, Any]) -> str:
        return f"query: {inputs['question']}"

    
    # Chain Construction
    

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """You are a senior Equity Research Analyst AI.
The user is asking about a SPECIFIC company.
Use the LIVE_NEWS (Summaries) and HISTORY contexts to answer.

<LIVE_NEWS_SUMMARIES>
{live_context}
</LIVE_NEWS_SUMMARIES>

<HISTORY>
{historical_context}
</HISTORY>

Question: {question}
Answer:"""
        )

        retrieval = RunnableParallel(
            historical_context=(
                RunnableLambda(self._get_embedding_query)
                | self.retriever
                | self._format_docs
            ),
            live_context=RunnableLambda(self._fetch_live_news),
            question=RunnablePassthrough(),
        )

        return (
            {"question": RunnablePassthrough()}
            | retrieval
            | prompt
        )

    def get_answer(self, query: str) -> str:
        messages = None
        try:
            messages = self.chain.invoke(query)
            if not messages: return "Error building context."
            
            return self.primary_llm.invoke(messages).content
        except Exception as e:
            logger.warning(f"Primary failed: {e}")
            
            # Fallback loop
            if messages:
                for llm in self.fallback_llms:
                    try:
                        return llm.invoke(messages).content
                    except:
                        continue
            
            # Local fallback
            try:
                return self.local_llm.invoke(messages).content
            except:
                return "System temporarily unavailable."

if __name__ == "__main__":
    pipeline = RAGPipeline()
    q = "What is update with nvidia stock today?"
    print(f"\n🔎 Query: {q}\n" + "-" * 40)
    print(pipeline.get_answer(q))