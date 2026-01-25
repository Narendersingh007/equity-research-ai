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
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq  

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
    - Google News RSS Summaries (Top 10)
    - PRIMARY LLM: Groq (Llama-3.3-70b) -> Fastest & Free
    - FALLBACKS: OpenRouter (Gemini/Mistral) -> Local (Phi3)
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

        # 4️⃣ LLMs Setup
        

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            logger.warning("⚠️ GROQ_API_KEY missing! Primary LLM might fail.")
            
        self.primary_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=groq_key
        )


        self.fallback_llms = [
            ChatOpenAI(
                model="google/gemini-2.0-flash-exp:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.3,
            ),
            ChatOpenAI(
                model="mistralai/mistral-small-3.1-24b-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.3,
            ),
        ]
        
        # LAST RESORT: Local Ollama
        self.local_llm = ChatOllama(model="phi3", temperature=0.3)

        # 5️⃣ Build chain
        self.chain = self._build_chain()
        logger.info("✅ RAG Chain initialized.")


    # Helpers


    def _extract_keywords(self, query) -> str:
        if isinstance(query, dict):
            query = query.get("question", "")

        stop_words = {
            "why", "is", "the", "what", "how", "when",
            "did", "does", "a", "an", "in", "on", "of", "for",
            "update", "with", "stock", "today", "news", "price"
        }

        clean = str(query).translate(str.maketrans("", "", string.punctuation))
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

    def _extract_sec_sources(self, docs):
        sources = []
        for doc in docs:
            meta = doc.metadata or {}

            company = meta.get("company")
            form = meta.get("form", "10-K")
            year = meta.get("year")

            if company and year:
                sources.append(f"{company} — FY{year} {form}")
            elif meta.get("source"):
                sources.append(meta["source"])

        return list(dict.fromkeys(sources))[:3]


    def _extract_news_sources(self, live_context: str):
        sources = []
        for block in live_context.split("SOURCE:")[1:]:
            header = block.split("\n", 1)[0].strip()
            if header:
                sources.append(header)
        return list(dict.fromkeys(sources))[:5]
    
    # Chain Construction
    

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(
    """
You are a Senior Equity Research Analyst AI producing institutional-grade research briefs.

The user is asking about a SPECIFIC publicly listed company.
Your response will be displayed in a professional research terminal.

CRITICAL RULES:
- Use ONLY the provided contexts.
- Do NOT speculate beyond the evidence.
- If data is insufficient, state that clearly.
- Maintain a neutral, analyst tone (no hype, no financial advice).
- Structure the answer clearly for readability.


LIVE NEWS (Recent Developments)

{live_context}


HISTORICAL CONTEXT (SEC Filings / Prior Data)

{historical_context}


TASK

Answer the user’s question as a concise but thorough **Research Brief**.

REQUIRED STRUCTURE:

1. **Direct Answer**
   - Address the question clearly in 2–4 sentences.

2. **Key Supporting Points**
   - Use bullet points or numbered points where appropriate.
   - Reference concrete facts from the provided contexts.

3. **Risk Factors / Uncertainty (if applicable)**
   - Highlight known risks, limitations, or conflicting signals.

4. **Source Attribution**
   - Implicitly rely on:
     - SEC filings → historical context
     - News articles → live context
   - Do NOT invent sources.


QUESTION

{question}


RESEARCH BRIEF

Write the answer in clean, well-formatted plain text.
Do NOT include markdown headings.
Do NOT include source lists.
"""
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

    def _build_context(self, query: str) -> Dict[str, str]:
        retrieval = RunnableParallel(
            historical_context=(
                RunnableLambda(self._get_embedding_query)
                | self.retriever
                | self._format_docs
            ),
            live_context=RunnableLambda(self._fetch_live_news),
        )

        return retrieval.invoke({"question": query})

    def get_answer(self, query: str) -> Dict[str, Any]:
        try:
            context = self._build_context(query)

            if not context:
                return {
                    "answer": "Unable to build research context.",
                    "sources": {"sec": [], "news": []},
                }

            prompt_input = {
                "question": query,
                "live_context": context["live_context"],
                "historical_context": context["historical_context"],
            }

            try:
                answer_text = self.primary_llm.invoke(
                    self.chain.invoke(prompt_input)
                ).content
            except Exception:
                answer_text = None
                for llm in self.fallback_llms:
                    try:
                        answer_text = llm.invoke(
                            self.chain.invoke(prompt_input)
                        ).content
                        break
                    except Exception:
                        continue

                if not answer_text:
                    answer_text = self.local_llm.invoke(
                        self.chain.invoke(prompt_input)
                    ).content

            historical_docs = self.retriever.invoke(
                self._get_embedding_query({"question": query})
            )

            sources = {
                "sec": self._extract_sec_sources(historical_docs),
                "news": self._extract_news_sources(context["live_context"]),
            }

            return {
                "answer": answer_text.strip(),
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "The system encountered an internal error while generating the research brief.",
                "sources": {"sec": [], "news": []},
            }

if __name__ == "__main__":
    pipeline = RAGPipeline()
    q = "What is update with nvidia stock today?"
    print(f"\n🔎 Query: {q}\n" + "-" * 40)
    print(pipeline.get_answer(q))