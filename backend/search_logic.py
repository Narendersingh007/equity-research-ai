import os
import logging
import requests
import string
from typing import Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- Environment hygiene ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

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

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL = "intfloat/e5-small-v2"
INDEX_NAME = os.getenv("INDEX_NAME")


class RAGPipeline:
    """
    Equity Research RAG Pipeline
    - Local embeddings (e5-small-v2)
    - Pinecone for historical context
    - Live NewsAPI for recent events
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

        # 4️⃣ LLMs (OpenRouter)
        self.primary_llm = ChatOpenAI(
            model="google/gemini-2.0-flash-exp:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.3,
        )

        # ✅ MULTI FALLBACK (SAFE MODELS)
        self.fallback_llms = [
            ChatOpenAI(
                model="mistralai/mistral-small-3.1-24b-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.3,
            ),
            ChatOpenAI(
                model="mistralai/mistral-7b-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.3,
            ),

        ]
        self.local_llm = ChatOllama(
            model="phi3",
            temperature=0.3,
        )

        # 5️⃣ Build chain
        self.chain = self._build_chain()
        logger.info("✅ RAG Chain initialized.")

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def _extract_keywords(self, query: str) -> str:
        stop_words = {
            "why", "is", "the", "what", "how", "when",
            "did", "does", "a", "an", "in", "on", "of", "for"
        }
        clean = query.translate(str.maketrans("", "", string.punctuation))
        return " ".join(w for w in clean.split() if w.lower() not in stop_words)
    def _sentiment_label(self, score: float) -> str:
        if score <= -0.2:
            return "Negative"
        elif score >= 0.2:
            return "Positive"
        return "Neutral"


    def _confidence_label(self, article_count: int, sentiment_strength: float) -> str:
        if article_count >= 3 or abs(sentiment_strength) >= 0.5:
            return "High"
        if article_count >= 1:
            return "Medium"
        return "Low"

    def _fetch_live_news(self, inputs: Dict[str, Any]) -> str:
        query = self._extract_keywords(inputs["question"])
        if len(query.split()) < 2:
            return ""

        from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        # =========================
        # 1️⃣ MARKETAUX (PRIMARY)
        # =========================
        marketaux_key = os.getenv("MARKETAUX_API_KEY")
        if marketaux_key:
            try:
                url = (
                    "https://api.marketaux.com/v1/news/all?"
                    f"search={query}"
                    f"&published_after={from_date}"
                    "&language=en"
                    "&limit=6"
                    f"&api_token={marketaux_key}"
                )
                resp = requests.get(url, timeout=10)
                data = resp.json()

                articles = data.get("data", [])
                if articles:
                    descriptions = []
                    sentiments = []

                    for art in articles[:6]:
                        # ---- description / snippet ----
                        desc = art.get("description") or art.get("snippet")
                        if desc:
                            descriptions.append(f"- {desc}")

                        # ---- extract sentiment for TSLA ----
                        for ent in art.get("entities", []):
                            if ent.get("symbol") in {"TSLA", "Tesla"}:
                                score = ent.get("sentiment_score")
                                if isinstance(score, (int, float)):
                                    sentiments.append(score)

                    avg_sentiment = (
                        sum(sentiments) / len(sentiments)
                        if sentiments else 0.0
                    )

                    sentiment_label = self._sentiment_label(avg_sentiment)
                    confidence = self._confidence_label(len(descriptions), avg_sentiment)

                    summary = (
                        "--- LIVE FINANCIAL NEWS (Marketaux | Last 24h) ---\n"
                        f"Market Sentiment: {sentiment_label} "
                        f"(avg score: {avg_sentiment:.2f})\n"
                        f"Confidence: {confidence}\n\n"
                        "Key Points:\n"
                        + "\n".join(descriptions)
                    )

                    return summary

            except Exception as e:
                logger.warning(f"Marketaux failed: {e}")

        # =========================
        # 2️⃣ GNEWS (FALLBACK)
        # =========================
        gnews_key = os.getenv("GNEWS_API_KEY")
        if gnews_key:
            try:
                url = (
                    "https://gnews.io/api/v4/search?"
                    f"q={query}&from={from_date}"
                    "&lang=en&max=3&sortby=publishedAt"
                    f"&apikey={gnews_key}"
                )
                resp = requests.get(url, timeout=10)
                data = resp.json()

                articles = data.get("articles", [])
                if articles:
                    summary = "--- LIVE NEWS (GNews | Last 24h) ---\n"
                    for i, art in enumerate(articles, 1):
                        summary += (
                            f"{i}. {art['title']} "
                            f"({art['publishedAt']}) - {art['source']['name']}\n"
                        )
                    return summary

            except Exception as e:
                logger.warning(f"GNews failed: {e}")

    # =========================
    # No live news found
    # =========================
        return ""
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _get_embedding_query(self, inputs: Dict[str, Any]) -> str:
        return f"query: {inputs['question']}"

    # -----------------------------------------------------
    # Chain (NO LLM here)
    # -----------------------------------------------------

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(
            """You are a senior Equity Research Analyst AI.

The user is asking about a SPECIFIC company.
You MUST ONLY discuss the company explicitly mentioned in the question.
Do NOT discuss other companies unless they are directly referenced in LIVE_NEWS
AND are clearly relevant to the same event.

If LIVE_NEWS does not contain information about the company in question,
you MUST explicitly say:
"No company-specific news was found in the past 7 days."

Rules:
- Prioritize LIVE_NEWS for the last 7 days
- Use HISTORY only for background on THE SAME COMPANY
- Do NOT substitute other companies (e.g., Microsoft, Apple) as examples
- Do NOT generalize to the broader market unless explicitly supported
- If evidence is weak, say so clearly

Format your answer as:
1. Company-specific recent developments (past 7 days)
2. Market sentiment related to THIS company
3. Relevant historical or recurring factors
4. Overall assessment

<LIVE_NEWS>
{live_context}
</LIVE_NEWS>

<HISTORY>
{historical_context}
</HISTORY>

Question: {question}

Answer:""")

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

    # -----------------------------------------------------
    # Public API with SAFE multi-fallback
    # -----------------------------------------------------

    def get_answer(self, query: str) -> str:
        messages = None

        # ---- Try primary cloud model ----
        try:
            messages = self.chain.invoke(query)
            response = self.primary_llm.invoke(messages).content
            logger.info(
                f"✅ Answered by PRIMARY model: {self.primary_llm.model_name}"
            )
            return response

        except Exception as e:
            logger.warning(
                f"Primary LLM failed ({self.primary_llm.model_name}): {e}"
            )

        # ---- Try cloud fallbacks sequentially ----
        for idx, llm in enumerate(self.fallback_llms, start=1):
            try:
                logger.info(
                    f"🔁 Trying FALLBACK #{idx}: {llm.model_name}"
                )
                response = llm.invoke(messages).content
                logger.info(
                    f"✅ Answered by FALLBACK #{idx}: {llm.model_name}"
                )
                return response

            except Exception as e:
                logger.warning(
                    f"Fallback #{idx} failed ({llm.model_name}): {e}"
                )

        # ---- FINAL LOCAL OLLAMA FALLBACK ----
        try:
            logger.info("🧠 Using LOCAL Ollama (phi3)")
            response = self.local_llm.invoke(messages).content
            logger.info("✅ Answered by LOCAL Ollama: phi3")
            return response

        except Exception as e:
            logger.error(f"❌ Local Ollama failed: {e}")

        logger.error("❌ All LLMs failed")
        return "The system is temporarily unavailable."

# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    pipeline = RAGPipeline()

    q = "Why is Tesla stock down today?"
    print(f"\n🔎 Query: {q}\n" + "-" * 40)
    print(pipeline.get_answer(q))