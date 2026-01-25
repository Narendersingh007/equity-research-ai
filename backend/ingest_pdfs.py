import os
import logging
import re
from typing import List
from dotenv import load_dotenv

# Parsing Libraries
from bs4 import BeautifulSoup

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DATA_DIRECTORY = "backend/data/reports"
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def clean_sec_text(raw_text: str) -> str:
    """
    AGGRESSIVE CLEANER:
    1. Deletes massive <XML> and <XBRL> data blocks (the main source of garbage).
    2. Strips HTML tags.
    3. Removes hexadecimal rubbish.
    """
    try:
        # 1. THE NUCLEAR OPTION: Regex remove XML/XBRL blocks entirely
        # Flags: DOTALL (dot matches newline), IGNORECASE
        text = re.sub(r'<XML>.*?</XML>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<XBRL>.*?</XBRL>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<GRAPHIC>.*?</GRAPHIC>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<JSON>.*?</JSON>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 2. Use BeautifulSoup to strip remaining HTML tags
        soup = BeautifulSoup(text, "lxml") 
        
        # Remove script and style elements (CSS/JS)
        for script in soup(["script", "style", "head", "title", "meta", "table"]): 
            # Note: Removing <table> helps reduce financial data noise, keep if you want tables
            script.extract()    

        # Get text
        clean_text = soup.get_text(separator=" ")
        
        # 3. Collapse multiple spaces/newlines
        clean_text = " ".join(clean_text.split())
        
        return clean_text
    except Exception as e:
        logger.warning(f"HTML cleaning failed: {e}")
        return raw_text

def validate_environment() -> bool:
    if not PINECONE_API_KEY or not INDEX_NAME:
        logger.error("Missing Pinecone credentials.")
        return False
    return True

def get_ticker_from_filename(filename: str) -> str:
    return filename.split("_")[0].upper()

def batch_upsert(vectorstore, chunks, ids, batch_size=100):
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        try:
            vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)
            logger.info(f"   📤 Uploaded batch {i}-{min(i+batch_size, total)} of {total}")
        except Exception as e:
            logger.error(f"   ❌ Batch {i} failed: {e}")

def ingest_docs():
    if not validate_environment(): return

    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})

    try:
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME, 
            embedding=embeddings, 
            pinecone_api_key=PINECONE_API_KEY
        )
    except Exception as e:
        logger.critical(f"Failed to connect to Pinecone: {e}")
        return

    all_files = [f for f in os.listdir(DATA_DIRECTORY) if f.lower().endswith(('.pdf', '.txt'))]
    logger.info(f"Found {len(all_files)} file(s) to process.")

    for file_name in all_files:
        file_path = os.path.join(DATA_DIRECTORY, file_name)
        ticker = get_ticker_from_filename(file_name)
        logger.info(f"Processing: {file_name} (Ticker: {ticker})")

        try:
            # 1. Load & Clean
            if file_name.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                raw_docs = loader.load()
                full_text = " ".join([d.page_content for d in raw_docs])
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_content = f.read()
                # --- APPLY THE NUCLEAR CLEANER ---
                full_text = clean_sec_text(raw_content)

            # 2. Check content size (Filter out empty files)
            if len(full_text) < 1000:
                logger.warning(f"File {file_name} seems empty after cleaning. Skipping.")
                continue

            doc = Document(page_content=full_text, metadata={"source": file_name, "ticker": ticker})

            # 3. Split Content
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents([doc])

            logger.info(f"📉 Reduced to {len(chunks)} clean chunks.")

            # 4. Generate IDs & Batch Upload
            ids = [f"{ticker}_chunk_{i}" for i in range(len(chunks))]
            batch_upsert(vectorstore, chunks, ids)
            logger.info(f"✅ Finished {ticker}")

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    ingest_docs()