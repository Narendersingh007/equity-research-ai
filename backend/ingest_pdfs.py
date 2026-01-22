import os
import logging
from typing import List
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configuration Constants
DATA_DIRECTORY = "data/reports"
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def validate_environment():
    """Validates that necessary environment variables and directories exist."""
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is missing in environment variables.")
        return False
    if not INDEX_NAME:
        logger.error("INDEX_NAME is missing in environment variables.")
        return False
    if not os.path.exists(DATA_DIRECTORY):
        logger.error(f"Data directory not found: {DATA_DIRECTORY}")
        return False
    return True

def ingest_pdfs():
    """
    Main ingestion pipeline:
    1. Loads PDF documents from the configured directory.
    2. Splits documents into semantic chunks.
    3. Generates vector embeddings.
    4. Upserts vectors to the Pinecone index.
    """
    if not validate_environment():
        return

    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return

    # Scan for PDF files
    pdf_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(".pdf")]
    if not pdf_files:
        logger.warning("No PDF files found in the data directory to ingest.")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process.")

    all_chunks: List[Document] = []

    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIRECTORY, pdf_file)
        
        # Extract Ticker from filename (e.g., "NVDA_2024.pdf" -> "NVDA")
        try:
            ticker = pdf_file.split("_")[0].upper()
        except IndexError:
            logger.warning(f"Could not extract ticker from filename: {pdf_file}. Defaulting to 'UNKNOWN'.")
            ticker = "UNKNOWN"

        logger.info(f"Processing file: {pdf_file} (Ticker: {ticker})")

        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()

            # Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                add_start_index=True
            )
            chunks = text_splitter.split_documents(raw_docs)

            # Enrich Metadata
            for chunk in chunks:
                chunk.metadata["ticker"] = ticker
                chunk.metadata["source"] = pdf_file

            logger.info(f"Successfully split {pdf_file} into {len(chunks)} chunks.")
            all_chunks.extend(chunks)

        except Exception as e:
            logger.error(f"Error processing file {pdf_file}: {e}")
            continue

    # Upload to Pinecone
    if all_chunks:
        logger.info(f"Starting upload of {len(all_chunks)} vector embeddings to Pinecone index '{INDEX_NAME}'...")
        try:
            vectorstore = PineconeVectorStore(
                index_name=INDEX_NAME,
                embedding=embeddings,
                pinecone_api_key=PINECONE_API_KEY
            )
            vectorstore.add_documents(all_chunks)
            logger.info("Ingestion pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Failed to upload vectors to Pinecone: {e}")
    else:
        logger.warning("No valid text chunks were extracted. Aborting upload.")

if __name__ == "__main__":
    ingest_pdfs()