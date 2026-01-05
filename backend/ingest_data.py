import os
import sys
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Configure logging for production-grade traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from the secure .env file
load_dotenv()

# Global Constants
MODEL_NAME = 'intfloat/e5-small-v2'
EMBEDDING_PREFIX = 'passage: '

class VectorPipeline:
    """
    Manages the lifecycle of loading embedding models and processing text data
    for vector database ingestion.
    """

    def __init__(self):
        """
        Initializes the pipeline by loading the local embedding model.
        Terminates execution if model loading fails.
        """
        logger.info(f"Initializing embedding model: {MODEL_NAME}...")
        try:
            self.model = SentenceTransformer(MODEL_NAME)
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Fatal Error: Failed to load model. Details: {e}")
            sys.exit(1)

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates a dense vector embedding for the provided text.
        
        Args:
            text (str): The raw text content to be embedded.
            
        Returns:
            Optional[List[float]]: A 384-dimensional vector, or None if generation fails.
        """
        try:
            # The 'passage:' prefix is mandatory for e5 models when indexing documents
            prefixed_text = f"{EMBEDDING_PREFIX}{text}"
            
            # Generate embedding on local CPU/GPU
            vector = self.model.encode(prefixed_text).tolist()
            return vector
        except Exception as e:
            logger.error(f"Embedding generation failed for text segment. Details: {e}")
            return None

def connect_to_pinecone() -> Any:
    """
    Establishes a secure connection to the Pinecone vector database.
    
    Returns:
        Any: The active Pinecone index instance.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("INDEX_NAME")

    if not api_key or not index_name:
        logger.critical("Environment variables PINECONE_API_KEY or INDEX_NAME are missing.")
        sys.exit(1)

    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        logger.info(f"✅ Connected to Pinecone Index: {index_name}")
        return index
    except Exception as e:
        logger.critical(f"Database connection failed. Details: {e}")
        sys.exit(1)

def ingest_data(data_source: List[Dict[str, str]]) -> None:
    """
    Orchestrates the ETL process: Extract text, Transform to vectors, Load to Pinecone.
    
    Args:
        data_source (List[Dict[str, str]]): List of dictionaries containing 'id' and 'text'.
    """
    # 1. Initialize Pipeline and Connection
    pipeline = VectorPipeline()
    index = connect_to_pinecone()

    logger.info(f"Starting ingestion for {len(data_source)} items...")
    
    vectors_to_upsert: List[Dict[str, Any]] = []

    # 2. Transform Loop
    for item in data_source:
        record_id = item.get('id')
        text_content = item.get('text')
        
        logger.info(f"Processing record ID: {record_id}")
        
        vector = pipeline.generate_embedding(text_content)
        
        if vector:
            # Prepare payload compatible with Pinecone API
            payload = {
                "id": record_id,
                "values": vector,
                "metadata": {
                    "text": text_content,
                    "source": "ingest_script"
                }
            }
            vectors_to_upsert.append(payload)

    # 3. Load (Upsert) Operation
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            logger.info(f"✅ Successfully ingested {len(vectors_to_upsert)} vectors to Pinecone.")
        except Exception as e:
            logger.error(f"Upsert operation failed. Details: {e}")
    else:
        logger.warning("No valid vectors were generated. Upsert skipped.")

if __name__ == "__main__":
    # Mock data simulating a production data source (e.g., CSV or Database)
    sample_news_data = [
        {"id": "news_1", "text": "Apple (AAPL) reports record Q4 earnings due to strong iPhone 16 sales."},
        {"id": "news_2", "text": "The Federal Reserve hints at a 25bps rate cut next month as inflation cools."},
        {"id": "news_3", "text": "Tesla stock drops 5% after missed delivery targets in China region."}
    ]

    ingest_data(sample_news_data)