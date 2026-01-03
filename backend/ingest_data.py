import os
import sys
import logging
import time
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load secrets
load_dotenv()

def get_embeddings(client, text):
    """
    Generates embeddings using Gemini 1.5 Flash.
    """
    try:
        # Use the specific embedding model
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text
        )
        return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def ingest_sample_data():
    """
    Ingests 3 sample financial news snippets to test the pipeline.
    """
    # 1. Initialize Clients
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        index_name = os.getenv("INDEX_NAME")
        
        # Connect to the index
        index = pc.Index(index_name)
        logger.info(f"✅ Connected to Pinecone Index: {index_name}")
    except Exception as e:
        logger.critical(f"Failed to connect to services: {e}")
        sys.exit(1)

    # 2. Sample Data
    sample_news = [
        {"id": "news_1", "text": "Apple (AAPL) reports record Q4 earnings due to strong iPhone 16 sales."},
        {"id": "news_2", "text": "The Federal Reserve hints at a 25bps rate cut next month as inflation cools."},
        {"id": "news_3", "text": "Tesla stock drops 5% after missed delivery targets in China region."}
    ]

    logger.info(f"🚀 Starting ingestion of {len(sample_news)} items...")

    vectors_to_upsert = []

    # 3. Processing Loop
    for item in sample_news:
        logger.info(f"Processing: {item['id']}")
        
        vector = get_embeddings(google_client, item['text'])
        
        if vector:
            vectors_to_upsert.append({
                "id": item["id"],
                "values": vector,
                "metadata": {"text": item["text"], "source": "sample_script"}
            })
            
            # --- CRITICAL FOR FREE TIER ---
            # Sleep 4 seconds to stay under the 15 RPM limit
            logger.info("Sleeping 4s to respect Free Tier limits...")
            time.sleep(4) 
    
    # 4. Upsert to Pinecone
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            logger.info(f"✅ Successfully uploaded {len(vectors_to_upsert)} vectors to Pinecone!")
        except Exception as e:
            logger.error(f"Failed to upsert to Pinecone: {e}")

if __name__ == "__main__":
    ingest_sample_data()