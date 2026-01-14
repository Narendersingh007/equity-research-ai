import os
import time
from dotenv import load_dotenv
# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables (takes PINECONE_API_KEY and INDEX_NAME)
load_dotenv()

# ==========================================
# CONFIGURATION (Matches your screenshot)
# ==========================================
# CRITICAL: This must match whatever you use in search_logic.py
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
INDEX_NAME = os.getenv("INDEX_NAME")

# ==========================================
# 1. THE DUMMY DATA GENERATOR
# ==========================================
# Realistic-looking fake risk factors for testing
DUMMY_TSLA_TEXT = """
TESLA, INC. - ANNUAL REPORT (FORM 10-K) - RISK FACTORS SECTION (TEST DATA)

The following are significant risk factors that could affect our business:

1. Automotive Sector Volatility and Competition: The automotive industry is highly competitive and cyclical. Demand for electric vehicles (EVs) may fluctuate due to global economic conditions, changes in interest rates, and consumer confidence. We face aggressive competition from established automakers (e.g., Ford, GM, Toyota) transitioning to EVs, as well as new entrants (e.g., Rivian, BYD), which could pressure our margins and market share.

2. Supply Chain and Production Challenges: We rely on complex global supply chains for critical components, including lithium-ion battery cells, semiconductors, and raw materials. Disruptions, shortages, or price increases in these materials could delay production, increase costs, and prevent us from meeting delivery targets. Ramping up production at new Gigafactories involves significant execution risk.

3. Regulatory and Policy Risks: Our business benefits from government subsidies, tax credits, and zero-emission vehicle mandates in various jurisdictions. Changes to, or the elimination of, these policies could adversely impact demand. Additionally, we are subject to increasing scrutiny regarding vehicle safety, particularly concerning Autopilot and Full Self-Driving (FSD) features, which could lead to recalls or restrictions.

4. Dependence on Key Personnel: We are highly dependent on the services of Elon Musk, our Technoking and Chief Executive Officer. His involvement in other ventures and highly public profile could impact our brand reputation or distract from core business operations.

5. Technology and Cyber Risks: Our vehicles and operations rely heavily on advanced software and connectivity. Any significant breach of our cybersecurity systems or failure in our vehicle software could lead to data loss, vehicle malfunctions, brand damage, and potential liability.
"""

def run_test_pipeline():
    print("--- Starting Test Ingestion Pipeline ---")
    
    # --- Step A: Create dummy text file locally ---
    os.makedirs('data', exist_ok=True)
    temp_file_path = 'data/temp_tsla_dummy.txt'
    with open(temp_file_path, 'w') as f:
        f.write(DUMMY_TSLA_TEXT)
    print(f"✅ Generated dummy data file at: {temp_file_path}")

    # --- Step B: Load and Chunk text ---
    print("Start loading and chunking...")
    loader = TextLoader(temp_file_path)
    raw_docs = loader.load()
    
    # Split text into smaller pieces for embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"✅ Split text into {len(chunks)} chunks.")

    # --- Step C: Add Metadata ---
    # This is crucial so you know WHICH company this text belongs to
    TICKER = "TSLA"
    for chunk in chunks:
        chunk.metadata["ticker"] = TICKER
        chunk.metadata["source"] = "dummy_test_data"

    # --- Step D: Initialize Embeddings (Matches your setup) ---
    print(f"⚡ Loading embedding model: {EMBEDDING_MODEL_NAME} (This runs locally, might take a moment first time)...")
    # This uses your local CPU/GPU to turn text into numbers
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # --- Step E: Upsert to Pinecone ---
    print(f"🚀 Uploading chunks to Pinecone Index: '{INDEX_NAME}'...")
    start_time = time.time()
    
    # This line does the magic: embeds the text and sends it to Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    
    end_time = time.time()
    print(f"🎉 SUCCESS! Uploaded {len(chunks)} vectors in {end_time - start_time:.2f} seconds.")
    
    # Cleanup local file
    os.remove(temp_file_path)
    print("🧹 Cleaned up temporary local file.")

if __name__ == "__main__":
    # Ensure you have .env file with PINECONE_API_KEY and INDEX_NAME before running
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("INDEX_NAME"):
        print("❌ ERROR: Missing PINECONE_API_KEY or INDEX_NAME in .env file.")
    else:
        run_test_pipeline()