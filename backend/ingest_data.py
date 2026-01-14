import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1. SETUP: Must match your search_logic.py exactly
EMBEDDING_MODEL = "intfloat/e5-small-v2"
INDEX_NAME = os.getenv("INDEX_NAME") 

def ingest_company_text(file_path, ticker):
    print(f"Processing {ticker}...")
    
    # Load Text
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split Text (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    # Add Metadata (Crucial for filtering later)
    for doc in docs:
        doc.metadata["ticker"] = ticker
        doc.metadata["source"] = "annual_report"

    # Embed & Upsert
    print(f"⚡ Generating embeddings using {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectorstore = PineconeVectorStore.from_documents(
        docs,
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    print(f"✅ Successfully added {len(docs)} chunks to Pinecone for {ticker}")

# USAGE EXAMPLE:
# ingest_company_text("data/reports/TSLA_risks.txt", "TSLA")