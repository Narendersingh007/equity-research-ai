import os
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict

# --- THE FIX IS HERE ---
# We must reference "backend." because we run uvicorn from the root folder
try:
    from backend.search_logic import RAGPipeline
except ImportError:
    # Fallback for local testing if running directly inside backend/
    from search_logic import RAGPipeline

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI")

app = FastAPI(title="Equity Research AI API", version="1.0.0")

# --- CORS POLICY ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    query: str
    ticker: Optional[str] = None

from typing import Dict, List

class ChatResponse(BaseModel):
    answer: str
    sources: Dict[str, List[str]]
    processing_time: float

# --- GLOBAL STATE ---
rag_engine = None

@app.on_event("startup")
async def startup_event():
    global rag_engine
    logger.info("🚀 Starting Neural Engine...")
    try:
        rag_engine = RAGPipeline()
        logger.info("✅ AI System Online")
    except Exception as e:
        logger.critical(f"❌ Failed to initialize RAG: {e}")

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "active", "model": "Llama-3-Groq-Tool-Use"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="AI System is still loading")
    
    start_time = time.time()
    
    try:
        result = rag_engine.get_answer(request.query)
        duration = time.time() - start_time

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            processing_time=duration
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-data")
async def get_market_data():
    import json
    # Fix path to be absolute based on this file's location
    data_path = os.path.join(os.path.dirname(__file__), "data/market_data.json")
    
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        return {"error": "Market data not found"}