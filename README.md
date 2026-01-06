# equity-research-ai

A serverless Retrieval-Augmented Generation (RAG) system designed for
financial equity research.

This project answers market questions like:
"Why is Tesla stock down today?"
by combining live financial news with historical context instead of
relying on LLM training data alone.

## Core Architecture

- Embeddings: Local semantic embeddings using `intfloat/e5-small-v2`
- Vector Store: Pinecone (serverless) for historical document retrieval
- Live Data: Real-time financial news ingestion (Marketaux, GNews)
- LLM Layer: Free-tier safe multi-LLM routing via OpenRouter
- Fallbacks: Automatic degradation to local Ollama when APIs are rate-limited

## Key Design Principles

- Evidence-first answers (no hallucinated market reasons)
- Clear separation between live signals and historical context
- Cost-efficient and free-tier friendly by default
- Production-ready RAG orchestration using LangChain LCEL

## High-Level Flow

1. User asks a natural language market question
2. Query is embedded locally and matched against Pinecone
3. Latest financial news is fetched in parallel
4. Context is injected into a structured analyst prompt
5. LLM generates a grounded, explainable answer

## Status

Backend RAG pipeline implemented.
Frontend and deployment layers coming next.
