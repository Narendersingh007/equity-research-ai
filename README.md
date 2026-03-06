# ERA — Equity Research AI Agent

ERA is a **production-grade Retrieval-Augmented Generation (RAG) system** designed to perform **evidence-grounded equity research** on the **top 30 globally traded public companies**.

Unlike generic LLM chatbots that rely on static, pre-trained knowledge, ERA retrieves and reasons over **primary financial disclosures and live market signals**, producing concise, analyst-style research briefs with explicit source grounding.


## Executive Summary

ERA is a production-oriented Retrieval-Augmented Generation (RAG) system for equity research on the top 30 global public companies.

It retrieves evidence from SEC 10-K filings and live market news, reasons over that context using a fault-tolerant multi-LLM setup, and produces analyst-style research briefs with explicit source grounding.

Key characteristics:
- Retrieval-first, evidence-grounded reasoning (no hallucinated analysis)
- 80K+ semantic vectors indexed from SEC filings
- Multi-provider LLM orchestration with automatic fallback
- Asynchronous FastAPI backend for low-latency inference

---

Table of Contents

1. Executive Summary
2. Problem Statement
3. Intended Use & Audience
4. What ERA Is Not
5. System Overview & Architecture
6. Data Ingestion & Knowledge Indexing
7. End-to-End Query Flow
8. Example Queries
9. Key Engineering Decisions & Trade-offs
10. Technology Stack
11. Reliability, Cost & Free-Tier Operation
12. Limitations & Known Constraints
13. Future Enhancements & Roadmap
---

## Problem Statement

### Unreliable Financial Analysis from Static LLMs
Pre-trained language models lack access to up-to-date SEC filings and real-time market developments, making them unsuitable for equity research that depends on current, verifiable information.

### Hallucination & Missing Source Traceability
When applied to dense regulatory documents, LLMs often generate confident but unsupported insights, with no clear linkage to the underlying disclosures—posing significant risk in analytical workflows.

### Fragile Single-Model Architectures
Most LLM-driven systems rely on a single provider, making them vulnerable to downtime, latency spikes, and quota limits, undermining reliability in real-world usage.

ERA addresses these challenges through a **retrieval-first, fault-tolerant architecture** that grounds every response in verifiable data while maintaining high availability.

---

## Intended Use & Audience

ERA is designed for:
- Students and engineers exploring production-grade RAG architectures
- Developers evaluating evidence-grounded LLM systems
- Recruiters reviewing system design, fault tolerance, and GenAI engineering depth

ERA is **not intended** for:
- Financial advice or trading decisions
- High-frequency or quantitative trading workflows

---

## What ERA Is Not

- Not a trading or investment recommendation system
- Not a price prediction or quantitative modeling tool
- Not a conversational chatbot with long-term memory

---

## System Overview & Architecture

ERA follows a **Retrieve → Verify → Reason → Respond** paradigm, closely mirroring how professional equity analysts combine historical disclosures with recent market developments.

### Architectural Highlights

- **Retrieval-First Reasoning**  
  Every user query is resolved against a semantic index of SEC 10-K filings before generation, ensuring responses are grounded in source documents rather than model memory.

- **Multi-Source Context Assembly**  
  Each response fuses:
  - Historical disclosures retrieved from SEC filings  
  - Live market context from Google News RSS summaries  
  This enables time-aware analysis without brittle web scraping pipelines.

- **Fault-Tolerant LLM Orchestration**  
  Generation is handled by a resilient multi-provider setup:
  - Primary: Llama-3.3-70B via Groq  
  - Cloud fallbacks: Gemini and Mistral via OpenRouter  
  - Local fallback: Phi-3 via Ollama  
  Automatic failover ensures graceful degradation under provider limits or outages.

- **Hallucination Guardrails**  
  Prompt-level constraints strictly confine generation to retrieved context, enforce uncertainty disclosure, and maintain a neutral analyst tone—positioning the LLM as a reasoning engine rather than a knowledge source.

- **Async, Composable Backend**  
  The pipeline is composed using LangChain Runnables and exposed via an asynchronous FastAPI backend, supporting concurrent requests and clean API boundaries.

---

## Data Ingestion & Knowledge Indexing

ERA’s knowledge layer is built from **primary-source financial disclosures**, engineered to maximize retrieval precision and semantic fidelity.

- **Automated SEC Filing Acquisition**  
  The system programmatically downloads the latest SEC 10-K filings via the EDGAR API, storing them in a normalized text format aligned by ticker and fiscal year.

- **Aggressive Cleaning & Normalization**  
  Raw filings are sanitized to remove XBRL/XML blocks, embedded tables, scripts, and boilerplate artifacts while preserving narrative sections such as Risk Factors and MD&A.

- **Sliding-Window Chunking & Embeddings**  
  Cleaned documents are chunked using a sliding-window strategy and embedded with `intfloat/e5-small-v2`, producing **80K+ normalized dense vectors** that capture semantic meaning rather than keyword overlap.

- **Metadata-Enriched Vector Indexing**  
  All embeddings are indexed in Pinecone with structured metadata (ticker, filing type, fiscal year, source), enabling traceable and explainable retrieval during analysis.

This ingestion pipeline transforms raw regulatory filings into a **production-ready semantic knowledge base**, forming the backbone of ERA’s retrieval-first reasoning architecture.

---

## End-to-End Query Flow

ERA processes each user query through a deterministic, retrieval-first pipeline designed for evidence-grounded analysis.

1. **Query Intake**
   - A user submits a natural language question (e.g., risk exposure, supply chain impact, recent developments) via the API or frontend.
   - The FastAPI backend acts as the orchestration layer.

2. **Semantic Query Encoding**
   - The query is transformed into a dense vector using the `intfloat/e5-small-v2` embedding model.
   - Embeddings are normalized to ensure consistent cosine similarity during retrieval.

3. **Historical Context Retrieval**
   - The encoded query is matched against the Pinecone vector index containing SEC 10-K chunks.
   - Top-K semantically relevant passages are retrieved using similarity search, independent of keyword overlap.

4. **Live Market Context Injection**
   - Query keywords are extracted and used to fetch recent market developments via Google News RSS.
   - The system aggregates the top 10 summaries to provide time-aware context without direct web scraping.

5. **Context Assembly**
   - Retrieved SEC excerpts and live news summaries are combined into a structured context payload.
   - Context size is intentionally constrained to fit within LLM token limits while preserving relevance.

6. **LLM Reasoning with Guardrails**
   - The assembled context is injected into a strict analyst-style prompt.
   - The primary LLM (Llama-3.3-70B via Groq) generates a response using **only** the provided evidence.

7. **Automatic Fallback Handling**
   - If the primary model fails (rate limits, timeouts), execution transparently falls back to:
     - Gemini or Mistral (OpenRouter)
     - Local Phi-3 (Ollama) as a last resort
   - This ensures high availability and graceful degradation.

8. **Source Attribution & Response Delivery**
   - SEC filings and news sources used during retrieval are extracted and attached to the response.
   - The final output is returned as a structured research brief with processing latency metadata.

---

## Example Queries

- “What supply chain risks does NVIDIA disclose in its latest 10-K?”
- “What regulatory risks impact Meta’s advertising business?”
- “Summarize recent developments affecting Tesla’s manufacturing outlook.”
- “What macroeconomic risks does Amazon highlight in its filings?”

ERA responds with concise, analyst-style research briefs grounded in SEC disclosures and live market context.

---

## Key Engineering Decisions & Trade-offs

ERA’s architecture reflects deliberate trade-offs between accuracy, latency, cost, and operational reliability.

### Retrieval-Augmented Generation over Fine-Tuning
A retrieval-first RAG design was chosen over fine-tuning to ensure access to up-to-date SEC filings and live news, preserve source traceability, and avoid retraining overhead.

### Dense Semantic Search over Keyword Search
Dense embeddings (`intfloat/e5-small-v2`) enable concept-level retrieval across varied financial language, outperforming keyword-based search in long-form disclosures.

### Sliding-Window Chunking
Overlapping sliding-window chunking preserves cross-sentence context in dense sections (e.g., Risk Factors, MD&A), reducing semantic fragmentation during retrieval.

### Multi-Provider LLM Orchestration
A prioritized fallback chain (Groq → OpenRouter → Ollama) trades minimal architectural complexity for high availability, cost control, and resilience to provider outages.

### Lightweight Live News Integration
Google News RSS provides near-real-time market context with low operational overhead, avoiding brittle scraping pipelines and paid data dependencies.

### Prompt-Level Hallucination Guardrails
Strict system prompts constrain generation to retrieved evidence and require uncertainty disclosure, positioning the LLM as a reasoning layer rather than a knowledge source.

---

## Technology Stack

ERA is built using a modular, production-oriented stack optimized for retrieval accuracy, low latency, and fault tolerance.

- **Backend & API**: Python, FastAPI (async, concurrent request handling)
- **RAG Orchestration**: LangChain (RunnableParallel, RunnableLambda, RunnablePassthrough)
- **Vector Database**: Pinecone (serverless, cosine similarity search)
- **Embeddings**: Sentence-Transformers (`intfloat/e5-small-v2`, normalized vectors)
- **LLM Inference**:
  - Primary: Llama-3.3-70B (Groq)
  - Cloud fallbacks: Gemini, Mistral (OpenRouter)
  - Local fallback: Phi-3 (Ollama)
- **Data Sources**:
  - SEC EDGAR (10-K filings)
  - Google News RSS (live market context)
- **Infrastructure**: Environment-based config, stateless backend, serverless-friendly design

---

## Reliability, Cost & Free-Tier Operation

ERA is engineered to remain **continuously operational with zero mandatory infrastructure cost**, leveraging free-tier limits and graceful degradation.

- **Zero fixed backend cost**: The system relies on managed, free-tier–friendly services (Groq, OpenRouter, Google News RSS) and avoids always-on compute instances.
- **Fault-tolerant inference**: A prioritized multi-LLM fallback chain (Groq → Gemini/Mistral → local Ollama) ensures continued operation under rate limits, API outages, or quota exhaustion.
- **Token-efficient design**: Retrieval-first architecture strictly limits LLM context to top-K SEC excerpts and news summaries, minimizing token usage and avoiding unnecessary inference cost.
- **Serverless-compatible backend**: Stateless FastAPI design enables deployment on serverless platforms or free hosting tiers without background workers or persistent services.
- **No paid data dependencies**: Live market awareness is sourced via Google News RSS rather than paid financial APIs or brittle scraping pipelines.
- **Graceful degradation**: When external services are unavailable, ERA degrades functionality predictably instead of failing silently or hallucinating outputs.

---

## Limitations & Known Constraints

ERA is designed for correctness and reliability, but it intentionally makes several trade-offs:

- **Scope-limited coverage**: The system currently indexes SEC 10-K filings for the top 30 global public companies only; coverage does not yet include 10-Qs, earnings calls, or non-US disclosures.
- **Latency variability under free tiers**: Inference latency may fluctuate due to shared, rate-limited free-tier LLM providers, especially during peak usage.
- **No fine-grained financial metrics extraction**: ERA focuses on qualitative analysis (risk factors, disclosures, narratives) rather than precise numerical modeling or valuation.
- **RSS-based news summarization**: Live market context relies on Google News RSS summaries, which may omit paywalled or niche analyst reports.
- **Stateless query execution**: The system does not maintain conversational memory across queries; each request is processed independently for determinism and cost control.

---

## Future Enhancements & Roadmap

- Conversational memory for multi-step, analyst-style research threads
- Full-featured interactive frontend (citations, comparisons, latency insights)
- Always-on, serverless deployment for continuous availability
- Optional premium LLMs for deeper reasoning and larger context windows
- Cross-company and cross-document comparative analysis
- Expanded coverage beyond 10-Ks (10-Qs, earnings calls, global filings)
- Intelligent follow-up suggestions and automated risk flags
- Evaluation pipelines for retrieval accuracy and response faithfulness
