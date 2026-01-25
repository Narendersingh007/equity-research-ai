import streamlit as st
import sys
import os
import time
import json


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.search_logic import RAGPipeline
except ImportError:
    st.error(
        "Backend module not found.\n\n"
        "Please run the application from the project root:\n"
        "streamlit run frontend/app.py"
    )
    st.stop()


st.set_page_config(
    page_title="Equity Research AI Agent",
    layout="wide",
)


st.markdown(
    """
<style>
body {
    background-color: #0E1117;
}

html, body, [class*="css"] {
    font-size: 17px;
}

/* Headings */
h1 {
    font-size: 2.5rem;
    font-weight: 700;
}
h2 {
    font-size: 1.8rem;
    font-weight: 600;
}
h3 {
    font-size: 1.4rem;
    font-weight: 600;
}

/* Text input */
.stTextInput input {
    background-color: #1E1E1E;
    color: white;
    font-size: 16px;
    padding: 12px;
}

/* Answer container */
.ai-answer {
    background-color: #161A23;
    padding: 28px;
    border-radius: 12px;
    border-left: 4px solid #4CAF50;
    margin-top: 1.5rem;
    font-size: 16px;
    line-height: 1.75;
}

/* Sidebar spacing */
section[data-testid="stSidebar"] {
    padding-top: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------
st.title("Equity Research AI Agent")
st.caption("AI-driven equity research grounded in SEC filings and market news")
st.divider()


st.markdown(
    """
## Project Overview

The **Equity Research AI Agent** is an intelligent research system designed to assist
financial analysts, students, and investors in extracting high-quality insights from
regulatory filings and financial news.

The application leverages **Retrieval-Augmented Generation (RAG)** to ensure that
responses are:

- Grounded in primary sources such as SEC 10-K filings
- Contextually accurate and traceable
- Written in a concise, analyst-style format

Instead of manually reviewing lengthy financial documents, users can ask natural
language questions and receive structured, source-aware responses in seconds.

This project emphasizes **reliability, transparency, and real-world financial analysis**
over generic conversational output.
"""
)

st.divider()


with st.sidebar:
    st.header("System Status")

    @st.cache_resource
    def initialize_pipeline():
        return RAGPipeline()

    with st.spinner("Initializing models and vector index..."):
        try:
            rag = initialize_pipeline()
            st.success("System initialized successfully")
        except Exception as e:
            st.error(f"Initialization failed:\n{e}")
            st.stop()

    st.divider()

    st.markdown("**Vector Index**")
    st.code(os.getenv("INDEX_NAME", "finance-news"))

    st.markdown("**Retrieval Configuration**")
    st.text("Top-10 semantic chunks")

    st.divider()

    st.markdown("**Coverage Universe**")
    st.text(
        "NVDA\n"
        "AAPL\n"
        "MSFT\n"
        "AMZN\n"
        "TSLA\n"
        "META\n"
        "+ additional large-cap equities"
    )


st.subheader("Research Query")

query = st.text_input(
    label="Query",
    placeholder="Example: What operational risks does Apple disclose in its latest 10-K?",
    label_visibility="collapsed",
)


if query:
    with st.spinner("Analyzing filings and market news..."):
        start_time = time.time()

        try:
            raw_response = rag.get_answer(query)

            answer_text = ""
            sources = {}

            if isinstance(raw_response, dict):
                answer_text = raw_response.get("answer", "")
                sources = raw_response.get("sources", {})
            elif isinstance(raw_response, str):
                try:
                    parsed = json.loads(raw_response)
                    answer_text = parsed.get("answer", raw_response)
                    sources = parsed.get("sources", {})
                except Exception:
                    answer_text = raw_response
            else:
                answer_text = str(raw_response)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

        latency = time.time() - start_time

    st.markdown(f"_Response generated in {latency:.2f} seconds_")

    st.markdown(
        f"""
<div class="ai-answer">
<strong>Equity Research Insight</strong><br><br>
{answer_text}
</div>
""",
        unsafe_allow_html=True,
    )

    if sources:
        with st.expander("Source Attribution"):
            if sources.get("sec"):
                st.markdown("**SEC Filings**")
                for s in sources["sec"]:
                    st.markdown(f"- {s}")

            if sources.get("news"):
                st.markdown("**Market News**")
                for n in sources["news"]:
                    st.markdown(f"- {n}")