import streamlit as st
import pandas as pd
import sys
import os
import time
import json
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.search_logic import RAGPipeline
except ImportError:
    st.error(
        "Backend not found.\n\n"
        "Run from project root:\n"
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
body { background-color: #0E1117; }
html, body, [class*="css"] { font-size: 16px; }

.stTextInput input {
    background-color: #1E1E1E;
    color: white;
}

.ai-answer {
    background-color: #161A23;
    padding: 22px;
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
    margin-top: 1rem;
    font-size: 15px;
    line-height: 1.7;
}

section[data-testid="stSidebar"] {
    padding-top: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Equity Research AI Agent")
st.caption("Live RAG analysis grounded in SEC filings and market news")
st.divider()

with st.sidebar:
    st.header("Neural Engine")

    @st.cache_resource
    def load_pipeline():
        return RAGPipeline()

    with st.spinner("Initializing models..."):
        try:
            rag = load_pipeline()
            st.success("System online")
        except Exception as e:
            st.error(f"Initialization failed:\n{e}")
            st.stop()

    st.divider()
    st.markdown("**Vector Index**")
    st.code(os.getenv("INDEX_NAME", "finance-news"))

    st.markdown("**Retrieval Strategy**")
    st.text("Top-10 semantic chunks")

    st.divider()
    st.markdown("**Coverage Universe**")
    st.text("NVDA\nAAPL\nMSFT\nAMZN\nTSLA\nMETA\n+24 more")

st.subheader("Ask the Analyst")

query = st.text_input(
    label="Query",
    placeholder="What supply chain risks does Apple disclose in its latest 10-K?",
    label_visibility="collapsed",
)

if query:
    with st.spinner("Analyzing filings and market news..."):
        start = time.time()
        try:
            raw = rag.get_answer(query)

            answer_text = ""
            sources = {}

            if isinstance(raw, dict):
                answer_text = raw.get("answer", "")
                sources = raw.get("sources", {})
            elif isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    answer_text = parsed.get("answer", raw)
                    sources = parsed.get("sources", {})
                except Exception:
                    answer_text = raw
            else:
                answer_text = str(raw)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

        latency = time.time() - start

    st.markdown(f"_Generated in {latency:.2f} seconds_")

    st.markdown(
        f"<div class='ai-answer'><strong>Equity Research Insight</strong><br><br>{answer_text}</div>",
        unsafe_allow_html=True,
    )

    if sources:
        with st.expander("Sources and attribution"):
            if sources.get("sec"):
                st.markdown("**SEC Filings**")
                for s in sources["sec"]:
                    st.markdown(f"- {s}")
            if sources.get("news"):
                st.markdown("**News**")
                for n in sources["news"]:
                    st.markdown(f"- {n}")

st.divider()
st.subheader("Market Intelligence")

DATA_PATH = "data/market_data.json"

if os.path.exists(DATA_PATH):
    df = pd.read_json(DATA_PATH)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked Companies", len(df))
    c2.metric("Average P/E", f"{df['pe_ratio'].mean():.2f}")
    c3.metric("High Volatility", df[df["beta"] > 1.5].shape[0])
    c4.metric("Sectors", df["sector"].nunique())

    fig = px.treemap(
        df,
        path=[px.Constant("Market"), "sector", "id"],
        values="market_cap",
        color="change_percent",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
    )

    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(t=0, l=0, r=0, b=0),
    )

    st.plotly_chart(fig, width="stretch")

else:
    st.warning(
        "Market data not found.\n\n"
        "This data is generated daily via GitHub Actions."
    )