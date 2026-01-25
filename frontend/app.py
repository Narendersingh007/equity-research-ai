import streamlit as st
import pandas as pd
import sys
import os
import time
import requests
import plotly.express as px

# --------------------------------------------------
# PATH SETUP (Allows backend imports)
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.search_logic import RAGPipeline
except ImportError:
    st.error(
        "⚠️ Backend not found.\n\n"
        "Run from project root:\n"
        "`streamlit run frontend/app.py`"
    )
    st.stop()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Equity Research AI Agent",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------
# GLOBAL STYLING (Dark, Institutional)
# --------------------------------------------------
st.markdown(
    """
<style>
body {
    background-color: #0E1117;
}
.stTextInput > div > div > input {
    background-color: #1E1E1E;
    color: white;
}
.ai-answer {
    background-color: #161A23;
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
    margin-top: 1rem;
}
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 1.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📈 Equity Research AI Agent")
st.caption(
    "Live RAG Analysis · SEC Filings (10-K) · Market News Intelligence"
)

st.divider()

# --------------------------------------------------
# SIDEBAR — SYSTEM CONTEXT
# --------------------------------------------------
with st.sidebar:
    st.header("🧠 Neural Engine")

    @st.cache_resource
    def load_pipeline():
        return RAGPipeline()

    with st.spinner("Initializing models..."):
        try:
            rag = load_pipeline()
            st.success("System Online")
        except Exception as e:
            st.error(f"Initialization failed:\n{e}")
            st.stop()

    st.divider()

    st.markdown("**Vector Index**")
    st.code(os.getenv("INDEX_NAME", "finance-news"))

    st.markdown("**Retrieval Strategy**")
    st.text("Top-10 semantic chunks")

    st.divider()

    st.markdown("### 📚 Coverage Universe")
    st.text(
        "NVDA\nAAPL\nMSFT\nAMZN\nTSLA\nMETA\n+24 more"
    )

# --------------------------------------------------
# QUERY INPUT
# --------------------------------------------------
st.subheader("🤖 Ask the Analyst")

query = st.text_input(
    label="Query",
    placeholder="e.g. What are Nvidia’s primary supply chain risks?",
    label_visibility="collapsed",
)

st.markdown(
    """
**Suggested questions**
- What macro risks affect Big Tech in 2025?
- Compare Apple and Microsoft revenue drivers
- What are Nvidia’s key operational risks?
"""
)

# --------------------------------------------------
# RAG RESPONSE
# --------------------------------------------------
if query:
    with st.spinner("Analyzing filings and market news..."):
        start = time.time()

        try:
            response = rag.get_answer(query)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

        latency = time.time() - start

    st.markdown(f"⏱️ _Generated in {latency:.2f} seconds_")

    st.markdown(
        f"""
<div class="ai-answer">
<strong>Equity Research Insight</strong><br><br>
{response}
</div>
""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# MARKET INTELLIGENCE DASHBOARD
# --------------------------------------------------
st.divider()
st.subheader("📊 Market Intelligence")

DATA_PATH = "backend/data/market_data.json"

@st.cache_data
def load_market_data(path):
    return pd.read_json(path)

if os.path.exists(DATA_PATH):
    try:
        df = load_market_data(DATA_PATH)

        if "id" in df.columns:
            df = df.rename(columns={"id": "ticker"})

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tracked Companies", len(df))
        c2.metric("Avg P/E Ratio", f"{df['pe_ratio'].mean():.2f}")
        c3.metric("High Volatility (>1.5β)", df[df["beta"] > 1.5].shape[0])
        c4.metric("Sector Spread", df["sector"].nunique())

        st.markdown("### 🗺️ Sector Heatmap")

        # Data safety
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(0)
        df["change_percent"] = pd.to_numeric(df["change_percent"], errors="coerce").fillna(0)

        fig = px.treemap(
            df,
            path=[px.Constant("Market"), "sector", "ticker"],
            values="market_cap",
            color="change_percent",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            hover_data={
                "price": True,
                "pe_ratio": True,
                "market_cap": ":.2e",
            },
        )

        fig.update_layout(
            margin=dict(t=0, l=0, r=0, b=0),
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Monospace"),
        )

        st.plotly_chart(fig, width="stretch")

    except Exception as e:
        st.error(f"Market dashboard error: {e}")

else:
    st.warning(
        "Market data not found.\n\n"
        "Run `python backend/update_prices.py` to generate it."
    )