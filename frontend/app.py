import streamlit as st
import pandas as pd
import sys
import os
import time

# --- PATH SETUP (Crucial for importing backend) ---
# This allows us to import from the 'backend' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from backend.search_logic import RAGPipeline
except ImportError:
    st.error("⚠️ Could not import Backend. Please run the app from the root directory: `streamlit run frontend/app.py`")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(
    layout="wide", 
    page_title="Equity Research AI",
    page_icon="📈"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30333F;
        text-align: center;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white; 
    }
    .ai-answer {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📈 Equity Research AI Agent")
st.markdown("**Live Market Data • RAG Analysis • SEC Filings (10-K)**")

# --- SIDEBAR: AI BRAIN CONTROL ---
with st.sidebar:
    st.header("⚙️ Neural Engine")
    
    # Initialize Pipeline (Cached so it doesn't reload on every click)
    @st.cache_resource
    def load_pipeline():
        return RAGPipeline()
    
    with st.spinner("🧠 Loading AI Models..."):
        try:
            rag = load_pipeline()
            st.success("✅ System Online")
        except Exception as e:
            st.error(f"❌ System Offline: {e}")
            st.stop()
            
    st.markdown("---")
    st.info(f"**Connected Index:** `{os.getenv('INDEX_NAME', 'finance-news')}`")
    st.markdown("### 📚 Ingested Knowledge")
    st.code("NVDA (2024 10-K)\nTSLA (Coming Soon)\nAAPL (Coming Soon)")

# --- MAIN SEARCH AREA ---
col_search, col_stats = st.columns([2, 1])

with col_search:
    st.subheader("🤖 Ask the Analyst")
    query = st.text_input(
        "Query", 
        placeholder="e.g., 'What are Nvidia's specific supply chain risks?'",
        label_visibility="collapsed"
    )

    if query:
        st.write("---")
        with st.spinner("🔎 Reading 10-K Reports & Checking Live News..."):
            # 1. Timer Start
            start_time = time.time()
            
            # 2. Get Answer from Backend
            response = rag.get_answer(query)
            
            # 3. Calculate Speed
            duration = time.time() - start_time
            
        # 4. Display Result
        st.markdown(f"**⏱️ Analysis generated in {duration:.2f} seconds**")
        st.markdown(f"""
        <div class="ai-answer">
            {response}
        </div>
        """, unsafe_allow_html=True)

# --- MARKET DATA DASHBOARD ---
st.markdown("---")
st.subheader("📊 Market Intelligence")

data_path = "backend/data/market_data.json"
if os.path.exists(data_path):
    try:
        # 1. Load Data
        df = pd.read_json(data_path)
        if 'id' in df.columns: df = df.rename(columns={'id': 'ticker'})
        
        # 2. Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tracked Companies", len(df))
        m2.metric("Avg P/E Ratio", f"{df['pe_ratio'].mean():.2f}")
        m3.metric("High Volatility (>1.5)", df[df['beta'] > 1.5].shape[0])
        m4.metric("Sector Spread", df['sector'].nunique())

        # 3. NATIVE PLOTLY HEATMAP (Browser-Safe)
        st.markdown("### 🗺️ Sector Heatmap")
        
        import plotly.express as px
        
        # Ensure we have numeric data for the chart
        df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce').fillna(0)
        df['change_percent'] = pd.to_numeric(df['change_percent'], errors='coerce').fillna(0)

        # Create the Tree Map
        fig = px.treemap(
            df,
            path=[px.Constant("Market"), 'sector', 'ticker'], # Hierarchy: Market -> Sector -> Company
            values='market_cap',                              # Size of block = Market Cap
            color='change_percent',                           # Color of block = Price Change
            color_continuous_scale='RdYlGn',                  # Red to Green
            color_continuous_midpoint=0,                      # Zero is the center (Yellow/White)
            hover_data={'price': True, 'pe_ratio': True, 'market_cap': ':.2e'},
            title=''
        )
        
        # Style it to look like a Dark Mode Terminal
        fig.update_layout(
            margin=dict(t=0, l=0, r=0, b=0),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Monospace", color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
else:
    st.warning("⚠️ Market data file not found. Run `python backend/update_prices.py` to populate this table.")