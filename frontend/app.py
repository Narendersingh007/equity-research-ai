import streamlit as st
import json
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 1. SETUP PATHS & ENV ---
# Get the root directory (one level up from frontend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CORRECTED: Load .env from the 'backend' folder
env_path = os.path.join(BASE_DIR, "backend", ".env")
load_dotenv(env_path, override=True)

# Debug: Print to terminal to confirm it found the file
print(f"Loading env from: {env_path}")
print(f"API Key found: {bool(os.getenv('PINECONE_API_KEY'))}")

# --- CONFIGURATION ---
ST_PAGE_TITLE = "Volatile 30 Tracker"
# Point to data/market_data.json in the root (created by update_prices.py)
DATA_FILE = os.path.join(BASE_DIR, "data", "market_data.json")
INDEX_NAME = os.getenv("INDEX_NAME")
EMBEDDING_MODEL = "intfloat/e5-small-v2"


st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide")

# --- 1. LOAD MARKET DATA (JSON) ---
@st.cache_data
def load_market_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

# --- 2. SETUP SEARCH ENGINE (PINECONE) ---
@st.cache_resource
def get_vectorstore():
    # Only load if user asks a question to save resources
    if not os.getenv("PINECONE_API_KEY"):
        st.error("Missing PINECONE_API_KEY in .env")
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )
    return vectorstore

# --- UI LAYOUT ---
st.title(f"⚡ {ST_PAGE_TITLE}")
st.markdown("Monitor the world's most volatile companies and ask semantic questions about their risks.")

# Load Data
companies = load_market_data()

if not companies:
    st.error("No data found! Run 'python backend/update_prices.py' first.")
else:
    # --- METRIC GRID ---
    # Display top 4 most volatile (highest Beta)
    sorted_companies = sorted(companies, key=lambda x: x['beta'], reverse=True)
    
    st.subheader("🔥 Top Movers (High Volatility)")
    cols = st.columns(4)
    for idx, company in enumerate(sorted_companies[:4]):
        with cols[idx]:
            st.metric(
                label=company['name'],
                value=f"${company['price']:.2f}",
                delta=f"{company['change_percent']:.2f}%"
            )

    st.divider()

    # --- MAIN INTERFACE: TWO COLUMNS ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select Company")
        selected_ticker = st.selectbox(
            "Choose a ticker to analyze:", 
            [c['id'] for c in companies]
        )
        
        # Show specific company stats
        company_info = next((c for c in companies if c['id'] == selected_ticker), None)
        if company_info:
            st.info(f"**Sector:** {company_info['sector']}")
            st.write(f"**Beta:** {company_info['beta']}")
            st.write(f"**P/E Ratio:** {company_info['pe_ratio']}")
            st.write(f"**Market Cap:** ${company_info['market_cap']:,}")

    with col2:
        st.subheader(f"🧠 AI Analysis: {selected_ticker}")
        
        query = st.text_input(f"Ask about {selected_ticker}'s long-term risks:", 
                             placeholder="e.g., What are the supply chain risks?")
        
        if query:
            if selected_ticker == "TSLA":
                # Connect to Pinecone for Real Search
                with st.spinner("Searching Pinecone..."):
                    vectorstore = get_vectorstore()
                    if vectorstore:
                        # Filter search specific to the selected company
                        results = vectorstore.similarity_search(
                            query, 
                            k=3,
                            filter={"ticker": selected_ticker}
                        )
                        
                        if results:
                            st.success("Found relevant insights:")
                            for doc in results:
                                with st.expander(f"Source: {doc.metadata.get('source', 'Unknown')}"):
                                    st.markdown(doc.page_content)
                        else:
                            st.warning("No information found for this specific query.")
            else:
                st.warning("⚠️ For this demo, only 'TSLA' has data uploaded. (Try selecting TSLA and asking about 'supply chain')")

# --- FOOTER ---
st.markdown("---")
st.caption("Data updated daily via 'update_prices.py' | Powered by Pinecone & HuggingFace")