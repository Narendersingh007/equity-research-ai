import yfinance as yf
import json
import os
from datetime import datetime

# --------------------------------------------------
# Resolve repository root
# backend/update_prices.py -> repo root
# --------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --------------------------------------------------
# Streamlit-readable data location
# --------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "frontend", "data")
DATA_PATH = os.path.join(DATA_DIR, "market_data.json")

# --------------------------------------------------
# Volatile 30 Universe
# --------------------------------------------------
TICKERS = [
    "TSLA", "NVDA", "COIN", "MSTR", "PLTR", "AMD", "NFLX", "META", "SHOP", "CRWD",
    "AMZN", "GOOGL", "MSFT", "AAPL", "CRM", "UBER", "ABNB", "SNOW", "RBLX", "SPOT",
    "AVGO", "INTC", "MRNA", "PFE", "DIS", "BA", "PYPL", "SQ", "ENPH", "RIVN"
]

def update_market_data():
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Fetching market data")

    market_data = []

    tickers = yf.Tickers(" ".join(TICKERS))

    for symbol in TICKERS:
        try:
            info = tickers.tickers[symbol].info

            market_data.append({
                "id": symbol,
                "name": info.get("shortName", symbol),
                "price": info.get("currentPrice", 0.0),
                "change_percent": info.get("regularMarketChangePercent", 0.0),
                "market_cap": info.get("marketCap", 0),
                "beta": info.get("beta", 0.0),
                "pe_ratio": info.get("trailingPE", 0.0),
                "sector": info.get("sector", "N/A"),
                "last_updated": datetime.utcnow().isoformat()
            })

            print(f"✓ {symbol}")

        except Exception as e:
            print(f"✗ {symbol}: {e}")

    # Ensure frontend data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Write JSON for Streamlit
    with open(DATA_PATH, "w") as f:
        json.dump(market_data, f, indent=2)

    print(f"Market data written to {DATA_PATH}")

if __name__ == "__main__":
    update_market_data()