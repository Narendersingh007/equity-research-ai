import yfinance as yf
import json
import os
from datetime import datetime

# Absolute base directory (repo root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directory and output file
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "market_data.json")

# The "Volatile 30" List
TICKERS = [
    "TSLA", "NVDA", "COIN", "MSTR", "PLTR", "AMD", "NFLX", "META", "SHOP", "CRWD",
    "AMZN", "GOOGL", "MSFT", "AAPL", "CRM", "UBER", "ABNB", "SNOW", "RBLX", "SPOT",
    "AVGO", "INTC", "MRNA", "PFE", "DIS", "BA", "PYPL", "SQ", "ENPH", "RIVN"
]

def update_market_data():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching market data...")

    market_data = []

    tickers_str = " ".join(TICKERS)
    tickers_data = yf.Tickers(tickers_str)

    for symbol in TICKERS:
        try:
            info = tickers_data.tickers[symbol].info

            company = {
                "id": symbol,
                "name": info.get("shortName", symbol),
                "price": info.get("currentPrice", 0.0),
                "change_percent": info.get("regularMarketChangePercent", 0.0),
                "market_cap": info.get("marketCap", 0),
                "beta": info.get("beta", 0.0),
                "pe_ratio": info.get("trailingPE", 0.0),
                "sector": info.get("sector", "N/A"),
                "last_updated": datetime.utcnow().isoformat()
            }

            market_data.append(company)
            print(f"✓ {symbol}")

        except Exception as e:
            print(f"✗ {symbol}: {e}")

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Write JSON
    with open(DATA_PATH, "w") as f:
        json.dump(market_data, f, indent=2)

    print(f"Data written to {DATA_PATH}")

if __name__ == "__main__":
    update_market_data()