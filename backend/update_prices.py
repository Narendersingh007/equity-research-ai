import yfinance as yf
import json
import os
from datetime import datetime

# The "Volatile 30" List
TICKERS = [
    "TSLA", "NVDA", "COIN", "MSTR", "PLTR", "AMD", "NFLX", "META", "SHOP", "CRWD",
    "AMZN", "GOOGL", "MSFT", "AAPL", "CRM", "UBER", "ABNB", "SNOW", "RBLX", "SPOT",
    "AVGO", "INTC", "MRNA", "PFE", "DIS", "BA", "PYPL", "SQ", "ENPH", "RIVN"
]

def update_market_data():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Fetching data for 30 companies...")
    
    market_data = []
    
    # Download all at once (Much faster)
    tickers_str = " ".join(TICKERS)
    tickers_data = yf.Tickers(tickers_str)
    
    for symbol in TICKERS:
        try:
            info = tickers_data.tickers[symbol].info
            
            # We construct a clean dictionary for the frontend
            company_obj = {
                "id": symbol,
                "name": info.get("shortName", symbol),
                "price": info.get("currentPrice", 0.0),
                "change_percent": info.get("regularMarketChangePercent", 0.0) * 100, # Convert to %
                "market_cap": info.get("marketCap", 0),
                "beta": info.get("beta", 0.0), # Volatility metric
                "pe_ratio": info.get("trailingPE", 0.0),
                "sector": info.get("sector", "N/A"),
                "last_updated": datetime.now().isoformat()
            }
            market_data.append(company_obj)
            print(f"✅ {symbol} fetched")
            
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

    # SAVE THE DATA
    # We save this in 'data/market_data.json' relative to where you run the script
    # Ensure the folder exists
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/market_data.json'
    with open(output_path, 'w') as f:
        json.dump(market_data, f, indent=2)
        
    print(f"✨ Success! Data saved to {output_path}")

if __name__ == "__main__":
    update_market_data()