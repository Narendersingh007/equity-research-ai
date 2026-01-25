import os
import json
import shutil
from sec_edgar_downloader import Downloader

# --- CONFIG ---
EMAIL = "your.email@example.com"  # Replace with a valid email (User-Agent requirement)
BASE_DIR = "backend/data/downloads"
FINAL_DIR = "backend/data/reports"
MARKET_DATA_PATH = "backend/data/market_data.json"

def get_tickers_from_json():
    """Reads the market_data.json file to get the live list of 30 companies."""
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"⚠️ Market data not found at {MARKET_DATA_PATH}. Using fallback list.")
        # Fallback if file is missing
        return ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "INTC", "PLTR"]
    
    try:
        with open(MARKET_DATA_PATH, "r") as f:
            data = json.load(f)
            # Extract tickers (supports 'id' or 'ticker' keys)
            tickers = [item.get("id", item.get("ticker")) for item in data]
            # Remove duplicates and None values
            unique_tickers = list(set([t for t in tickers if t]))
            print(f"📋 Loaded {len(unique_tickers)} tickers from market_data.json")
            return unique_tickers
    except Exception as e:
        print(f"❌ Error reading market data: {e}")
        return []

def download_10k():
    # 1. Get the list of tickers
    tickers = get_tickers_from_json()
    
    if not tickers:
        print("❌ No tickers found to download.")
        return

    # 2. Initialize Downloader
    dl = Downloader("MyCompany", EMAIL, BASE_DIR)
    
    # 3. Create destination folder if not exists
    if not os.path.exists(FINAL_DIR):
        os.makedirs(FINAL_DIR)

    # 4. Loop through all tickers
    for ticker in tickers:
        print(f"⬇️  Fetching 10-K for {ticker}...")
        try:
            # Download the latest 10-K (limit=1)
            num_downloaded = dl.get("10-K", ticker, limit=1)
            
            if num_downloaded == 0:
                print(f"⚠️  No 10-K found for {ticker}")
                continue

            # Move and Rename Logic
            # Structure: backend/data/downloads/sec-edgar-filings/{ticker}/10-K/{filing_id}/full-submission.txt
            ticker_path = os.path.join(BASE_DIR, "sec-edgar-filings", ticker, "10-K")
            
            # Find the downloaded folder (it uses a random filing ID)
            if os.path.exists(ticker_path):
                for folder in os.listdir(ticker_path):
                    full_path = os.path.join(ticker_path, folder, "full-submission.txt")
                    
                    if os.path.exists(full_path):
                        # Save as .txt (Universal format)
                        final_name = f"{ticker}_2024.txt"
                        destination = os.path.join(FINAL_DIR, final_name)
                        
                        shutil.copy(full_path, destination)
                        print(f"✅ Saved {final_name}")
                        break # Only need the first one found
            
        except Exception as e:
            print(f"❌ Failed to download {ticker}: {e}")

    print("\n🎉 Bulk Download Complete!")

if __name__ == "__main__":
    download_10k()