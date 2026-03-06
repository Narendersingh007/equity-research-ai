import os
import json
import shutil
from sec_edgar_downloader import Downloader


EMAIL = "your.email@example.com"  
BASE_DIR = "backend/data/downloads"
FINAL_DIR = "backend/data/reports"
MARKET_DATA_PATH = "backend/data/market_data.json"

def get_tickers_from_json():
    """Reads the market_data.json file to get the live list of 30 companies."""
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"⚠️ Market data not found at {MARKET_DATA_PATH}. Using fallback list.")

        return ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "INTC", "PLTR"]
    
    try:
        with open(MARKET_DATA_PATH, "r") as f:
            data = json.load(f)

            tickers = [item.get("id", item.get("ticker")) for item in data]

            unique_tickers = list(set([t for t in tickers if t]))
            print(f"📋 Loaded {len(unique_tickers)} tickers from market_data.json")
            return unique_tickers
    except Exception as e:
        print(f"❌ Error reading market data: {e}")
        return []

def download_10k():

    tickers = get_tickers_from_json()
    
    if not tickers:
        print("❌ No tickers found to download.")
        return


    dl = Downloader("MyCompany", EMAIL, BASE_DIR)
    

    if not os.path.exists(FINAL_DIR):
        os.makedirs(FINAL_DIR)


    for ticker in tickers:
        print(f"⬇️  Fetching 10-K for {ticker}...")
        try:
            # Download the latest 10-K (limit=1)
            num_downloaded = dl.get("10-K", ticker, limit=1)
            
            if num_downloaded == 0:
                print(f"⚠️  No 10-K found for {ticker}")
                continue



            ticker_path = os.path.join(BASE_DIR, "sec-edgar-filings", ticker, "10-K")
            

            if os.path.exists(ticker_path):
                for folder in os.listdir(ticker_path):
                    full_path = os.path.join(ticker_path, folder, "full-submission.txt")
                    
                    if os.path.exists(full_path):

                        final_name = f"{ticker}_2024.txt"
                        destination = os.path.join(FINAL_DIR, final_name)
                        
                        shutil.copy(full_path, destination)
                        print(f"✅ Saved {final_name}")
                        break 
            
        except Exception as e:
            print(f"❌ Failed to download {ticker}: {e}")

    print("\n Bulk Download Complete!")

if __name__ == "__main__":
    download_10k()
