import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MARKETAUX_API_KEY")
if not API_KEY:
    raise RuntimeError("MARKETAUX_API_KEY not set")

query = "Tesla OR TSLA"
from_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

url = (
    "https://api.marketaux.com/v1/news/all?"
    f"search={query}"
    f"&published_after={from_date}"
    "&language=en"
    "&limit=5"
    f"&api_token={API_KEY}"
)

response = requests.get(url, timeout=10)
data = response.json()

print("\n🔎 Marketaux API Test (with descriptions)")
print("-" * 50)

articles = data.get("data", [])
if not articles:
    print("No articles found.")
else:
    for i, art in enumerate(articles, 1):
        description = art.get("description") or art.get("snippet") or "No description available."

        print(f"\n{i}. {art['title']}")
        print(f"   Description : {description}")
        print(f"   Source      : {art['source']}")
        print(f"   Published   : {art['published_at']}")
        print(f"   URL         : {art['url']}")