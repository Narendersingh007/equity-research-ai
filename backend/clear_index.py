import os
import time
from pinecone import Pinecone
from dotenv import load_dotenv


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

def clear_index():
    if not PINECONE_API_KEY or not INDEX_NAME:
        print("❌ Error: Missing Pinecone credentials in .env")
        return

    print(f" Connecting to Pinecone Index: {INDEX_NAME}...")
    

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)


    stats = index.describe_index_stats()
    count = stats.get('total_vector_count', 0)
    print(f"📉 Current Vector Count: {count}")

    if count == 0:
        print("Index is already empty.")
        return


    confirm = input(f"  WARNING: You are about to DELETE ALL {count} vectors. Type 'DELETE' to confirm: ")
    
    if confirm == "DELETE":
        print("🔥 Deleting all vectors... (This may take a few seconds)")
        index.delete(delete_all=True)
        
        # Wait for consistency
        time.sleep(5)
        
        # Verify
        stats = index.describe_index_stats()
        new_count = stats.get('total_vector_count', 0)
        print(f"✨ Done! New Vector Count: {new_count}")
    else:
        print(" Operation cancelled.")

if __name__ == "__main__":
    clear_index()
