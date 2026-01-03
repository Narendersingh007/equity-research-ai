import os
import sys
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def validate_environment():
    """
    Validates that all required environment variables are set.
    Exits with status code 1 if configuration is incomplete.
    """
    load_dotenv()
    
    required_keys = ["PINECONE_API_KEY", "GOOGLE_API_KEY", "INDEX_NAME"]
    missing_keys = []

    logger.info("Validating project configuration...")

    for key in required_keys:
        value = os.getenv(key)
        
        if not value:
            missing_keys.append(key)
        else:
          
            display_value = f"{value[:4]}..." if "KEY" in key else value
            logger.info(f"Loaded {key}: {display_value}")

    if missing_keys:
        logger.error(f"Configuration failed. Missing variables: {', '.join(missing_keys)}")
        sys.exit(1)

    logger.info("Environment configuration valid.")

if __name__ == "__main__":
    try:
        # verify imports
        from google import genai
        from pinecone import Pinecone
        logger.info("Dependencies imported successfully.")
        
        validate_environment()
        
    except ImportError as e:
        logger.critical(f"Import Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected Error: {e}")
        sys.exit(1)