import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Securely grab the key from the environment
# If it's not found in .env, it falls back to None or a placeholder
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# Core application paths
DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"
MEMORY_FILE_PATH = "long_term_memory.json"

# Operational models - Using the optimal v1beta native embedding model
EMBEDDING_MODEL = "gemini-embedding-2-preview"
LLM_MODEL = "gemini-2.5-flash"