import os
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv("../.env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('RAG_USER', 'rag_user')}:"
    f"{os.getenv('RAG_USER_PASSWORD', 'ragpassword')}@"
    f"{os.getenv('POSTGRES_HOST', 'vectoria')}:5432/{os.getenv('POSTGRES_DB', 'vectordb')}"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
