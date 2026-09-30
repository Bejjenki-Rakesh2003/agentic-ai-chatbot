import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone
PINECONE_API_KEY = (os.getenv("PINECONE_API_KEY") or "").strip()

# Groq
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()

# Hugging Face
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

# Pinecone configuration
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-rag").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "agentic-ai").strip()


def missing_credentials() -> list[str]:
    """Return a list of required environment variables that are not configured."""
    missing = []

    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    return missing