from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = None
index = None
pc = None
embedding_dimension = None
index_dimension = None

try:
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model loaded!")

    # Compatible with the installed SentenceTransformer versions.
    embedding_dimension = embedding_model.get_sentence_embedding_dimension()
    print("Embedding dimension:", embedding_dimension)

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is missing. Add it to your .env file."
        )

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.Index(
        PINECONE_INDEX_NAME
    )
    index_dimension = index.describe_index_stats().dimension

    if embedding_dimension > index_dimension:
        raise ValueError(
            f"Embedding model dimension {embedding_dimension} exceeds "
            f"Pinecone index dimension {index_dimension}."
        )

    print("Connected to Pinecone!")
    print("Index:", PINECONE_INDEX_NAME)
    print("Index dimension:", index_dimension)

except Exception as exc:
    print(f"\nWarning: vector database setup failed: {exc}\n")
    embedding_model = None if embedding_model is None else embedding_model
    index = None
    pc = None
    embedding_dimension = None
    index_dimension = None


def align_embedding(vector, target_dimension=None):
    """Pad model vectors with zeros to match the Pinecone index dimension."""
    target_dimension = target_dimension or index_dimension
    if target_dimension is None:
        raise RuntimeError("Pinecone index dimension is unavailable.")

    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    if len(values) > target_dimension:
        raise ValueError(
            f"Embedding dimension {len(values)} exceeds index dimension "
            f"{target_dimension}."
        )

    return values + [0.0] * (target_dimension - len(values))


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text: str):
    """
    Convert text into a 384-dimensional embedding vector.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    vector = embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return align_embedding(vector)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_text = "What is Agentic AI?"

    vector = create_embedding(test_text)

    print("Embedding created successfully!")
    print("Vector length:", len(vector))
    print("First 5 values:", vector[:5])