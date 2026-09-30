from src.config import PINECONE_NAMESPACE
from src.embedding import create_embedding, embedding_model, index


# ============================================================
# RETRIEVE DOCUMENTS FROM PINECONE
# ============================================================

def retrieve_documents(
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.0
):
    """
    Retrieve relevant documents from Pinecone.

    Parameters
    ----------
    question : str
        User's question.

    top_k : int
        Number of documents to retrieve.

    score_threshold : float
        Minimum similarity score.
        Default is 0.0 so that the graph can decide
        whether the retrieved context is useful.

    Returns
    -------
    Pinecone QueryResponse
        Pinecone response containing matching documents.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if embedding_model is None or index is None:
        raise RuntimeError(
            "The vector database is not configured. Add valid PINECONE_API_KEY and "
            "PINECONE_INDEX_NAME values in your .env file before running the app."
        )

    # --------------------------------------------------------
    # 1. Create embedding for the user's question
    # --------------------------------------------------------

    query_vector = create_embedding(question)

    # --------------------------------------------------------
    # 2. Query Pinecone
    # --------------------------------------------------------

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True
    )

    # --------------------------------------------------------
    # 3. Optional score filtering
    # --------------------------------------------------------

    if score_threshold > 0:

        filtered_matches = [
            match
            for match in results.matches
            if match.score >= score_threshold
        ]

        # We don't modify the Pinecone response.
        # Instead, return a simple structure if filtering
        # is requested.
        return filtered_matches

    return results


# ============================================================
# TEST RETRIEVER DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PINECONE RETRIEVER TEST")
    print("=" * 70)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        raise SystemExit

    print("\nSearching Pinecone...")

    results = retrieve_documents(
        question=question,
        top_k=5
    )

    print("\nRetrieved Documents:")
    print("=" * 70)

    # Pinecone QueryResponse contains matches
    matches = results.matches

    if not matches:
        print("No documents found.")
        raise SystemExit

    for i, match in enumerate(matches, start=1):

        metadata = match.metadata or {}

        text = metadata.get(
            "text",
            "No text available"
        )

        page = metadata.get(
            "page",
            "N/A"
        )

        print(f"\nResult: {i}")
        print(f"Score: {match.score:.4f}")
        print(f"ID: {match.id}")
        print(f"Page: {page}")

        print("\nText:")
        print(text)

        print("-" * 70)