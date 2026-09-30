from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END

from src.retriever import retrieve_documents
from src.llm import llm


# ============================================================
# STATE
# ============================================================

class RAGState(TypedDict, total=False):
    question: str
    documents: List[Dict[str, Any]]
    context: str
    answer: str


# ============================================================
# RETRIEVE NODE
# ============================================================

def retrieve_node(state: RAGState) -> RAGState:
    """
    Retrieve relevant documents from Pinecone
    using the user's question.
    """

    question = state["question"]

    print("\n[RETRIEVE]")
    print(f"Question: {question}")

    # Call retriever
    results = retrieve_documents(question, top_k=5)

    documents = []

    print("\nRetrieved documents:")

    # --------------------------------------------------------
    # Handle Pinecone QueryResponse safely
    # --------------------------------------------------------

    matches = results.matches if hasattr(results, "matches") else []

    for i, match in enumerate(matches, start=1):

        # Pinecone match can be an object
        # or a dictionary depending on SDK version.
        if hasattr(match, "id"):
            doc_id = match.id
            score = match.score
            metadata = match.metadata or {}
        else:
            doc_id = match.get("id", "")
            score = match.get("score", 0.0)
            metadata = match.get("metadata", {})

        text = metadata.get("text", "")
        page = metadata.get("page", "N/A")

        documents.append(
            {
                "id": doc_id,
                "score": score,
                "page": page,
                "text": text,
                "metadata": metadata,
            }
        )

        print(
            f"{i}. "
            f"Score: {score:.4f} | "
            f"Page: {page}"
        )

    print(f"\nRetrieved {len(documents)} documents")

    # --------------------------------------------------------
    # Create context for LLM
    # --------------------------------------------------------

    context_parts = []

    for doc in documents:

        page = doc["page"]
        text = doc["text"]

        context_parts.append(
            f"[Page {page}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    return {
        "documents": documents,
        "context": context,
    }


# ============================================================
# GENERATE NODE
# ============================================================

def generate_node(state: RAGState) -> RAGState:
    """
    Generate an answer using only the retrieved
    reference material.
    """

    question = state["question"]
    context = state.get("context", "")

    print("\n[GENERATE]")

    # --------------------------------------------------------
    # No relevant documents
    # --------------------------------------------------------

    if not context.strip():

        answer = (
            "I couldn't find any indexed reference passages for this question. "
            "Please confirm the PDF has been ingested into the configured "
            "Pinecone index and namespace, then try again."
        )

        print("\nAssistant:")
        print(answer)

        return {
            "answer": answer
        }

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant.

Your task is to answer the user's question using ONLY
the provided reference material.

IMPORTANT RULES:

1. Use only information from the reference material.
2. Do not use outside knowledge.
3. Do not invent or hallucinate information.
4. If the answer is not present in the reference material,
   clearly say:

   "I'm sorry, but the provided reference material
   does not contain information about this."

5. Give a clear and concise answer.
6. Use bullet points when they improve readability.
7. If the reference material contains relevant page numbers,
   mention them naturally when useful.
8. Do not mention internal implementation details such as
   Pinecone, embeddings, LangGraph, or retrieval unless
   the user specifically asks about the system.

REFERENCE MATERIAL:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # --------------------------------------------------------
    # Call LLM
    # --------------------------------------------------------

    response = llm.invoke(prompt)

    # LangChain chat models normally return AIMessage
    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    print("\nAssistant:")
    print(answer)

    return {
        "answer": answer
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(RAGState)

# Add nodes
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

# Workflow
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

# Compile
rag_graph = builder.compile()


# ============================================================
# OPTIONAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AGENTIC AI RAG CHATBOT")
    print("=" * 70)
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question:
            continue

        result = rag_graph.invoke(
            {
                "question": question
            }
        )

        print("\n" + "-" * 70)
        print()