from src.embedding import index

index.delete(
    ids=["test-001"],
    namespace="agentic-ai"
)

print("Deleted test-001")