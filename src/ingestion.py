import os

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is missing")


# -----------------------------------------
# Configuration
# -----------------------------------------

PDF_PATH = "data/Ebook-Agentic-AI.pdf"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------------------
# Load embedding model
# -----------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")

print(
    "Embedding dimension:",
    embedding_model.get_sentence_embedding_dimension()
)


# -----------------------------------------
# Connect to Pinecone
# -----------------------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)

print("Connected to Pinecone!")
print("Index:", PINECONE_INDEX_NAME)


# -----------------------------------------
# Load PDF
# -----------------------------------------

print("\nLoading PDF...")

reader = PdfReader(PDF_PATH)

print("Pages loaded:", len(reader.pages))


# -----------------------------------------
# Create chunks
# -----------------------------------------

chunks = []

chunk_size = 1200
chunk_overlap = 200


for page_number, page in enumerate(reader.pages):

    text = page.extract_text()

    if not text:
        continue

    text = text.strip()

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end].strip()

        if chunk_text:

            chunks.append({
                "text": chunk_text,
                "page": page_number + 1
            })

        start += chunk_size - chunk_overlap


print("Total chunks created:", len(chunks))

if not chunks:
    raise ValueError(f"No extractable text was found in {PDF_PATH}")


# -----------------------------------------
# Create embeddings
# -----------------------------------------

print("\nCreating embeddings...")

texts = [chunk["text"] for chunk in chunks]

vectors = embedding_model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embeddings created!")

print("Number of embeddings:", len(vectors))

print("Embedding dimension:", len(vectors[0]))


# -----------------------------------------
# Prepare Pinecone records
# -----------------------------------------

records = []

for i, (chunk, vector) in enumerate(zip(chunks, vectors)):

    record = {
        "id": f"chunk-{i:04d}",
        "values": vector.tolist(),
        "metadata": {
            "text": chunk["text"],
            "source": PDF_PATH,
            "page": chunk["page"]
        }
    }

    records.append(record)


# -----------------------------------------
# Upload to Pinecone
# -----------------------------------------

print("\nUploading vectors to Pinecone...")

batch_size = 50

for start in range(0, len(records), batch_size):

    batch = records[start:start + batch_size]

    index.upsert(
        vectors=batch,
        namespace=PINECONE_NAMESPACE
    )

    print(
        f"Uploaded {min(start + batch_size, len(records))}"
        f"/{len(records)}"
    )


# -----------------------------------------
# Verify
# -----------------------------------------

stats = index.describe_index_stats()

print("\n--------------------------------")
print("INGESTION COMPLETED")
print("--------------------------------")

print("Index:", PINECONE_INDEX_NAME)

print("Namespace:", PINECONE_NAMESPACE)

print("Total vectors:", stats.total_vector_count)

print("Namespace statistics:")
print(stats.namespaces)