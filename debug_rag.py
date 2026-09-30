import os
from dotenv import load_dotenv

load_dotenv()
print('START')

from src.embedding import embedding_model, index
print('MODEL_LOADED', embedding_model is not None)
print('INDEX_LOADED', index is not None)

from src.retriever import retrieve_documents
result = retrieve_documents('What is Agentic AI?', top_k=3)
print('MATCHES', len(result.matches))
if result.matches:
    match = result.matches[0]
    print('SCORE', match.score)
    metadata = match.metadata or {}
    print('PAGE', metadata.get('page'))
    text = metadata.get('text') or ''
    print(text[:500])
