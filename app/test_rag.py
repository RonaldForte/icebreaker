import os
from services.github_loader import load_github_repo
from services.vectordb_service import (
    create_or_get_collection,
    ingest_documents,
    search_collection,
)

# Manual smoke test for local repo ingestion and similarity search.
repo_url = "https://github.com/260223-ML-GenAI/GenAI_APIs.git"
print(f"Loading repo: {repo_url}")
documents = load_github_repo(repo_url)
print(f"Documents loaded: {len(documents)}")

# Create or reuse the collection that will hold embedded repo chunks.
repo_name = os.path.basename(repo_url).replace(".git", "")
collection = create_or_get_collection(repo_name)

# Ingest the loaded files into Chroma before searching.
ingest_count = ingest_documents(repo_name, documents)
print(f"Ingested {ingest_count} documents into collection '{repo_name}'")

# Run a simple semantic search to validate that embeddings and storage work end to end.
query = "authentication"
results = search_collection(repo_name, query, k=3)
print(f"\nTop results for query: '{query}'")
for r in results:
    print(f"- Score: {r['score']}, Content snippet: {r['content'][:100]}...")
