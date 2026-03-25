import hashlib
import shutil
import os

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_ollama import OllamaEmbeddings
from app.services.chunker import chunk_documents

# This Service is full of functions that let our VectorDB work
# Creating/Getting collections, ingesting data as vector embeddings, and searching the data

PERSIST_DIRECTORY = "app/chroma_db" # Where the DB lives in our application

# The model we pulled - this is the AI model that turns our data into vector embeddings
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

# The Chroma store itself (which contains the collections)
vector_store: dict[str, Chroma] = {}

# Create a unique collection name based on the repo url and branch name (to avoid duplicates and mixing)
def get_hashed_collection_name(repo_url: str, branch: str):
    identifier = f"{repo_url}-{branch}"
    return "repo_" + hashlib.md5(identifier.encode()).hexdigest()[:16]

def create_or_get_vectorstore (repo_url, branch):
    # 1. Generate the unique name
    collection_name = get_hashed_collection_name(repo_url, branch)
    temp_path = f"./temp/{collection_name}"
    
    # 2. Connect to Chroma (This points us to the right "folder")
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory="./chroma_db",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )

    # 3. If it's empty, we need to load and add the data
    if vectorstore._collection.count() == 0:
        try:
            print(f"Ingesting {repo_url}...")
            loader = GitLoader(clone_url=repo_url, repo_path=temp_path, branch=branch)
            docs = loader.load()
            chunks = chunk_documents(docs)
            vectorstore.add_documents(chunks)
            print("Ingestion complete.")
        finally:
            # DELETE the temp folder now that Chroma has the data
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
                print(f"Cleaned up temporary files at {temp_path}")
    else:
        print(f"Collection {collection_name} already exists with data. Skipping loader.")

    return vectorstore