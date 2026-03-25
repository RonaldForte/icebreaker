import hashlib
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This Service is full of functions that let our VectorDB work
# Creating/Getting collections, ingesting data as vector embeddings, and searching the data

PERSIST_DIRECTORY = "app/chroma_db" # Where the DB lives in our application

# The model we pulled - this is the AI model that turns our data into vector embeddings
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

# The Chroma store itself (which contains the collections)
vector_store: dict[str, Chroma] = {}

# Create a new collection in VectorDB - a collection in VDB is like a table in SQL.
# If the collection already exists, just return it.
def create_or_get_collection(collection_name:str):

    # If the Collection doesn't already exist, make a new one
    if collection_name not in vector_store:
        vector_store[collection_name] = Chroma(
            collection_name = collection_name,
            persist_directory = PERSIST_DIRECTORY,
            embedding_function = EMBEDDING_MODEL
        )

    return vector_store[collection_name]

# Ingest JSON items into the DB
def ingest_json(collection_name:str, items:list[dict[str, Any]]):

    # Get an instance of the collection in the vector store
    collection = create_or_get_collection(collection_name)

    # Turn the list of items into a list of document to get ingested
    docs = [
        Document(page_content=item["text"], metadata=item.get("metadata", {}))
        for item in items
    ]

    # Attach IDs to each document (easy cuz it's the sample data)
    ids = [item["id"] for item in items]

    # Add the documents, perform the ingestion, and we have embedded vectors!
    collection.add_documents(docs, ids=ids)

    # Return the length of the ingested items just for visibility
    return len(items)



# We will need to repurpose this, depending on what the collections are
# Just pasted this here for now
def search_collection(collection_name:str, query:str, k:int=5, game_title:str=None):

    # Get the collection from the vector store
    collection = create_or_get_collection(collection_name)

    # Perform the similarity search
    # (This is a Retriever, we're using it to retrieve data for the LLM's response)
    results = (collection
               .similarity_search_with_score(query,
                k=k,
                filter={"game_title": game_title} if game_title else None))

    # NOTE: the filter is optional, and only comes into play if the user passes in a game title

    return results