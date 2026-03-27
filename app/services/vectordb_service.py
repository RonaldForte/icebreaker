import hashlib
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

PERSIST_DIRECTORY = "app/chroma_db"

EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

vector_store: dict[str, Chroma] = {}


def create_or_get_collection(collection_name: str) -> Chroma:
    if collection_name not in vector_store:
        vector_store[collection_name] = Chroma(
            collection_name=collection_name,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDING_MODEL
        )
    return vector_store[collection_name]


def ingest_documents(collection_name: str, documents: list[Document]) -> int:
    collection = create_or_get_collection(collection_name)

    ids = [
        hashlib.md5(doc.page_content.encode()).hexdigest() #take text -> convert to bytes(.encode) -> hashing algorithm (md5) -> readable string (hexdigest) -> store as ID
        for doc in documents
    ]

    collection.add_documents(documents, ids=ids)
    return len(documents)


def get_retriever(collection_name: str, k: int = 5):
    collection = create_or_get_collection(collection_name)
    return collection.as_retriever(search_kwargs={"k": k})


def search_collection(collection_name: str, query: str, k: int = 5):
    collection = create_or_get_collection(collection_name)

    results = collection.similarity_search_with_score(query, k=k)

    return [
        {
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source"),
            "score": score
        }
        for doc, score in results
    ]