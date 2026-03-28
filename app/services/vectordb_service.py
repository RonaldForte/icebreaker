import hashlib
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

PERSIST_DIRECTORY = "app/chroma_db"

EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

vector_store: dict[str, Chroma] = {}


def create_or_get_collection(collection_name: str) -> Chroma:
    if not collection_name:
        raise ValueError("Collection name cannot be None or empty")

    if collection_name not in vector_store:
        vector_store[collection_name] = Chroma(
            collection_name=collection_name,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDING_MODEL
        )
    return vector_store[collection_name]


def ingest_documents(collection_name: str, documents: list[Document]) -> int:
    collection = create_or_get_collection(collection_name)

    texts = []
    metadatas = []

    for doc in documents:
        if not doc.page_content or not doc.page_content.strip():
            continue

        texts.append(doc.page_content)
        metadatas.append(doc.metadata or {})

    if not texts:
        raise ValueError("No valid documents to ingest")

    collection.add_texts(
        texts=texts,
        metadatas=metadatas
    )

    return len(texts)


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