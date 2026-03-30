from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import os
from app.services.chunker import chunk_documents

# The persist path lives outside app/ so uvicorn reload does not loop on Chroma writes.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PERSIST_DIRECTORY = os.path.join(_PROJECT_ROOT, "chroma_db")

EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

vector_store: dict[str, Chroma] = {}


def create_or_get_collection(collection_name: str) -> Chroma:
    # Collections are cached per process to avoid reopening Chroma clients on every request.
    if not collection_name:
        raise ValueError("Collection name cannot be None or empty")

    if collection_name not in vector_store:
        vector_store[collection_name] = Chroma(
            collection_name=collection_name,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDING_MODEL,
        )
    return vector_store[collection_name]


def reset_collection(collection_name: str) -> None:
    # Rebuilding from scratch keeps demos deterministic after repeated repo loads.
    if not collection_name:
        raise ValueError("Collection name cannot be None or empty")

    try:
        collection = create_or_get_collection(collection_name)
        collection.delete_collection()
    except Exception:
        # If the collection does not exist yet, there is nothing to delete.
        pass
    finally:
        vector_store.pop(collection_name, None)


def ingest_documents(collection_name: str, documents: list[Document]) -> int:
    collection = create_or_get_collection(collection_name)

    # Chunking happens once on ingest so retrieval can work at code-block granularity.
    chunked_documents = chunk_documents(documents)

    texts = []
    metadatas = []

    for doc in chunked_documents:
        if not doc.page_content or not doc.page_content.strip():
            continue

        texts.append(doc.page_content)
        metadatas.append(doc.metadata or {})

    if not texts:
        raise ValueError("No valid documents to ingest")

    collection.add_texts(texts=texts, metadatas=metadatas)

    return len(texts)


def get_retriever(collection_name: str, k: int = 5):
    collection = create_or_get_collection(collection_name)
    return collection.as_retriever(search_kwargs={"k": k})


def search_collection(
    collection_name: str,
    query: str,
    k: int = 5,
    preview_chars: int | None = 200,
    strategy: str = "mmr",
    fetch_k: int | None = None,
):
    collection = create_or_get_collection(collection_name)

    # MMR is the default because it gives more varied context than plain similarity search.
    if strategy == "similarity":
        scored_results = collection.similarity_search_with_score(query, k=k)
        raw_results = [(doc, score) for doc, score in scored_results]
    else:
        mmr_docs = collection.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k or max(20, k * 4),
        )
        raw_results = [(doc, None) for doc in mmr_docs]

    formatted_results = []
    for doc, score in raw_results:
        content = doc.page_content
        if preview_chars is not None:
            content = content[:preview_chars]

        formatted_results.append(
            {
                "content": content,
                "source": doc.metadata.get("source"),
                "score": score,
            }
        )

    return formatted_results


def get_grounded_context(
    collection_name: str,
    query: str,
    k: int = 5,
):
    # This is the shared formatter used by repo Q&A and fallback memory-grounding paths.
    results = search_collection(
        collection_name=collection_name,
        query=query,
        k=k,
        preview_chars=None,
    )

    if not results:
        return {"results": [], "context": "", "sources": []}

    context_blocks = []
    sources: list[str] = []

    for idx, row in enumerate(results, start=1):
        source = row.get("source") or "unknown"
        sources.append(source)
        context_blocks.append(f"[{idx}] source={source}\n{row['content']}")

    return {
        "results": results,
        "context": "\n\n".join(context_blocks),
        "sources": list(dict.fromkeys(sources)),
    }
