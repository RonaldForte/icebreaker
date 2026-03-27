from fastapi import APIRouter
from pydantic import BaseModel
from app.services.github_loader import load_github_repo
from app.services.chunker import chunk_documents
from app.services.vectordb_service import (
    ingest_documents,
    get_retriever,
    search_collection
)
from app.services.rag_chain import get_rag_chain
from app.services.langchain_service import llm

router = APIRouter(prefix="/vectordb", tags=["VectorDB"])

retriever = None
rag_chain = None
current_collection = None


class QueryRequest(BaseModel):
    question: str


class RepoRequest(BaseModel):
    repo_url: str


def check_repo_loaded():
    return retriever is not None and rag_chain is not None


@router.post("/load-repo")
def load_repo(request: RepoRequest):
    global retriever, rag_chain, current_collection

    repo_name = request.repo_url.rstrip("/").split("/")[-1]
    current_collection = repo_name

    docs = load_github_repo(request.repo_url)
    docs = chunk_documents(docs)

    ingest_documents(current_collection, docs)

    retriever = get_retriever(current_collection)
    rag_chain = get_rag_chain(llm, retriever)

    return {
        "message": f"Repository '{repo_name}' loaded successfully!",
        "documents_ingested": len(docs)
    }


@router.post("/rag-query")
def rag_query(request: QueryRequest):
    if not check_repo_loaded():
        return {"error": "No repo loaded yet."}

    response = rag_chain.invoke(request.question)
    return {"answer": response.content}


@router.post("/search")
def search(request: QueryRequest):
    if current_collection is None:
        return {"error": "No repo loaded yet."}

    return search_collection(current_collection, request.question)