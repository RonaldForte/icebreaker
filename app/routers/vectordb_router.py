from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.langchain_service import llm
import os

from langchain_core.documents import Document
from app.services.vectordb_service import (
    create_or_get_collection,
    ingest_documents,
    search_collection,
)
from app.services.github_loader import load_github_repo

router = APIRouter()
active_collection: Optional[str] = None

class RepoLoadRequest(BaseModel):
    repo_url: str

class RAGQueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None

class SearchRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None


@router.post("/vectordb/load-repo")
def load_repo(request: RepoLoadRequest):
    global active_collection
    try:
        documents = load_github_repo(request.repo_url)

        if not documents:
            return {"message": "No documents found in repo."}

        repo_name = os.path.basename(request.repo_url).replace(".git", "")
        create_or_get_collection(repo_name)
        active_collection = repo_name

        ingested_count = ingest_documents(repo_name, documents)

        return {
            "message": f"Repository loaded and ingested into collection '{repo_name}'.",
            "documents_ingested": ingested_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectordb/search")
def search(request: SearchRequest):
    try:
        collection_name = request.collection_name
        if not collection_name:
            raise HTTPException(status_code=400, detail="Collection name is required.")
        results = search_collection(collection_name, request.question, k=5)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectordb/rag-query")
def rag_query(request: RAGQueryRequest):
    try:
        collection_name = request.collection_name or active_collection
        if not collection_name:
            raise HTTPException(
                status_code=400,
                detail="Collection name is required. Load a repo first or provide collection_name."
            )
        results = search_collection(collection_name, request.question, k=5)

        if not results:
            return {"answer": "No relevant documents found."}

        context = "\n\n".join([r["content"] for r in results])

        prompt = f"""
        You are a software onboarding assistant.

        Use the context below to answer the question.

        Context:
        {context}

        Question:
        {request.question}
        """

        response = llm.invoke(prompt)

        return {"answer": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))