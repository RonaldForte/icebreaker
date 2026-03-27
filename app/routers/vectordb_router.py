from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

from app.services.vectordb_service import (
    create_or_get_collection,
    ingest_documents,
    search_collection,
)
from app.services.github_loader import load_github_repo

router = APIRouter()

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
    try:
        documents = load_github_repo(request.repo_url)

        if not documents:
            return {"message": "No documents found in repo."}

        repo_name = os.path.basename(request.repo_url).replace(".git", "")
        collection = create_or_get_collection(repo_name)

        items = [
            {"id": str(i), "text": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(documents)
        ]
        ingest_documents(repo_name, items)

        return {"message": f"Repository loaded and ingested into collection '{repo_name}'."}

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
        collection_name = request.collection_name or "main_repo"
        results = search_collection(collection_name, request.question, k=5)
        answer = results[0]["content"] if results else "No relevant documents found."
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))