from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.langchain_service import llm
import os
import re

from app.services.vectordb_service import (
    create_or_get_collection,
    get_grounded_context,
    ingest_documents,
    reset_collection,
    search_collection,
)
from app.services.github_loader import load_github_repo

router = APIRouter()
# The last loaded repo becomes the default active collection for repo-aware endpoints.
active_collection: Optional[str] = None


def _format_context_from_results(results: list[dict]) -> tuple[str, list[str]]:
    # This converts vector hits into the numbered context format used by prompts and source citations.
    context_blocks = []
    sources: list[str] = []

    for idx, row in enumerate(results, start=1):
        source = row.get("source") or "unknown"
        content = row.get("content") or ""
        sources.append(source)
        context_blocks.append(f"[{idx}] source={source}\n{content}")

    return "\n\n".join(context_blocks), list(dict.fromkeys(sources))


def _collect_summary_results(collection_name: str, prompt: str, k: int) -> list[dict]:
    # Summary prompts work best when they mix the user request with a small amount of
    # repo-overview retrieval instead of relying on a single embedding query.
    prompt_only_queries = [prompt]
    overview_queries = [
        "README project overview purpose architecture",
        "entrypoint main app startup configuration",
        "api routes services repositories workflow",
    ]

    is_high_level = _is_high_level_repo_question(prompt)
    queries = prompt_only_queries + (
        overview_queries if is_high_level else overview_queries[:1]
    )

    primary_k = max(6, min(12, k + 2))
    secondary_k = max(3, min(6, k // 2 + 2))
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for idx, query in enumerate(queries):
        current_k = primary_k if idx == 0 else secondary_k
        rows = search_collection(
            collection_name=collection_name,
            query=query,
            k=current_k,
            preview_chars=None,
            strategy="mmr",
            fetch_k=max(24, current_k * 6),
        )
        for row in rows:
            key = (row.get("source") or "", row.get("content") or "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)

    return merged[: max(12, min(28, k * 3))]


def _extract_repo_signals(results: list[dict]) -> dict:
    # The LLM writes the narrative summary, while this deterministic pass exposes concrete
    # technical signals that help users verify what was actually retrieved.
    detected_functions: set[str] = set()
    detected_classes: set[str] = set()
    detected_routes: set[str] = set()
    detected_modules: set[str] = set()

    route_pattern = re.compile(r"@router\.(get|post|put|delete|patch)\(([^)]*)\)")
    fn_pattern = re.compile(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
    class_pattern = re.compile(r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:(]")

    for row in results:
        source = (row.get("source") or "").replace("\\", "/")
        if source:
            module_name = source.split("/")[-1]
            detected_modules.add(module_name)

        content = row.get("content", "")
        if not content:
            continue

        for line in content.splitlines():
            fn_match = fn_pattern.search(line)
            if fn_match:
                detected_functions.add(fn_match.group(1))

            class_match = class_pattern.search(line)
            if class_match:
                detected_classes.add(class_match.group(1))

            route_match = route_pattern.search(line)
            if route_match:
                method = route_match.group(1).upper()
                path = route_match.group(2).strip().strip("\"'")
                detected_routes.add(f"{method} {path}")

    return {
        "functions": sorted(detected_functions),
        "classes": sorted(detected_classes),
        "routes": sorted(detected_routes),
        "modules": sorted(detected_modules),
    }


def _build_grounded_summary(prompt: str, results: list[dict]) -> dict:
    if not results:
        return {
            "summary": "No relevant code context found to summarize.",
            "sources": [],
        }

    context, sources = _format_context_from_results(results)
    signals = _extract_repo_signals(results)

    if _is_high_level_repo_question(prompt):
        # Broad prompts should yield a repo overview instead of a narrow implementation answer.
        instruction_block = """
Write a concise but useful repository overview with these sections:
1) What this repository appears to do
2) Main components and responsibilities
3) Likely runtime flow (if inferable)
4) Unknowns / missing context
"""
    else:
        # Specific prompts should answer the request directly instead of reusing the overview template.
        instruction_block = """
Answer the user request directly and specifically.
Do not use a generic repository-overview template.
Include a short "Unknowns / missing context" section only if needed.
"""

    llm_prompt = f"""
You are a software onboarding assistant.

Use ONLY the retrieved repository context below.
Do not invent product details, company/team information, or behavior not explicitly supported.
When evidence is missing, explicitly say "Unknown from retrieved context".
Cite evidence with [number] notation.

User request:
{prompt}

{instruction_block}

Retrieved context:
{context}
"""

    narrative = llm.invoke(llm_prompt).content

    signal_lines = [
        "Confirmed Technical Signals:",
        f"- Retrieved {len(results)} snippets from {len(sources)} files.",
        "- Sources: "
        + (", ".join(sources[:10]) + (" ..." if len(sources) > 10 else "")),
    ]

    if signals["routes"]:
        signal_lines.append(
            "- API routes seen: "
            + ", ".join(signals["routes"][:12])
            + (" ..." if len(signals["routes"]) > 12 else "")
        )
    if signals["classes"]:
        signal_lines.append(
            "- Classes seen: "
            + ", ".join(signals["classes"][:12])
            + (" ..." if len(signals["classes"]) > 12 else "")
        )
    if signals["functions"]:
        signal_lines.append(
            "- Functions seen: "
            + ", ".join(signals["functions"][:20])
            + (" ..." if len(signals["functions"]) > 20 else "")
        )

    summary = f"{narrative}\n\n" + "\n".join(signal_lines)
    return {"summary": summary, "sources": sources}


def _is_high_level_repo_question(question: str) -> bool:
    q = question.strip().lower()
    broad_markers = [
        "what does this repo do",
        "what does this repository do",
        "summarize",
        "overview",
        "high level",
        "architecture",
        "purpose",
        "what is this project",
    ]
    return any(marker in q for marker in broad_markers)


class RepoLoadRequest(BaseModel):
    repo_url: str


class RAGQueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None


class SearchRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None


class CodeSummaryRequest(BaseModel):
    prompt: str = "Summarize what this codebase does."
    collection_name: Optional[str] = None
    k: int = 8


@router.post("/vectordb/load-repo")
def load_repo(request: RepoLoadRequest):
    global active_collection
    try:
        # Each load pulls the repo locally and rebuilds its collection from scratch so stale
        # vectors do not accumulate between repeated demos.
        documents = load_github_repo(request.repo_url)

        if not documents:
            return {"message": "No documents found in repo."}

        repo_name = os.path.basename(request.repo_url).replace(".git", "")
        reset_collection(repo_name)
        create_or_get_collection(repo_name)
        active_collection = repo_name

        ingested_count = ingest_documents(repo_name, documents)

        return {
            "message": f"Repository loaded and ingested into collection '{repo_name}'.",
            "documents_ingested": ingested_count,
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
                detail="Collection name is required. Load a repo first or provide collection_name.",
            )
        if _is_high_level_repo_question(request.question):
            # High-level questions route through the richer summary pipeline because plain
            # similarity search tends to over-focus on one service/module.
            summary_results = _collect_summary_results(
                collection_name=collection_name,
                prompt=request.question,
                k=10,
            )
            summary_payload = _build_grounded_summary(
                prompt=request.question,
                results=summary_results,
            )
            return {
                "answer": summary_payload["summary"],
                "sources": summary_payload["sources"],
            }

        grounded = get_grounded_context(
            collection_name=collection_name,
            query=request.question,
            k=10,
        )
        results = grounded["results"]

        if not results:
            return {"answer": "No relevant documents found."}

        prompt = f"""
        You are a software onboarding assistant.

        Use ONLY the context below to answer the question.
        If the answer is not clearly supported by the context, say:
        "I don't have enough information in the retrieved repository context to answer that."
        Do not invent company names, products, teams, or project details.
        Cite supporting sources using [number] notation.

        Context:
        {grounded['context']}

        Question:
        {request.question}
        """

        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": grounded["sources"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectordb/code-summary")
def code_summary(request: CodeSummaryRequest):
    try:
        collection_name = request.collection_name or active_collection
        if not collection_name:
            raise HTTPException(
                status_code=400,
                detail="Collection name is required. Load a repo first or provide collection_name.",
            )

        k = max(4, min(request.k, 12))
        results = _collect_summary_results(
            collection_name=collection_name,
            prompt=request.prompt,
            k=k,
        )
        return _build_grounded_summary(prompt=request.prompt, results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
