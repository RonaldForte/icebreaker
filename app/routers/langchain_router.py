from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.langchain_service import (
    get_basic_chain,
    get_sequential_chain,
    run_user_memory_turn,
    reset_user_memory_chain,
)
from app.services.vectordb_service import get_grounded_context
from app.services.vectordb_service import search_collection

router = APIRouter(prefix="/langchain", tags=["LangChain"])


def _is_high_level_repo_question(question: str) -> bool:
    # Broad prompts need broader retrieval than implementation-specific questions.
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


def _get_memory_grounded_context(collection_name: str, question: str) -> dict:
    # Blend prompt-specific retrieval with repo-overview queries so memory chat can answer
    # both personal follow-ups and repository-level questions.
    if _is_high_level_repo_question(question):
        queries = [
            question,
            "README project overview purpose architecture",
            "entrypoint main app startup configuration",
            "api routes services repositories workflow",
        ]
        per_query_k = 4
    else:
        queries = [
            question,
            f"{question} implementation details",
            f"{question} related functions classes files",
        ]
        per_query_k = 5

    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for query in queries:
        rows = search_collection(
            collection_name=collection_name,
            query=query,
            k=per_query_k,
            preview_chars=None,
            strategy="mmr",
            fetch_k=max(24, per_query_k * 6),
        )
        for row in rows:
            key = (row.get("source") or "", row.get("content") or "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)

    merged = merged[:18]
    if not merged:
        return {"results": [], "context": "", "sources": []}

    context_blocks = []
    sources: list[str] = []

    for idx, row in enumerate(merged, start=1):
        source = row.get("source") or "unknown"
        sources.append(source)
        context_blocks.append(f"[{idx}] source={source}\n{row.get('content', '')}")

    return {
        "results": merged,
        "context": "\n\n".join(context_blocks),
        "sources": list(dict.fromkeys(sources)),
    }


class ChatRequest(BaseModel):
    input: str


class MemoryChatRequest(BaseModel):
    user_id: int
    input: str
    collection_name: Optional[str] = None


# Initialize reusable stateless chains once at import time.
basic_chain = get_basic_chain()
sequential_chain = get_sequential_chain()


@router.post("/chat")
def general_chat(chat: ChatRequest):
    return basic_chain.invoke(input=chat.input)


@router.post("/support-chat")
def support_chat(chat: ChatRequest):
    return sequential_chain.invoke(input=chat.input)


@router.post("/memory-chat")
def memory_chat(chat: MemoryChatRequest):
    # Memory chat stores the raw user message in memory and injects repo context separately,
    # which keeps facts like names/preferences from being buried by retrieved code snippets.
    repo_context = ""
    sources: list[str] = []
    if chat.collection_name:
        grounded = _get_memory_grounded_context(
            collection_name=chat.collection_name,
            question=chat.input,
        )

        # Fallback for edge-cases where blended retrieval comes up empty.
        if not grounded["results"]:
            grounded = get_grounded_context(
                collection_name=chat.collection_name,
                query=chat.input,
                k=10,
            )

        if grounded["results"]:
            sources = grounded["sources"]
            repo_context = (
                "Use ONLY this retrieved repository context for repository-specific claims. "
                "If the answer is not supported by this context, explicitly say so.\n\n"
                f"Retrieved Context:\n{grounded['context']}"
            )

    result = run_user_memory_turn(
        user_id=str(chat.user_id),
        user_input=chat.input,
        repo_context=repo_context,
    )

    if isinstance(result, dict):
        output = result.get("response") or result.get("output")
        return {
            "output": output,
            "history": result.get("history", ""),
            "sources": sources,
        }

    return {"output": str(result), "sources": sources}


@router.post("/memory-chat/reset/{user_id}")
def reset_memory_chat(user_id: int):
    # Reset is exposed so the Streamlit demo can quickly start a clean conversation.
    cleared = reset_user_memory_chain(str(user_id))
    if cleared:
        return {"message": f"Memory reset for user {user_id}."}
    return {"message": f"No active memory found for user {user_id}."}
