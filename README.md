# Cold-Start AI Onboarding Assistant

This project is a FastAPI + Streamlit demo that ingests a GitHub repository, stores its contents in ChromaDB, and uses LangChain + local Ollama models to answer grounded questions about that codebase.

It is designed to show three things clearly:

- a small but complete FastAPI backend
- a user-facing Streamlit interface
- repository-aware AI features including semantic search, grounded summaries, repo Q&A, and memory chat

## What The App Does

The app walks a user through four steps:

1. Create or select a demo user.
2. Load a GitHub repository into a local Chroma vector store.
3. Explore the repository with grounded summary, semantic search, and repo Q&A.
4. Chat with a memory-enabled assistant that can retain personal details while also using repository context.

## Tech Stack

- FastAPI for the backend API
- Streamlit for the demo UI
- LangChain for chain/memory orchestration
- Ollama for the local LLM and embedding model
- ChromaDB for vector storage
- GitPython for cloning and refreshing GitHub repositories

## Project Structure

```text
app/
  main.py                  FastAPI app entry point
  streamlit.py             Guided Streamlit UI
  routers/
    crud_router.py         Simple in-memory user CRUD endpoints
    langchain_router.py    Chat, support chat, and memory chat endpoints
    vectordb_router.py     Repo loading, search, repo Q&A, and grounded summary
  services/
    chunker.py             Document chunking for embeddings
    github_loader.py       Git clone/pull and file ingestion
    langchain_service.py   LLM, chains, and user memory helpers
    rag_chain.py           Minimal example RAG helper
    vectordb_service.py    Chroma collection, search, and grounding utilities
  test_rag.py              Manual smoke test script
requirements.txt
start-app.ps1              One-command setup and launch for Windows
```

## Requirements

- Windows PowerShell
- Python 3.11+ recommended
- Ollama installed locally

Required Ollama models:

- `llama3.2:3b`
- `nomic-embed-text`

## Quick Start

From the project root, run:

```powershell
.\start-app.ps1
```

The script will:

1. create the local `venv` if it does not exist
2. install `requirements.txt`
3. pull the required Ollama models when Ollama is available
4. start FastAPI in one PowerShell window
5. start Streamlit in another PowerShell window
6. open the Streamlit app in your browser

## Manual Start

If you prefer to run services yourself:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
uvicorn app.main:app --reload
```

In a second terminal:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app/streamlit.py
```

## API Overview

### CRUD

- `POST /users`
- `GET /users`
- `GET /users/{user_id}`
- `DELETE /users/{user_id}`

### LangChain

- `POST /langchain/chat`
- `POST /langchain/support-chat`
- `POST /langchain/memory-chat`
- `POST /langchain/memory-chat/reset/{user_id}`

### Repo + VectorDB

- `POST /vectordb/load-repo`
- `POST /vectordb/search`
- `POST /vectordb/rag-query`
- `POST /vectordb/code-summary`

## Notes

- Chroma persists to `chroma_db/` at the project root, outside `app/`, to avoid uvicorn reload loops.
- Re-loading a repository resets its vector collection before ingesting again, which keeps demos deterministic.
- Memory chat stores raw user messages in memory and uses repo context separately so personal facts are not lost.

## Troubleshooting

If Streamlit says the backend is outdated or an endpoint is missing:

1. stop old uvicorn terminals
2. start one clean FastAPI process from this workspace
3. refresh Streamlit

If repository answers are weak:

1. reload the repository so the collection is rebuilt
2. try a more specific prompt
3. verify Ollama is running and both models are installed