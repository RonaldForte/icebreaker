from fastapi import FastAPI
from app.routers import langchain_router, vectordb_router, crud_router

# The FastAPI app stays intentionally small and delegates behavior to routers.
app = FastAPI(title="Cold-Start AI Onboarding Assistant")

# Routers are split by responsibility: basic CRUD, LangChain chat, and repo/vector workflows.
app.include_router(langchain_router.router)
app.include_router(vectordb_router.router)
app.include_router(crud_router.router)
