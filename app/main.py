from fastapi import FastAPI
from app.routers import langchain_router, vectordb_router, crud_router

app = FastAPI(title="Cold-Start AI Onboarding Assistant")

app.include_router(langchain_router.router)
app.include_router(vectordb_router.router)
app.include_router(crud_router.router)