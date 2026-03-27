from fastapi import APIRouter
from pydantic import BaseModel
from app.services.langchain_service import get_basic_chain, get_sequential_chain, get_transform_chain, get_memory_chain

router = APIRouter(prefix="/langchain", tags=["LangChain"])

class ChatRequest(BaseModel):
    input: str

basic_chain = get_basic_chain()
sequential_chain = get_sequential_chain()
transform_chain = get_transform_chain()
memory_chain = get_memory_chain()

@router.post("/chat")
def general_chat(chat: ChatRequest):
    return basic_chain.invoke(input=chat.input)

@router.post("/support-chat")
def support_chat(chat: ChatRequest):
    return sequential_chain.invoke(input=chat.input)

@router.post("/transform-chat")
def transform_chat(chat: ChatRequest):
    return transform_chain.invoke(input=chat.input)

@router.post("/memory-chat")
def memory_chat(chat: ChatRequest):
    return memory_chain.invoke(input=chat.input)