from langchain_ollama import ChatOllama
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.services.vectordb_service import create_or_get_vectorstore
from app.graph.state import AgentState

# Small schema to keep LLM focused
class ExtractionSchema(BaseModel):
    filename: Optional[str] = Field(None)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.5
)

def extractor_node(state: AgentState):
    """Initial pass to find branch/file."""
    prompt = ChatPromptTemplate.from_template(
        "Extract 'filename' from: {query}. Return null if not found."
    )
    chain = prompt | llm.with_structured_output(ExtractionSchema)
    result = chain.invoke({"query": state["query"]})
    return {"filename": result.filename, "retry_count": 0}

def re_extractor_node(state: AgentState):
    """The 'Deep Scan' if the first one missed a filename."""
    print("--- SELF-CORRECTION: Re-scanning for filename ---")
    prompt = ChatPromptTemplate.from_template(
        "You missed the filename in the first pass. Look again at: {query}. "
        "Search for strings like 'main.py', 'app', etc. Return null if truly missing."
    )
    chain = prompt | llm.with_structured_output(ExtractionSchema)
    result = chain.invoke({"query": state["query"]})
    # Increment retry_count to prevent infinite loops
    return {"filename": result.filename, "retry_count": state.get("retry_count", 0) + 1}

def retriever_node(state: AgentState):
    """Handles the actual ChromaDB search."""
    vectorstore = create_or_get_vectorstore(state["repo_url"], state["branch"])
    
    search_filter = None
    if state.get("filename"):
        search_filter = {"source": {"$contains": state["filename"]}}

    docs = vectorstore.similarity_search(state["query"], k=5, filter=search_filter)
    return {"context": [d.page_content for d in docs]}

def generation_node(state: AgentState):
    """Final answer based on the retrieved code context."""
    print("--- GENERATING ANSWER ---")
    
    # 1. Join all retrieved chunks into one big string
    context_text = "\n\n".join(state["context"])
    
    # 2. Build a prompt that forces the LLM to use the context
    prompt = ChatPromptTemplate.from_template(
        """You are a senior developer. Use the provided code snippets to answer the user's question. 
        If you don't know the answer based on the code, just say you don't know.
        
        Question: {query}
        
        Code Context:
        {context}
        
        Answer:"""
    )
    
    # 3. Run the chain
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "context": context_text})
    
    # 4. Update the state with the REAL answer
    return {"answer": response}