from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from app.services.github_loader import load_github_repo
from app.services.vectordb_service import create_or_get_vectorstore
from app.services.chunker import chunk_documents
from app.services.rag_chain import get_rag_chain

# 1. Load repo
repo_url = "https://github.com/RonaldForte/icebreaker"
branch = "main"
#docs = load_github_repo(repo_url)

# 2. Chunk docs
#docs = chunk_documents(docs)

# 3. Vector store
"""
"""
vectorstore = create_or_get_vectorstore(repo_url, branch)

retriever = vectorstore.as_retriever()

# 4. LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.5
)

# 5. Chain
chain = get_rag_chain(llm, retriever)


while True:
    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        print("Goodbye 👋")
        break

    response = chain.invoke(question)

    print("\nAnswer:\n", response.content)