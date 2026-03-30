from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def get_rag_chain(llm, retriever):
    # This helper is kept as a minimal example of a classic LangChain RAG pipeline.
    prompt = ChatPromptTemplate.from_template(
        """
    You are an expert software engineer assistant.

    Use the provided context to answer the question.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Be precise
    - Mention file names from metadata if possible
    - If answer is not in context, say so

    Answer:
    """
    )
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
