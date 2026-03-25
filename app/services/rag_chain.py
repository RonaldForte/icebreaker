from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# the prompt needs some changes. The LLM is referring to the context we are providing it
# it should only respond to the user's query without any "fourth wall breaks"

def get_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template("""
    You are an expert software engineer assistant.

    Use the provided context to answer the question.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Be precise
    - If possible, mention file names from metadata
    - If the answer is not in the context, say that clearly

    Answer:
    """)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain