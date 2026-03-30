from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferWindowMemory

# One local Ollama chat model powers all chat-style experiences in the app.
llm = ChatOllama(model="llama3.2:3b", temperature=0.5)

# The basic chain is intentionally simple and acts as the foundation for the others.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
     You are a helpful assistant for software onboarding.
     Answer clearly and concisely.
     """,
        ),
        ("user", "{input}"),
    ]
)


def get_basic_chain():
    return prompt | llm


# The sequential chain demonstrates that the LLM can do more than one-step response generation.
def get_sequential_chain():
    first_chain = get_basic_chain()
    second_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the initial response to be more professional and helpful in a kind way.",
            ),
            ("user", "Initial LLM response: {input}"),
        ]
    )
    second_chain = second_prompt | llm
    return first_chain | second_chain


def get_memory_chain():
    # Window memory preserves recent user facts more reliably than summary memory for this demo.
    memory = ConversationBufferWindowMemory(k=12)
    memory_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful onboarding assistant. Keep track of user-introduced facts "
                "(like names, preferences, and goals) and use them naturally.",
            ),
            ("user", "Conversation History: {history}\nCurrent Input: {input}"),
        ]
    )
    return ConversationChain(llm=llm, prompt=memory_prompt, memory=memory)


user_memory_chains: dict[str, ConversationChain] = {}


def get_user_memory_chain(user_id: str) -> ConversationChain:
    if user_id not in user_memory_chains:
        user_memory_chains[user_id] = get_memory_chain()
    return user_memory_chains[user_id]


def run_user_memory_turn(user_id: str, user_input: str, repo_context: str = "") -> dict:
    # Repo-aware memory uses a separate grounded prompt, then stores only the raw user turn and
    # final answer so personal details do not get polluted by giant retrieved context blocks.
    user_chain = get_user_memory_chain(user_id)

    if not repo_context:
        result = user_chain.invoke({"input": user_input})
        if isinstance(result, dict):
            return {
                "output": result.get("response") or result.get("output") or "",
                "history": result.get("history", ""),
            }
        return {"output": str(result), "history": ""}

    # Generate grounded response while preserving memory with raw user input only.
    history = user_chain.memory.load_memory_variables({}).get("history", "")
    grounded_prompt = f"""
You are a helpful onboarding assistant.

Use the repository context only for repository-specific claims.
If a repository-specific claim is not supported by context, say so clearly.
Also preserve and use personal conversational facts from history (like user name/preferences).

Conversation History:
{history}

Repository Context:
{repo_context}

Current Input:
{user_input}
"""

    output = llm.invoke(grounded_prompt).content
    user_chain.memory.save_context({"input": user_input}, {"output": output})
    updated_history = user_chain.memory.load_memory_variables({}).get("history", "")
    return {"output": output, "history": updated_history}


def reset_user_memory_chain(user_id: str) -> bool:
    return user_memory_chains.pop(user_id, None) is not None
