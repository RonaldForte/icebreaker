from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.chains.transform import TransformChain
from langchain_classic.memory import ConversationSummaryMemory

#LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0.5)

#Basic Chain
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a helpful assistant for software onboarding.
     Answer clearly and concisely.
     """),
    ("user", "{input}")
])
def get_basic_chain():
    return prompt | llm

# Sequential chain example that rewrites tone
def get_sequential_chain():
    first_chain = get_basic_chain()
    second_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the initial response to be more professional and helpful in a kind way."),
        ("user", "Initial LLM response: {input}")
    ])
    second_chain = second_prompt | llm
    return first_chain | second_chain

def get_transform_chain():
    vague_phrases = [
        "what does this do",
        "explain this",
        "where is logic",
        "help",
        "what is happening"
    ]

    def transform_fn(inputs):
        user_input = inputs["input"].lower()

        if any(p in user_input for p in vague_phrases):
            return {
                "output": (
                    "Your question is too vague. Please specify:\n"
                    "- File name\n"
                    "- Function/class name\n"
                    "- Or describe what you're trying to understand\n\n"
                    "Example: 'Where is the authentication logic in this repo?'"
                )
            }

        return {"output": inputs["input"]}

    return TransformChain(
        input_variables=["input"],
        output_variables=["output"],
        transform=transform_fn
    )
    
# Memory chain
def get_memory_chain():
    memory = ConversationSummaryMemory(llm=llm)
    memory_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful onboarding assistant."),
        ("user", "Conversation History: {history}\nCurrent Input: {input}")
    ])
    return ConversationChain(llm=llm, prompt=memory_prompt, memory=memory)