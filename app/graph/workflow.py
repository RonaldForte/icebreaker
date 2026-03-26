from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.graph.nodes import extractor_node, re_extractor_node, retriever_node, generation_node

def should_continue(state: AgentState):
    """The Logic Gate: Decide whether to loop or finish."""
    # 1. If we have context, we are successful. Move to Answer.
    if state.get("context"):
        return "generate"
    
    # 2. If no context, and we haven't retried yet, try to RE-EXTRACT.
    if state.get("retry_count", 0) < 1:
        return "re_extract"
    
    # 3. If we already retried and still have nothing, just try to answer.
    return "generate"

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("extract", extractor_node)
workflow.add_node("re_extract", re_extractor_node)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("generate", generation_node)

# Connect them
workflow.set_entry_point("extract")
workflow.add_edge("extract", "retrieve")

# The Router: Checks the retrieval results
workflow.add_conditional_edges(
    "retrieve", 
    should_continue,
    {
        "re_extract": "re_extract",
        "generate": "generate"
    }
)

# After re-extracting, we try retrieving one more time
workflow.add_edge("re_extract", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()