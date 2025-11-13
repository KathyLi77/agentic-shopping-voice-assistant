# graph/nodes.py
from graph.state import GraphState
from graph.router.model import load_router_model
from graph.router.prompts import router_prompt
from graph.router.parser import parse_router_output
from langchain_core.runnables import RunnablePassthrough
import logging
logger = logging.getLogger(__name__)

# Not loaded when the file is imported
_llm = None  
_router_chain = None  

def get_router_chain():
    """Get router chain, loading model lazily if needed."""
    global _llm, _router_chain
    if _router_chain is None:   # Only load when first called
        _llm = load_router_model()  # Loads model only when needed
        _router_chain = (
            {"query": RunnablePassthrough()}
            | router_prompt
            | _llm
            | parse_router_output
        )
    return _router_chain

def router_node(state: GraphState) -> GraphState:
    """Extract task, constraints, and safety flags using LangChain + HuggingFace."""
    
    query = state["query"]
    
    try:
        router_chain = get_router_chain()
        result = router_chain.invoke(query)
        
        # Update state
        state["task"] = result.task
        state["constraints"] = result.constraints.model_dump(exclude_none=True)
        state["safety_flags"] = result.safety_flags
        
        # Log
        state["step_log"].append({
            "node": "router",
            "input": query,
            "output": {
                "task": result.task,
                "constraints": state["constraints"],
                "safety_flags": result.safety_flags
            },
            "success": True
        })
        
    except Exception as e:
        # Fallback on error
        logger.error(f"Router error: {e}", exc_info=True)
        
        state["task"] = "product_search"
        state["constraints"] = {}
        state["safety_flags"] = []
        state["step_log"].append({
            "node": "router",
            "error": str(e),
            "success": False
        })
    
    return state