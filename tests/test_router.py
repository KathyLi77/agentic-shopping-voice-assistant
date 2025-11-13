# tests/test_router.py
import pytest
import sys
import logging
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging (only show warnings and errors)
logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

from graph.graph import create_graph
from graph.state import GraphState

def test_router_basic():
    """Test basic router functionality."""
    
    graph = create_graph()
    
    initial_state = GraphState(
        query="organic shampoo under $20",
        step_log=[]
    )
    
    # Run only the router node for testing
    result = graph.invoke(initial_state)
    
    # Assertions
    assert result["task"] == "product_search", f"Expected 'product_search', got '{result.get('task')}'"
    assert result["constraints"].get("max_budget") in [20, 20.0], f"Expected max_budget=20, got {result['constraints'].get('max_budget')}"
    assert result["constraints"].get("material") == "organic", f"Expected material='organic', got '{result['constraints'].get('material')}'"
    assert len(result["safety_flags"]) == 0, f"Expected no safety flags, got {result['safety_flags']}"
    
def test_router_comparison():
    """Test comparison query."""
    
    graph = create_graph()
    
    result = graph.invoke({
        "query": "compare Nike vs Adidas running shoes",
        "step_log": []
    })
    
    assert result["task"] == "comparison", f"Expected 'comparison', got '{result.get('task')}'"
    assert "Nike" in result["constraints"].get("brand", []) or "Adidas" in result["constraints"].get("brand", []), f"Expected brand with Nike/Adidas, got {result['constraints'].get('brand')}"

def test_router_safety():
    """Test safety flag detection."""
    
    graph = create_graph()
    
    result = graph.invoke({
        "query": "medicine to cure my disease",
        "step_log": []
    })
    
    assert "medical_advice" in result["safety_flags"], f"Expected 'medical_advice' in safety_flags, got {result.get('safety_flags')}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])