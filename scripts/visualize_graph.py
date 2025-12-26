"""
Script to visualize the LangGraph multi-agent workflow.
Generates both PNG and Mermaid diagram outputs.
"""
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.graph import MultiAgentWorkflow


def visualize_graph():
    """Generate visualization of the multi-agent graph."""
    
    # Create workflow instance (without checkpointer for visualization)
    # We need to mock the required dependencies
    class MockQdrantClient:
        pass
    
    class MockEmbeddingModel:
        def encode(self, *args, **kwargs):
            return []
    
    workflow = MultiAgentWorkflow(
        qdrant_client=MockQdrantClient(),
        embedding_model=MockEmbeddingModel(),
        checkpointer=None,
        document_repository=None,
    )
    
    # Get the compiled graph
    compiled_graph = workflow.graph
    
    # Generate Mermaid diagram
    print("=" * 60)
    print("MERMAID DIAGRAM")
    print("=" * 60)
    mermaid_png = compiled_graph.get_graph().draw_mermaid()
    print(mermaid_png)
    print()
    
    # Save Mermaid to file
    mermaid_path = os.path.join(os.path.dirname(__file__), "graph_visualization.mmd")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_png)
    print(f"Mermaid diagram saved to: {mermaid_path}")
    
    # Try to generate PNG (requires graphviz)
    try:
        png_path = os.path.join(os.path.dirname(__file__), "graph_visualization.png")
        png_data = compiled_graph.get_graph().draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_data)
        print(f"PNG diagram saved to: {png_path}")
    except Exception as e:
        print(f"Could not generate PNG (may need graphviz/pyppeteer): {e}")
        print("You can paste the Mermaid code above into https://mermaid.live/ to view it.")


if __name__ == "__main__":
    visualize_graph()
