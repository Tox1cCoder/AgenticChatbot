import os
import sys
import json
from pathlib import Path

# Add parent directories to path to import from app
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

mcp = FastMCP("Tavily")


@mcp.tool()
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily API for real-time information"""
    try:
        # Try to get API key from environment first
        api_key = os.getenv("TAVILY_API_KEY")

        # If not in environment, try to load from settings
        if not api_key:
            try:
                from app.core.config import settings

                api_key = settings.tavily_api_key
            except Exception as e:
                pass

        if not api_key:
            return json.dumps(
                {
                    "error": "TAVILY_API_KEY not configured. Please set it in environment or config.py"
                }
            )

        # Initialize Tavily client
        tavily_client = TavilyClient(api_key=api_key)

        # Perform search
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_images=True,
            include_image_descriptions=True,
        )

        # Format results
        if "results" in response:
            formatted_results = []
            for idx, result in enumerate(response["results"], 1):
                formatted_results.append(
                    {
                        "index": idx,
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0),
                    }
                )

            # Extract images from response
            images = []
            if "images" in response:
                for img in response["images"]:
                    images.append(
                        {
                            "url": img.get("url", ""),
                            "description": img.get("description", ""),
                        }
                    )

            return json.dumps(
                {
                    "query": query,
                    "answer": response.get("answer", ""),
                    "images": images,
                    "results": formatted_results,
                    "total_results": len(formatted_results),
                },
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "query": query,
                    "answer": "",
                    "images": [],
                    "results": [],
                    "total_results": 0,
                }
            )

    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
