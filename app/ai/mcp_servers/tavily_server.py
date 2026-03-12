import os
import sys
import json
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from app.core.config import settings


mcp = FastMCP("Tavily")


@mcp.tool()
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web for current news, recent events, or information you don't know.

    USE THIS TOOL WHEN:
    - User asks about recent news, current events, or today's information
    - User asks about something you're uncertain about or don't have knowledge of
    - User asks about real-time data (stock prices, weather, sports scores, etc.)
    - User wants to verify or fact-check information

    DO NOT USE THIS TOOL WHEN:
    - You already have the information from your training or previous tool results
    - The question is about general knowledge that doesn't require current data
    - You're asked for opinions, advice, or creative tasks

    Args:
        query: The search query - be specific and include relevant context
        max_results: Number of results to return (default: 5)

    Returns:
        JSON with search results including titles, URLs, content snippets, and relevance scores
    """
    try:
        # Try to get API key from environment first
        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            try:
                api_key = settings.tavily_api_key
            except Exception:
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
