import json
import sys
from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from app.core.config import settings  # noqa: E402

mcp = FastMCP("FormFiller")


@mcp.tool()
def fill_form(natural_language_input: str) -> str:
    """
    Parse natural language input and extract structured incident report information.

    This tool takes a natural language description of an incident (in Vietnamese or English)
    and extracts structured data including reporter info, incident details, location, severity, etc.

    Args:
        natural_language_input: Natural language description of the incident.
            Example: "Giúp tôi điền form. Hôm nay 10h tôi thấy rò rỉ nước ở tầng 3"

    Returns:
        JSON string with structured incident report data including:
        - reporter_name, reporter_email, incident_date, incident_time
        - location, severity, category, title, description
        - witnesses (optional), immediate_action (optional)
    """
    try:
        # Get API key
        api_key = settings.gemini_api_key
        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        # Initialize Gemini client. Application owns retries; attempts=1
        # disables the SDK's own retry of the original request.
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        )

        # Get current date as reference
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        # Create prompt for Gemini to extract structured data
        prompt = f"""Fill incident reports by extracting structured details from the user's input.

Current date: {current_date}
Current time: {current_time}

User input: {natural_language_input}

Extract the following information and return ONLY a valid JSON object (no markdown, no explanation):

{{
  "reporter_name": "string or null if not mentioned",
  "reporter_email": "string or null if not mentioned",
  "incident_date": "YYYY-MM-DD format or null (use current date if 'today' or 'hôm nay' mentioned)",
  "incident_time": "HH:MM format or null (extract from input like '10h' -> '10:00')",
  "location": "string describing the location or null",
  "severity": "low, medium, high, or critical based on the incident description",
  "category": "incident type (e.g., 'Water Leak', 'Fire', 'Safety', or 'Equipment Failure')",
  "title": "brief title summarizing the incident",
  "description": "detailed description of the incident",
  "witnesses": "string or null if not mentioned",
  "immediate_action": "string or null if not mentioned"
}}

Guidelines:
- If information is not provided, use null
- For severity: assess based on incident type (water leak = medium/high, fire = high/critical, etc.)
- For category: infer from the incident description
- For title: create a concise summary (max 100 chars)
- For description: expand the user's input into a clear incident description
- Handle both Vietnamese and English input
- For dates: if "hôm nay" or "today" -> use current date
- For times: convert "10h", "10 giờ", "10am" to "10:00" format

Return ONLY the JSON object, nothing else."""

        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )

        # Extract the response text
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Validate it's valid JSON
        parsed_json = json.loads(response_text)

        # Return the formatted JSON
        return json.dumps(parsed_json, indent=2, ensure_ascii=False)

    except json.JSONDecodeError as e:
        return json.dumps(
            {
                "error": f"Failed to parse LLM response as JSON: {str(e)}",
                "raw_response": response_text if "response_text" in locals() else None,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": f"Form filling failed: {str(e)}"}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
