"""
QUERY ROUTER & API CONNECTOR (router.py)

What this script does in simple terms:
1. Keyword Router: Looks for trigger words (e.g. "weather", "deal", "stock") in the user's question.
   If keywords match -> sends request to live FastAPI endpoints.
   Otherwise -> routes request to static FAISS vector search.
2. LLM Router: An intelligent alternative that uses AI Function Calling (Groq LLM) to analyze the intent
   of the question and automatically select the right tool or API.
3. fetch_api_data: Helper function using `httpx` to send web requests to live API endpoints and get back JSON data.
"""

import os       # Operating system module to access env variables
import httpx    # Asynchronous/Synchronous HTTP client for calling APIs
from groq import Groq             # Groq SDK for AI function calling
from dotenv import load_dotenv    # Loads secrets from .env file

load_dotenv()

# Base URL where our FastAPI live microservices run (started by python apis/main.py)
API_BASE = "http://localhost:8001"

# Module-level variable to cache the Groq client instance (avoids re-creating client on every call)
_groq_client: Groq | None = None

def _get_groq_client() -> Groq:
    """
    Returns a cached Groq client instance.
    Initialised lazily on the first function call.
    """
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ===================================================================
# ROUTER OPTION 1: KEYWORD ROUTER (Fast & Free, Rule-Based)
# ===================================================================
# Mapping of API endpoint names to lists of trigger keywords
KEYWORD_MAP = {
    "weather": ["weather", "rain", "storm", "delivery", "shipping"],  # Weather/Shipping questions
    "deals":   ["deal", "discount", "offer", "promo", "sale", "coupon"],  # Promotion questions
    "stock":   ["stock", "available", "inventory", "units"]  # Inventory questions
}

def keyword_router(question: str) -> dict:
    """
    Fast, rule-based decision engine:
    1. Converts question to lower case.
    2. Checks if any keyword matches.
    3. Returns list of matched API endpoints (or defaults to FAISS).
    """
    question_lower = question.lower()
    matched_apis   = []

    # Check question against keyword lists for weather, deals, and stock
    for endpoint, keywords in KEYWORD_MAP.items():
        if any(kw in question_lower for kw in keywords):
            matched_apis.append(f"{API_BASE}/api/{endpoint}")

    # If keywords matched live topics, route to live API endpoints
    if matched_apis:
        return {"source": "api", "api_endpoints": matched_apis}

    # Otherwise, fallback to static FAISS vector database
    return {"source": "faiss", "api_endpoints": []}


# ===================================================================
# ROUTER OPTION 2: LLM ROUTER (Smart AI Function Calling)
# ===================================================================
# Tools schema defining available tools for Groq LLM function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search static store policies, products, and company info in FAISS.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Check current weather and shipping delay status.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_deals",
            "description": "Get today's active promo deals and discounts.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check live product inventory availability.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# Mapping function names to live URL endpoints
TOOL_TO_URL = {
    "get_weather":  f"{API_BASE}/api/weather",
    "get_deals":    f"{API_BASE}/api/deals",
    "check_stock":  f"{API_BASE}/api/stock"
}

def llm_router(question: str) -> dict:
    """
    Intelligent AI router using Groq Function Calling:
    Asks the AI model to pick the best tool(s) based on semantic meaning.
    """
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Select the correct tool for the user question."},
            {"role": "user",   "content": question}
        ],
        tools=TOOLS,
        tool_choice="required"  # Forces model to select at least one tool
    )

    # Extract chosen tool calls from model response
    tool_calls    = resp.choices[0].message.tool_calls or []
    chosen_tools  = [tc.function.name for tc in tool_calls]

    endpoints  = [TOOL_TO_URL[name] for name in chosen_tools if name in TOOL_TO_URL]
    use_faiss  = "search_knowledge_base" in chosen_tools

    # Determine data source target based on AI choices
    if use_faiss and endpoints:
        return {"source": "both",  "api_endpoints": endpoints}
    elif endpoints:
        return {"source": "api",   "api_endpoints": endpoints}
    return       {"source": "faiss","api_endpoints": []}


# ===================================================================
# HELPER: Make HTTP GET calls to live microservices
# ===================================================================
def fetch_api_data(endpoints: list) -> list:
    """
    Makes HTTP GET requests to live API URLs and returns their response data.
    Handles network errors gracefully (connection failures, timeouts, etc.).
    """
    results = []
    # Use httpx Client with a 5-second timeout safeguard
    with httpx.Client(timeout=5.0) as client:
        for url in endpoints:
            try:
                r = client.get(url)
                r.raise_for_status()  # Check for 4xx / 5xx HTTP error statuses
                results.append({"url": url, "data": r.json()})
            except httpx.ConnectError:
                # Triggers if the FastAPI server (apis/main.py) is not currently running
                results.append({"url": url, "error": f"Cannot connect to {url}. Is the API server running? (python apis/main.py)"})
            except httpx.TimeoutException:
                # Triggers if the API server takes longer than 5s to respond
                results.append({"url": url, "error": f"Request to {url} timed out after 5s."})
            except Exception as e:
                results.append({"url": url, "error": str(e)})
    return results

