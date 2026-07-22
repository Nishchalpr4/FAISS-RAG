import os
import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Base URL where our live FastAPI service runs
API_BASE = "http://localhost:8001"

# -------------------------------------------------------------------
# ROUTER OPTION 1: KEYWORD ROUTER (Fast & Free)
# Checks if the question contains specific keywords.
# -------------------------------------------------------------------
KEYWORD_MAP = {
    "weather": ["weather", "rain", "storm", "delivery", "shipping"],
    "deals":   ["deal", "discount", "offer", "promo", "sale", "coupon"],
    "stock":   ["stock", "available", "inventory", "units"]
}

def keyword_router(question: str) -> dict:
    """If question has live data keywords -> call API. Else -> search FAISS database."""
    question_lower = question.lower()
    matched_apis = []

    # Check if user asked for weather, deals, or stock
    for endpoint, keywords in KEYWORD_MAP.items():
        if any(kw in question_lower for kw in keywords):
            matched_apis.append(f"{API_BASE}/api/{endpoint}")

    if matched_apis:
        return {"source": "api", "api_endpoints": matched_apis}
    
    # Default to static vector database
    return {"source": "faiss", "api_endpoints": []}


# -------------------------------------------------------------------
# ROUTER OPTION 2: LLM ROUTER (Smart Function Calling)
# Asks the Groq AI model to choose which tool to use.
# -------------------------------------------------------------------
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

TOOL_TO_URL = {
    "get_weather": f"{API_BASE}/api/weather",
    "get_deals": f"{API_BASE}/api/deals",
    "check_stock": f"{API_BASE}/api/stock"
}

def llm_router(question: str) -> dict:
    """Uses Groq AI model to intelligently pick the tool."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Select the correct tool for the user question."},
            {"role": "user", "content": question}
        ],
        tools=TOOLS,
        tool_choice="required"
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    chosen_tools = [tc.function.name for tc in tool_calls]

    endpoints = [TOOL_TO_URL[name] for name in chosen_tools if name in TOOL_TO_URL]
    use_faiss = "search_knowledge_base" in chosen_tools

    if use_faiss and endpoints:
        return {"source": "both", "api_endpoints": endpoints}
    elif endpoints:
        return {"source": "api", "api_endpoints": endpoints}
    return {"source": "faiss", "api_endpoints": []}


# -------------------------------------------------------------------
# HELPER: Make HTTP GET call to live APIs
# -------------------------------------------------------------------
def fetch_api_data(endpoints: list) -> list:
    """Calls external API endpoints and returns their JSON data."""
    results = []
    with httpx.Client(timeout=5.0) as client:
        for url in endpoints:
            try:
                r = client.get(url)
                r.raise_for_status()
                results.append({"url": url, "data": r.json()})
            except Exception as e:
                results.append({"url": url, "error": str(e)})
    return results
