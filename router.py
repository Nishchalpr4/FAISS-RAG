"""
router.py — Question Router
============================
Decides WHERE to look for the answer to a user's question.

Two approaches are implemented here:
  1. keyword_router()  — simple, fast, zero-cost, fully explainable
  2. llm_router()      — smarter, handles edge cases, costs API tokens

In chat.py we use keyword_router() by default.
You can swap to llm_router() and compare the results.
"""

import os
import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Base URL of our dummy API server (must be running separately)
API_BASE = "http://localhost:8001"


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 1: KEYWORD ROUTER
# ══════════════════════════════════════════════════════════════════════════════
# How it works:
#   - We define a set of trigger keywords for each API endpoint.
#   - If the question contains any of those words → call that API.
#   - If none match → fall back to FAISS (static knowledge base).
#
# Pros: Instant, free, transparent, easy to debug
# Cons: Brittle — "Is it raining on campus?" won't match "weather" if user
#       writes "raining". Needs manual maintenance as you add more sources.

KEYWORD_MAP = {
    "weather": ["weather", "rain", "storm", "delivery delay", "shipping delay", "temperature"],
    "deals":   ["deal", "deals", "discount", "offer", "promo", "promotion", "sale", "coupon", "code"],
    "stock":   ["stock", "available", "availability", "in stock", "out of stock", "units", "inventory"],
}

def keyword_router(question: str) -> dict:
    """
    Returns a dict with:
      - 'source': 'faiss', 'api', or 'both'
      - 'api_endpoints': list of API URLs to call (if source is 'api' or 'both')
    """
    question_lower = question.lower()
    matched_endpoints = []

    for endpoint_name, keywords in KEYWORD_MAP.items():
        if any(kw in question_lower for kw in keywords):
            matched_endpoints.append(f"{API_BASE}/api/{endpoint_name}")

    if matched_endpoints:
        return {"source": "api", "api_endpoints": matched_endpoints}
    else:
        return {"source": "faiss", "api_endpoints": []}


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 2: LLM ROUTER (Function Calling)
# ══════════════════════════════════════════════════════════════════════════════
# How it works:
#   - We describe our available "tools" (FAISS, weather API, events API, canteen API)
#     as JSON schemas — this is the standard "function calling" format.
#   - We send the question + tool descriptions to the LLM.
#   - The LLM replies with which tool(s) to use, not with an answer.
#   - We then call those tools and get the real answer.
#
# Pros: Understands paraphrase ("Is it raining?" → calls weather),
#       handles complex multi-part questions gracefully
# Cons: Adds ~0.5-1s latency, costs tokens even for routing,
#       can hallucinate tool choices if prompt isn't precise

# Tool descriptions in the format LLMs understand (OpenAI-compatible)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the ShopEase knowledge base (FAISS) for questions about "
                "store policies, return policy, shipping, products, loyalty program, "
                "bulk orders, or any static store information."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get today's weather and check if deliveries/shipping will be delayed.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_deals",
            "description": "Get today's active deals, discounts, and promotional offers.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check if a product is in stock or available in the warehouse.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Maps LLM tool names → actual API endpoint URLs
TOOL_TO_URL = {
    "get_weather":  f"{API_BASE}/api/weather",
    "get_deals":    f"{API_BASE}/api/deals",
    "check_stock":  f"{API_BASE}/api/stock",
}

def llm_router(question: str) -> dict:
    """
    Uses Groq LLM with function calling to decide which source to use.
    Returns same format as keyword_router() for drop-in compatibility.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a router for a university help desk chatbot. "
                    "Based on the user's question, select the right tool(s) to call. "
                    "You MUST call at least one tool — never answer directly."
                ),
            },
            {"role": "user", "content": question},
        ],
        tools=TOOLS,
        tool_choice="required",  # Force the LLM to always pick a tool
    )

    # Parse which tools the LLM chose
    tool_calls = response.choices[0].message.tool_calls or []
    chosen_tools = [tc.function.name for tc in tool_calls]

    api_endpoints = []
    use_faiss = False

    for tool_name in chosen_tools:
        if tool_name == "search_knowledge_base":
            use_faiss = True
        elif tool_name in TOOL_TO_URL:
            api_endpoints.append(TOOL_TO_URL[tool_name])

    if use_faiss and api_endpoints:
        return {"source": "both", "api_endpoints": api_endpoints}
    elif api_endpoints:
        return {"source": "api", "api_endpoints": api_endpoints}
    else:
        return {"source": "faiss", "api_endpoints": []}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED: Call API endpoints
# ══════════════════════════════════════════════════════════════════════════════
def fetch_api_data(endpoints: list) -> list:
    """
    Calls each API endpoint and returns a list of result dicts.
    Uses httpx for simple HTTP GET requests.
    """
    results = []
    with httpx.Client(timeout=5.0) as client:
        for url in endpoints:
            try:
                resp = client.get(url)
                resp.raise_for_status()
                results.append({"url": url, "data": resp.json()})
            except Exception as e:
                results.append({"url": url, "error": str(e)})
    return results
