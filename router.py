import os
import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_BASE = "http://localhost:8001"

KEYWORD_MAP = {
    "weather": ["weather", "rain", "storm", "delivery", "shipping"],
    "deals":   ["deal", "discount", "offer", "promo", "sale", "coupon"],
    "stock":   ["stock", "available", "inventory", "units"]
}

def keyword_router(question: str) -> dict:
    """Keyword-based routing (Fast & Free)."""
    question_lower = question.lower()
    matched = []

    for endpoint, keywords in KEYWORD_MAP.items():
        if any(kw in question_lower for kw in keywords):
            matched.append(f"{API_BASE}/api/{endpoint}")

    if matched:
        return {"source": "api", "api_endpoints": matched}
    return {"source": "faiss", "api_endpoints": []}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search store policies, products, and company facts in FAISS.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather and shipping delay info.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_deals",
            "description": "Get active promotions and discounts.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check inventory availability for products.",
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
    """LLM Function Calling routing (Smart & Contextual)."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Pick the correct tool for the user question."},
            {"role": "user", "content": question}
        ],
        tools=TOOLS,
        tool_choice="required"
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    chosen = [tc.function.name for tc in tool_calls]

    endpoints = [TOOL_TO_URL[name] for name in chosen if name in TOOL_TO_URL]
    use_faiss = "search_knowledge_base" in chosen

    if use_faiss and endpoints:
        return {"source": "both", "api_endpoints": endpoints}
    elif endpoints:
        return {"source": "api", "api_endpoints": endpoints}
    return {"source": "faiss", "api_endpoints": []}


def fetch_api_data(endpoints: list) -> list:
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
