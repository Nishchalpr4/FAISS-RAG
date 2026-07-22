"""
university_apis.py — Dummy REST API Server (Store Live Data)
=============================================================
Three fake endpoints simulating live store data that changes daily.
These can't be stored in FAISS because they're dynamic.

Run with:
    uvicorn apis.university_apis:app --port 8001 --reload

Test in browser:
    http://localhost:8001/api/weather
    http://localhost:8001/api/deals
    http://localhost:8001/api/stock?product=headphones
"""

import random
from datetime import date
from fastapi import FastAPI, Query

app = FastAPI(title="ShopEase Live Data API", version="1.0")


# ── Endpoint 1: Current Weather ────────────────────────────────────────────────
# Used to answer questions like "Will my delivery be delayed today?"
@app.get("/api/weather")
def get_weather():
    """Current weather — affects shipping estimates."""
    conditions = ["Clear", "Cloudy", "Heavy Rain", "Storm", "Sunny"]
    condition  = random.choice(conditions)
    delay_note = "Deliveries may be delayed today." if condition in ("Heavy Rain", "Storm") else "Deliveries on schedule."
    return {
        "date":               str(date.today()),
        "condition":          condition,
        "temperature_celsius": random.randint(18, 38),
        "shipping_note":      delay_note,
    }


# ── Endpoint 2: Today's Deals ──────────────────────────────────────────────────
# Randomly picks 2 active promotions from a fixed pool. Changes each request.
ALL_DEALS = [
    "10% off all keyboards this week — use code KEYS10",
    "Buy any hub, get free shipping — no minimum order",
    "Webcams at 15% off — limited stock available",
    "Laptop stands bundled with USB-C hubs for $49 (save $5)",
    "Flash sale: Wireless headphones at $39.99 today only",
    "Students get 20% off with a valid .edu email address",
]

@app.get("/api/deals")
def get_deals():
    """Today's active promotions."""
    return {
        "date":        str(date.today()),
        "active_deals": random.sample(ALL_DEALS, k=2),
    }


# ── Endpoint 3: Live Stock Check ───────────────────────────────────────────────
# Returns random stock level for a product. Simulates a warehouse inventory API.
@app.get("/api/stock")
def check_stock(product: str = Query(default="headphones", description="Product name to check")):
    """Check live stock availability for a product."""
    stock_count = random.randint(0, 50)
    status = "In Stock" if stock_count > 5 else ("Low Stock" if stock_count > 0 else "Out of Stock")
    return {
        "product": product,
        "units_available": stock_count,
        "status": status,
    }


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "ShopEase Live Data API",
        "endpoints": ["/api/weather", "/api/deals", "/api/stock?product=headphones"],
    }
