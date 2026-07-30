"""
LIVE DATA MICROSERVICES (apis/main.py)

What this script does in simple terms:
1. Creates a web server using FastAPI running on port 8001.
2. Serves 3 live API endpoints that simulate real-time dynamic data:
   - GET /api/weather: Returns current weather and delivery delays.
   - GET /api/deals: Returns active promotional coupon codes and store offers.
   - GET /api/stock: Returns real-time product inventory count.
"""

import random              # For generating dynamic random sample data
from datetime import date   # For fetching current server date
from fastapi import FastAPI # Web framework for building fast REST APIs

# Initialize the FastAPI backend application
app = FastAPI(title="Live Data Microservices")


# ── 1. LIVE WEATHER API ENDPOINT ───────────────────────────────────────────────
# Returns random weather conditions and delivery status notes
@app.get("/api/weather")
def get_weather():
    """
    Returns today's date, weather status (Clear/Rain/Sunny), and shipping notes.
    URL: http://localhost:8001/api/weather
    """
    return {
        "date": str(date.today()),
        "condition": random.choice(["Clear", "Rain", "Sunny"]),
        "shipping_note": "Deliveries might be slightly delayed." if random.choice([True, False]) else "Deliveries on schedule."
    }


# ── 2. LIVE DEALS & PROMOTIONS API ENDPOINT ─────────────────────────────────────
# Returns today's active coupon codes and discount offers
@app.get("/api/deals")
def get_deals():
    """
    Returns list of active discount codes and promotions.
    URL: http://localhost:8001/api/deals
    """
    return {
        "date": str(date.today()),
        "active_deals": ["10% off keyboards - code KEYS10", "Free shipping on orders above $50"]
    }


# ── 3. LIVE STOCK INVENTORY API ENDPOINT ───────────────────────────────────────
# Returns real-time stock availability numbers for a specified product
@app.get("/api/stock")
def check_stock(product: str = "headphones"):
    """
    Returns dynamic unit inventory count for a given product.
    URL: http://localhost:8001/api/stock?product=headphones
    """
    return {
        "product": product,
        "units_available": random.randint(5, 50),
        "status": "In Stock"
    }

