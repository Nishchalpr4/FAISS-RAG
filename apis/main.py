import random
from datetime import date
from fastapi import FastAPI

# Create the FastAPI backend application
app = FastAPI(title="Live Data Microservices")

# 1. LIVE WEATHER API
# Returns random weather condition and shipping delay information
@app.get("/api/weather")
def get_weather():
    return {
        "date": str(date.today()),
        "condition": random.choice(["Clear", "Rain", "Sunny"]),
        "shipping_note": "Deliveries might be slightly delayed." if random.choice([True, False]) else "Deliveries on schedule."
    }

# 2. LIVE DEALS API
# Returns today's active promotion codes and discounts
@app.get("/api/deals")
def get_deals():
    return {
        "date": str(date.today()),
        "active_deals": ["10% off keyboards - code KEYS10", "Free shipping on orders above $50"]
    }

# 3. LIVE STOCK INVENTORY API
# Returns current product inventory numbers
@app.get("/api/stock")
def check_stock(product: str = "headphones"):
    return {
        "product": product,
        "units_available": random.randint(5, 50),
        "status": "In Stock"
    }
