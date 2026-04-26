import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Config
n_records = 5000
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)

categories = {
    "Electronics":    {"brands": ["Samsung","Apple","Sony","LG","Xiaomi"],        "price_range": (5000, 80000)},
    "Clothing":       {"brands": ["Zara","H&M","Levi's","Nike","Adidas"],         "price_range": (300, 5000)},
    "Grocery":        {"brands": ["Amul","Nestle","ITC","Britannia","Patanjali"],  "price_range": (20, 1500)},
    "Home & Kitchen": {"brands": ["Prestige","Philips","Bosch","Usha","Bajaj"],   "price_range": (500, 20000)},
    "Sports":         {"brands": ["Decathlon","Nike","Puma","Adidas","Yonex"],    "price_range": (200, 8000)},
    "Beauty":         {"brands": ["Lakme","Maybelline","L'Oreal","Nykaa","Dove"], "price_range": (100, 3000)},
}

cities = ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Pune","Kolkata","Ahmedabad","Jaipur","Lucknow"]
channels = ["In-Store", "Online", "Mobile App"]
payment_methods = ["UPI", "Credit Card", "Debit Card", "Cash", "EMI"]
customer_segments = ["Regular", "Premium", "New", "VIP"]

def seasonal_multiplier(date):
    m = date.month
    if m in [10, 11, 12]: return 1.6   # Diwali / festive
    if m in [7, 8]:       return 1.2   # Monsoon sale
    if m in [1, 2]:       return 0.85  # post-holiday dip
    return 1.0

records = []
order_id = 1001

for _ in range(n_records):
    date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    category = random.choice(list(categories.keys()))
    cat_data = categories[category]
    brand = random.choice(cat_data["brands"])
    lo, hi = cat_data["price_range"]
    unit_price = round(random.uniform(lo, hi), 2)
    quantity = random.choices([1,2,3,4,5], weights=[50,25,12,8,5])[0]
    discount_pct = random.choices([0,5,10,15,20,25,30], weights=[30,20,20,12,10,5,3])[0]
    revenue = round(unit_price * quantity * (1 - discount_pct/100) * seasonal_multiplier(date), 2)
    profit_margin = round(random.uniform(0.08, 0.35), 3)
    profit = round(revenue * profit_margin, 2)
    city = random.choice(cities)
    channel = random.choices(channels, weights=[40,40,20])[0]
    payment = random.choices(payment_methods, weights=[35,25,20,12,8])[0]
    segment = random.choices(customer_segments, weights=[50,20,20,10])[0]
    returned = random.choices([0,1], weights=[92,8])[0]
    rating = round(random.uniform(2.5, 5.0), 1) if random.random() > 0.3 else None

    records.append({
        "order_id": f"ORD{order_id}",
        "date": date.strftime("%Y-%m-%d"),
        "category": category,
        "brand": brand,
        "unit_price": unit_price,
        "quantity": quantity,
        "discount_pct": discount_pct,
        "revenue": revenue,
        "profit": profit,
        "profit_margin_pct": round(profit_margin*100, 1),
        "city": city,
        "channel": channel,
        "payment_method": payment,
        "customer_segment": segment,
        "returned": returned,
        "customer_rating": rating,
    })
    order_id += 1

df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
df.to_csv("retail_sales_data.csv", index=False)
print(f"Generated {len(df)} records → retail_sales_data.csv")
print(df.head(3).to_string())