import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Every possible alias a column might have in the wild.
# Key = internal standard name the app uses.
# Value = list of possible names in the uploaded CSV (all lowercase).
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "date": [
        "date", "order_date", "transaction_date", "sale_date",
        "invoice_date", "purchase_date", "created_at", "created_date",
        "order_placed", "date_of_sale", "dt", "datetime", "timestamp",
    ],
    "revenue": [
        "revenue", "sales", "total_sales", "total_revenue", "amount",
        "total_amount", "sale_amount", "gross_revenue", "net_sales",
        "total_price", "sales_amount", "invoice_amount", "price",
        "total", "value", "gmv",
    ],
    "profit": [
        "profit", "net_profit", "gross_profit", "profit_amount",
        "earnings", "income", "margin_amount", "net_income",
    ],
    "quantity": [
        "quantity", "qty", "units", "units_sold", "quantity_sold",
        "items", "count", "num_items", "volume", "pieces",
    ],
    "category": [
        "category", "product_category", "item_category", "dept",
        "department", "segment", "product_type", "type", "cat",
        "product_group", "group", "subcategory", "sub_category",
    ],
    "brand": [
        "brand", "brand_name", "manufacturer", "vendor", "supplier",
        "make", "label", "product_brand",
    ],
    "city": [
        "city", "location", "region", "store_location", "store_city",
        "branch", "area", "zone", "territory", "market", "geo",
        "store", "outlet",
    ],
    "channel": [
        "channel", "sales_channel", "platform", "source",
        "order_source", "medium", "channel_type", "mode",
    ],
    "payment_method": [
        "payment_method", "payment", "payment_type", "pay_method",
        "transaction_type", "pay_mode", "payment_mode",
    ],
    "customer_segment": [
        "customer_segment", "segment", "customer_type", "tier",
        "customer_tier", "membership", "loyalty_tier", "customer_class",
    ],
    "returned": [
        "returned", "is_returned", "return", "return_flag",
        "refunded", "is_refund", "cancelled", "chargeback",
    ],
    "customer_rating": [
        "customer_rating", "rating", "review_score", "score",
        "stars", "review", "satisfaction", "nps",
    ],
    "discount_pct": [
        "discount_pct", "discount", "discount_percent",
        "discount_rate", "promo_discount", "offer_pct",
    ],
    "order_id": [
        "order_id", "id", "transaction_id", "invoice_id",
        "sale_id", "record_id", "order_no", "txn_id",
    ],
    "unit_price": [
        "unit_price", "price", "selling_price", "mrp",
        "list_price", "item_price", "cost_price",
    ],
    "cost": [
        "cost", "cogs", "cost_of_goods", "cost_price",
        "purchase_price", "unit_cost",
    ],
}

# Columns that are truly required — without a date and at least one numeric column we can't do anything
REQUIRED = ["date"]


def detect_and_map(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Given any DataFrame:
    1. Normalise column names (lowercase, strip, spaces→underscore)
    2. Map columns to standard internal names using COLUMN_ALIASES
    3. Fill missing optional columns with sensible defaults
    4. Return (mapped_df, mapping_report)
    """
    # Step 1 — normalise column names
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-/\\]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    original_cols = list(df.columns)
    mapping_report = {}   # standard_name -> original_col_name or "generated"

    # Step 2 — map aliases
    for standard_name, aliases in COLUMN_ALIASES.items():
        if standard_name in df.columns:
            mapping_report[standard_name] = standard_name
            continue
        for alias in aliases:
            if alias in df.columns:
                df = df.rename(columns={alias: standard_name})
                mapping_report[standard_name] = alias
                break

    # Step 3 — check required columns
    missing_required = [c for c in REQUIRED if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Could not find a date column in your CSV.\n\n"
            f"Your columns: {original_cols}\n\n"
            f"Please rename your date column to one of: {COLUMN_ALIASES['date']}"
        )

    # Step 4 — smart fill for missing revenue
    if "revenue" not in df.columns:
        if "unit_price" in df.columns and "quantity" in df.columns:
            df["revenue"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0) * \
                            pd.to_numeric(df["quantity"],   errors="coerce").fillna(1)
            mapping_report["revenue"] = "calculated: unit_price × quantity"
        else:
            # Find any numeric column and use it as revenue proxy
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                df["revenue"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce").fillna(0)
                mapping_report["revenue"] = f"proxy: {numeric_cols[0]}"
            else:
                df["revenue"] = 1.0
                mapping_report["revenue"] = "generated: constant 1"

    # Step 5 — smart fill for missing profit
    if "profit" not in df.columns:
        if "cost" in df.columns:
            df["profit"] = df["revenue"] - pd.to_numeric(df["cost"], errors="coerce").fillna(0)
            mapping_report["profit"] = "calculated: revenue − cost"
        else:
            df["profit"] = df["revenue"] * 0.20   # assume 20% margin
            mapping_report["profit"] = "estimated: 20% of revenue"

    # Step 6 — fill remaining optional columns with defaults
    defaults = {
        "quantity":         1,
        "discount_pct":     0,
        "returned":         0,
        "customer_rating":  None,
        "category":         "General",
        "brand":            "Unknown",
        "city":             "Unknown",
        "channel":          "Direct",
        "payment_method":   "Unknown",
        "customer_segment": "General",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
            mapping_report[col] = f"generated: default '{default}'"

    # Step 7 — generate order_id if missing
    if "order_id" not in df.columns:
        df["order_id"] = [f"ROW{i+1}" for i in range(len(df))]
        mapping_report["order_id"] = "generated: row index"

    # Step 8 — coerce numeric types
    for col in ["revenue", "profit", "quantity", "discount_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["returned"] = pd.to_numeric(df["returned"], errors="coerce").fillna(0).astype(int)

    # Clamp returned to 0/1 (some CSVs use True/False or Yes/No)
    if df["returned"].dtype == object:
        df["returned"] = df["returned"].str.lower().isin(["1","yes","true","y"]).astype(int)
    df["returned"] = df["returned"].clip(0, 1)

    return df, mapping_report


def mapping_summary(report: dict) -> str:
    """Human-readable string of what was mapped, renamed, or generated."""
    lines = []
    for std, src in report.items():
        if src == std:
            lines.append(f"  ✓ {std}")
        elif src.startswith("generated") or src.startswith("estimated") or src.startswith("calculated") or src.startswith("proxy"):
            lines.append(f"  ~ {std}  ({src})")
        else:
            lines.append(f"  ↳ {std}  (from '{src}')")
    return "\n".join(lines)