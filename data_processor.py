import pandas as pd
import numpy as np
from column_mapper import detect_and_map, mapping_summary


def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df, mapping_report = detect_and_map(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["date"])
    if len(df) == 0:
        raise ValueError("After parsing dates, no valid rows remain. Check your date column.")
    df["month"]       = df["date"].dt.to_period("M").astype(str)
    df["month_name"]  = df["date"].dt.strftime("%b %Y")
    df["year"]        = df["date"].dt.year
    df["quarter"]     = df["date"].dt.to_period("Q").astype(str)
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"]  = df["date"].dt.dayofweek >= 5
    df.attrs["mapping_report"] = mapping_report
    df.attrs["mapping_summary"] = mapping_summary(mapping_report)
    return df


def get_kpis(df):
    total_revenue = df["revenue"].sum()
    total_profit  = df["profit"].sum()
    total_orders  = len(df)
    avg_order_val = df["revenue"].mean() if total_orders else 0
    profit_margin = (total_profit / total_revenue * 100) if total_revenue else 0
    return_rate   = df["returned"].mean() * 100
    avg_rating_raw = df["customer_rating"].dropna().mean() if "customer_rating" in df.columns else None
    avg_rating = round(float(avg_rating_raw), 2) if pd.notna(avg_rating_raw) else "N/A"
    cat_rev = df.groupby("category")["revenue"].sum()
    top_category = cat_rev.idxmax() if len(cat_rev) > 0 else "N/A"
    return {
        "total_revenue":   round(total_revenue, 2),
        "total_profit":    round(total_profit, 2),
        "total_orders":    total_orders,
        "avg_order_value": round(avg_order_val, 2),
        "profit_margin":   round(profit_margin, 2),
        "return_rate":     round(return_rate, 2),
        "avg_rating":      avg_rating,
        "top_category":    top_category,
    }


def monthly_revenue(df):
    return (
        df.groupby("month")
          .agg(revenue=("revenue","sum"), orders=("order_id","count"), profit=("profit","sum"))
          .reset_index().sort_values("month")
    )


def category_analysis(df):
    has_rating = df["customer_rating"].notna().any()
    agg = {"revenue":("revenue","sum"),"profit":("profit","sum"),"orders":("order_id","count")}
    if has_rating:
        agg["avg_rating"] = ("customer_rating","mean")
    result = df.groupby("category").agg(**agg).reset_index().sort_values("revenue", ascending=False)
    if "avg_rating" not in result.columns:
        result["avg_rating"] = None
    return result


def city_analysis(df):
    return df.groupby("city").agg(revenue=("revenue","sum"), orders=("order_id","count")).reset_index().sort_values("revenue", ascending=False)


def channel_analysis(df):
    return df.groupby("channel").agg(revenue=("revenue","sum"), orders=("order_id","count")).reset_index()


def top_brands(df, n=10):
    return df.groupby(["category","brand"]).agg(revenue=("revenue","sum"), orders=("order_id","count")).reset_index().sort_values("revenue", ascending=False).head(n)


def data_quality_score(df):
    missing_pct   = df.isnull().mean().mean() * 100
    duplicate_pct = df.duplicated().mean() * 100
    neg_revenue   = (df["revenue"] < 0).mean() * 100
    return round(max(0, 100 - missing_pct*2 - duplicate_pct*5 - neg_revenue*3), 1)


def anomaly_detection(df):
    from sklearn.ensemble import IsolationForest
    feature_cols = [c for c in ["revenue","quantity","discount_pct"] if df[c].std() > 0]
    df = df.copy()
    if not feature_cols:
        df["is_anomaly"] = False
        return df
    df["is_anomaly"] = IsolationForest(contamination=0.05, random_state=42).fit_predict(df[feature_cols].fillna(0)) == -1
    return df


def customer_segment_analysis(df):
    return df.groupby("customer_segment").agg(revenue=("revenue","sum"), orders=("order_id","count"), avg_order=("revenue","mean"), profit=("profit","sum")).reset_index().sort_values("revenue", ascending=False)


def payment_analysis(df):
    return df.groupby("payment_method").agg(revenue=("revenue","sum"), orders=("order_id","count")).reset_index()