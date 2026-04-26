import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from data_processor import (
    load_and_clean, get_kpis, monthly_revenue, category_analysis,
    city_analysis, channel_analysis, top_brands, data_quality_score,
    anomaly_detection, customer_segment_analysis, payment_analysis
)
from forecasting import forecast_revenue, growth_metrics, detect_seasonality, trend_summary
from ai_insights import generate_insights, quick_qa

st.set_page_config(
    page_title="RetailIQ — Automated Insights",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .insight-box {
        background: linear-gradient(135deg,#667eea0d,#764ba20d);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        white-space: pre-line;
    }
    .quality-badge {
        display:inline-block; padding:4px 14px;
        border-radius:20px; font-weight:600; font-size:.9rem;
    }
    .map-box {
        background:#f8f9fa; border-radius:8px;
        padding:.8rem 1rem; font-family:monospace;
        font-size:.82rem; line-height:1.7;
        border:1px solid #e9ecef;
    }
    section[data-testid="stSidebar"] { background: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RetailIQ")
    st.markdown("*Automated Data Insights*")
    st.divider()
    uploaded = st.file_uploader("Upload any CSV", type=["csv"])
    api_key  = st.text_input("Claude API Key (optional)", type="password",
                              help="Get one free at console.anthropic.com")
    st.divider()
    st.markdown("**Filters**")

    DATA_PATH = uploaded if uploaded else os.path.join(BASE_DIR, "retail_sales_data.csv")

    if not uploaded and not os.path.exists(DATA_PATH):
        st.error(
            "No data file found.\n\n"
            "Either upload a CSV above, or run:\n```\npython generate_data.py\n```"
        )
        st.stop()

    @st.cache_data
    def load_data(path):
        return load_and_clean(path)

    try:
        df_raw = load_data(DATA_PATH)
    except ValueError as e:
        st.error(f"Could not load your file:\n\n{e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading file:\n\n{e}")
        st.stop()

    years      = sorted(df_raw["year"].unique())
    categories = sorted(df_raw["category"].unique())
    cities     = sorted(df_raw["city"].unique())
    channels   = sorted(df_raw["channel"].unique())

    sel_years   = st.multiselect("Year",     years,      default=years)
    sel_cats    = st.multiselect("Category", categories, default=categories)
    sel_cities  = st.multiselect("City",     cities,     default=cities)
    sel_channel = st.multiselect("Channel",  channels,   default=channels)

df = df_raw[
    df_raw["year"].isin(sel_years) &
    df_raw["category"].isin(sel_cats) &
    df_raw["city"].isin(sel_cities) &
    df_raw["channel"].isin(sel_channel)
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("RetailIQ — Automated Sales Intelligence")
st.caption(
    f"Analysing {len(df):,} rows  |  "
    f"{df['date'].min().strftime('%d %b %Y')} → {df['date'].max().strftime('%d %b %Y')}"
)

# Column mapping expander — always visible so user knows what was detected
mapping_txt = df_raw.attrs.get("mapping_summary", "")
with st.expander("Column mapping — what was detected in your file", expanded=bool(uploaded)):
    if mapping_txt:
        st.markdown(f'<div class="map-box">{mapping_txt}</div>', unsafe_allow_html=True)
        st.caption(
            "✓ = column found as-is  |  ↳ = renamed from your column  |  ~ = auto-calculated or default"
        )
    else:
        st.write("No mapping info available.")

dq = data_quality_score(df)
color = "#28a745" if dq >= 85 else ("#ffc107" if dq >= 65 else "#dc3545")
st.markdown(
    f'<span class="quality-badge" style="background:{color}22;color:{color};border:1px solid {color}">'
    f'Data Quality Score: {dq}/100</span>', unsafe_allow_html=True
)
st.divider()

# Helper — safe groupby chart (skip if column has only one unique value like "Unknown")
def has_real_values(series, min_unique=2):
    return series.nunique() >= min_unique

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Overview", "Sales Trends", "Category & Brand",
                "Geographic", "Forecasting", "AI Insights", "Anomaly Detection"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    kpis = get_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Revenue",   f"₹{kpis['total_revenue']/1e6:.2f}M")
    with c2: st.metric("Total Profit",    f"₹{kpis['total_profit']/1e6:.2f}M",  f"{kpis['profit_margin']}% margin")
    with c3: st.metric("Total Orders",    f"{kpis['total_orders']:,}",           f"AOV ₹{kpis['avg_order_value']:,.0f}")
    with c4: st.metric("Avg Rating",      str(kpis['avg_rating']),               f"Return rate {kpis['return_rate']}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by channel")
        ch = channel_analysis(df)
        if has_real_values(df["channel"]):
            fig = px.pie(ch, values="revenue", names="channel", hole=0.45,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Channel data not available in your file.")

    with col2:
        st.subheader("Customer segments")
        if has_real_values(df["customer_segment"]):
            seg = customer_segment_analysis(df)
            fig = px.bar(seg, x="customer_segment", y="revenue", color="customer_segment",
                         text_auto=".2s", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Customer segment data not available in your file.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Payment methods")
        if has_real_values(df["payment_method"]):
            pay = payment_analysis(df)
            fig = px.bar(pay.sort_values("revenue"), x="revenue", y="payment_method",
                         orientation="h", text_auto=".2s",
                         color="revenue", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Payment method data not available in your file.")

    with col4:
        st.subheader("Revenue vs profit by category")
        cat = category_analysis(df)
        if len(cat) > 1:
            fig = px.scatter(cat, x="revenue", y="profit", size="orders",
                             color="category", text="category",
                             color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more than one category for this chart.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SALES TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Monthly revenue trend")
    monthly = monthly_revenue(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["revenue"],
                             mode="lines+markers", name="Revenue",
                             line=dict(color="#667eea", width=2.5),
                             fill="tozeroy", fillcolor="rgba(102,126,234,0.08)"))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["profit"],
                             mode="lines+markers", name="Profit",
                             line=dict(color="#28a745", width=2, dash="dot")))
    fig.update_layout(hovermode="x unified", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly order volume")
        fig = px.bar(monthly, x="month", y="orders", color="orders",
                     color_continuous_scale="Purples")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Seasonality pattern")
        seasonal = detect_seasonality(df)
        fig = px.line(seasonal, x="month_label", y="revenue",
                      markers=True, color_discrete_sequence=["#f093fb"])
        fig.update_traces(line_width=2.5)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revenue heatmap — day of week vs month")
    df["month_short"] = df["date"].dt.strftime("%b")
    pivot = df.groupby(["day_of_week","month_short"])["revenue"].mean().unstack(fill_value=0)
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    if not pivot.empty:
        fig = px.imshow(pivot, color_continuous_scale="YlOrRd", aspect="auto",
                        labels=dict(color="Avg Revenue"))
        st.plotly_chart(fig, use_container_width=True)

    _, yoy = growth_metrics(df)
    if yoy:
        st.subheader("Year-over-year growth")
        yoy_df = pd.DataFrame(list(yoy.items()), columns=["Year","YoY Growth (%)"])
        fig = px.bar(yoy_df, x="Year", y="YoY Growth (%)",
                     color="YoY Growth (%)",
                     color_continuous_scale=["#dc3545","#ffffff","#28a745"],
                     color_continuous_midpoint=0, text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CATEGORY & BRAND
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    cat = category_analysis(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by category")
        fig = px.bar(cat, x="revenue", y="category", orientation="h",
                     color="revenue", text_auto=".2s", color_continuous_scale="Teal")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Profit margin by category")
        cat["margin_pct"] = (cat["profit"] / cat["revenue"].replace(0, np.nan) * 100).round(1).fillna(0)
        fig = px.bar(cat, x="category", y="margin_pct", text_auto=".1f",
                     color="margin_pct", color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=20)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Category revenue over time")
    cat_monthly = df.groupby(["month","category"])["revenue"].sum().reset_index()
    if len(cat_monthly["category"].unique()) > 1:
        fig = px.line(cat_monthly, x="month", y="revenue", color="category",
                      color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 brands by revenue")
    brands = top_brands(df, 10)
    if has_real_values(df["brand"]):
        fig = px.treemap(brands, path=["category","brand"], values="revenue",
                         color="revenue", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Brand data not available in your file.")

    st.subheader("Discount vs revenue impact")
    if df["discount_pct"].sum() > 0:
        disc = df.groupby("discount_pct").agg(revenue=("revenue","sum"), orders=("order_id","count")).reset_index()
        fig = px.bar(disc, x="discount_pct", y="revenue", color="orders",
                     labels={"discount_pct":"Discount %"}, color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No discount data found in your file.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — GEOGRAPHIC
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    city_df = city_analysis(df)

    if has_real_values(df["city"]):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue by city / region")
            fig = px.bar(city_df, x="city", y="revenue", color="revenue",
                         text_auto=".2s", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Orders by city / region")
            fig = px.pie(city_df, values="orders", names="city", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        if has_real_values(df["category"]):
            st.subheader("City vs category breakdown")
            city_cat = df.groupby(["city","category"])["revenue"].sum().reset_index()
            fig = px.bar(city_cat, x="city", y="revenue", color="category",
                         barmode="stack", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        if has_real_values(df["channel"]):
            st.subheader("Channel preference by city")
            city_ch = df.groupby(["city","channel"])["revenue"].sum().reset_index()
            fig = px.bar(city_ch, x="city", y="revenue", color="channel",
                         barmode="group", color_discrete_sequence=["#667eea","#f093fb","#4facfe"])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No city / location data found in your file. This tab shows geographic breakdowns.")
        monthly_fb = monthly_revenue(df)
        st.subheader("Monthly revenue (fallback)")
        fig = px.bar(monthly_fb, x="month", y="revenue", color="revenue",
                     color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Revenue forecast")
    st.info("Uses Facebook Prophet for time-series forecasting (falls back to linear trend if Prophet is not installed).")

    forecast_periods = st.slider("Forecast horizon (days)", 30, 180, 90, step=30)

    with st.spinner("Running forecast model..."):
        forecast_df, method = forecast_revenue(df, periods=forecast_periods)

    st.caption(f"Model: **{method.upper()}**")

    historical_daily = df.groupby("date")["revenue"].sum().reset_index()
    historical_daily["date"] = pd.to_datetime(historical_daily["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_daily["date"], y=historical_daily["revenue"],
                             mode="lines", name="Historical",
                             line=dict(color="#667eea", width=1.5)))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["predicted"],
                             mode="lines", name="Forecast",
                             line=dict(color="#f093fb", width=2.5, dash="dot")))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
        y=pd.concat([forecast_df["upper"], forecast_df["lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(240,147,251,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="80% confidence"
    ))
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    total_forecast = forecast_df["predicted"].sum()
    st.metric("Projected revenue", f"₹{total_forecast:,.0f}",
              f"±₹{(forecast_df['upper']-forecast_df['lower']).mean()/2:,.0f} avg daily range")

    st.subheader("Trend narrative")
    st.info(trend_summary(df))

    st.subheader("Category-level 30-day projection")
    cat_forecasts = []
    for cat_name in df["category"].unique():
        cat_df = df[df["category"] == cat_name]
        if len(cat_df) >= 30:
            f, _ = forecast_revenue(cat_df, periods=30)
            cat_forecasts.append({
                "Category": cat_name,
                "30-day forecast": round(f["predicted"].sum(), 0),
                "Daily avg":       round(f["predicted"].mean(), 0),
            })
    if cat_forecasts:
        st.dataframe(
            pd.DataFrame(cat_forecasts).sort_values("30-day forecast", ascending=False),
            use_container_width=True, hide_index=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — AI INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    kpis    = get_kpis(df)
    monthly = monthly_revenue(df)
    cat_df  = category_analysis(df)

    st.subheader("AI-generated business insights")
    if st.button("Generate insights", type="primary"):
        with st.spinner("Analysing your data..."):
            insights = generate_insights(kpis, monthly, cat_df,
                                         api_key=api_key if api_key else None)
        st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Ask your data a question")
    st.caption("Examples: *What is driving the highest revenue?*  |  *Which category has best margin?*")
    user_q = st.text_input("Your question")
    if user_q:
        unique_cats = list(df["category"].unique())
        unique_cities = list(df["city"].unique()[:5])
        df_summary = (
            f"Categories: {unique_cats}. Cities: {unique_cities}. "
            f"Revenue range: ₹{df['revenue'].min():.0f}–₹{df['revenue'].max():.0f}."
        )
        with st.spinner("Thinking..."):
            answer = quick_qa(user_q, kpis, df_summary, api_key=api_key if api_key else None)
        st.success(answer)

    st.divider()
    st.subheader("KPI summary")
    col1, col2, col3 = st.columns(3)
    metrics = [
        ("Total Revenue",   f"₹{kpis['total_revenue']/1e6:.2f}M"),
        ("Total Profit",    f"₹{kpis['total_profit']/1e6:.2f}M"),
        ("Total Orders",    f"{kpis['total_orders']:,}"),
        ("Avg Order Value", f"₹{kpis['avg_order_value']:,.0f}"),
        ("Profit Margin",   f"{kpis['profit_margin']}%"),
        ("Return Rate",     f"{kpis['return_rate']}%"),
        ("Avg Rating",      str(kpis['avg_rating'])),
        ("Top Category",    kpis["top_category"]),
        ("Data Quality",    f"{dq}/100"),
    ]
    for i, (label, val) in enumerate(metrics):
        with [col1, col2, col3][i % 3]:
            st.metric(label, val)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Anomaly detection — Isolation Forest")
    st.caption("Automatically flags transactions with unusual revenue, quantity, or discount patterns.")

    with st.spinner("Running anomaly model..."):
        df_anomaly = anomaly_detection(df)

    n_anom = df_anomaly["is_anomaly"].sum()
    st.metric("Anomalies detected", f"{n_anom} rows",
              f"{n_anom/len(df_anomaly)*100:.1f}% of dataset")

    hover_cols = [c for c in ["category","city","channel"] if df_anomaly[c].nunique() > 1]
    fig = px.scatter(df_anomaly, x="date", y="revenue",
                     color="is_anomaly",
                     color_discrete_map={True:"#dc3545", False:"#667eea"},
                     opacity=0.6, hover_data=hover_cols or None,
                     labels={"is_anomaly":"Anomaly"})
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    # Show anomaly table with whatever columns are available
    display_cols = [c for c in ["order_id","date","category","city","revenue","quantity","discount_pct","channel"]
                    if c in df_anomaly.columns]
    st.subheader("Anomalous rows")
    st.dataframe(
        df_anomaly[df_anomaly["is_anomaly"]][display_cols]
          .sort_values("revenue", ascending=False)
          .head(50),
        use_container_width=True, hide_index=True
    )

    if has_real_values(df["category"]):
        st.subheader("Anomaly count by category")
        anom_cat = df_anomaly.groupby("category")["is_anomaly"].sum().reset_index()
        anom_cat.columns = ["Category","Anomalies"]
        fig = px.bar(anom_cat.sort_values("Anomalies", ascending=False),
                     x="Category", y="Anomalies", color="Anomalies",
                     color_continuous_scale="Reds", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("RetailIQ — Automated Data Insights | Streamlit + Prophet + Scikit-learn + Claude AI")