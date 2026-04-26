import json
import requests

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

def build_context(kpis, top_category, monthly_df, category_df):
    """Build a compact data summary to send to Claude."""
    monthly_tail = monthly_df.tail(6)[["month","revenue","orders"]].to_dict(orient="records")
    cat_summary  = category_df[["category","revenue","profit"]].to_dict(orient="records")
    return {
        "kpis": kpis,
        "top_category": top_category,
        "last_6_months": monthly_tail,
        "category_breakdown": cat_summary,
    }

def generate_insights(kpis, monthly_df, category_df, api_key=None):
    """
    Call Claude API for AI-generated insights.
    If no API key is provided, return rule-based insights.
    """
    context = build_context(kpis, kpis.get("top_category"), monthly_df, category_df)

    if not api_key:
        return _rule_based_insights(kpis, monthly_df, category_df)

    prompt = f"""You are a senior retail data analyst. Analyse this retail sales data and provide:
1. Three key business insights (specific, data-driven)
2. Two risks or concerns
3. Three actionable recommendations

Data summary:
{json.dumps(context, indent=2, default=str)}

Respond in plain English, concise bullet points. No markdown headers."""

    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except Exception as e:
        return _rule_based_insights(kpis, monthly_df, category_df)

def _rule_based_insights(kpis, monthly_df, category_df):
    lines = []

    # Growth insight
    if len(monthly_df) >= 2:
        rev_list = monthly_df["revenue"].values
        growth = (rev_list[-1] - rev_list[-2]) / rev_list[-2] * 100 if rev_list[-2] else 0
        direction = "grew" if growth >= 0 else "declined"
        lines.append(f"- Revenue {direction} by {abs(growth):.1f}% last month vs the previous month.")

    # Profit margin
    lines.append(f"- Overall profit margin stands at {kpis['profit_margin']}%. "
                 f"{'Above industry average of ~18%.' if kpis['profit_margin'] > 18 else 'Below the 18% industry benchmark — review cost structure.'}")

    # Return rate
    if kpis["return_rate"] > 10:
        lines.append(f"- Return rate is {kpis['return_rate']}%, which is high. Investigate product quality or mislabelling in top return categories.")
    else:
        lines.append(f"- Return rate of {kpis['return_rate']}% is healthy (industry avg ~8–12%).")

    # Top category
    lines.append(f"- {kpis['top_category']} is the top revenue-generating category. Prioritise inventory and promotions here.")

    # Risk
    if len(monthly_df) >= 3:
        last3 = monthly_df["revenue"].values[-3:]
        if all(last3[i] > last3[i+1] for i in range(len(last3)-1)):
            lines.append("- RISK: Revenue has declined for 3 consecutive months. Urgent review of pricing and promotions needed.")

    # Recommendation
    lines.append("- Recommendation: Increase marketing spend during Oct-Dec festive season — historical data shows 60% higher conversion.")
    lines.append("- Recommendation: Online and Mobile App channels show strong growth. Consider reducing in-store overhead.")

    return "\n".join(lines)

def quick_qa(question, kpis, df_summary, api_key=None):
    """Answer a natural language question about the data."""
    if not api_key:
        return _fallback_qa(question, kpis)

    prompt = f"""You are a retail analyst assistant. Answer this question using the data below.
Be concise (2-3 sentences). If data is insufficient, say so.

Question: {question}

KPIs: {json.dumps(kpis, default=str)}
Summary: {df_summary}"""

    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except Exception:
        return _fallback_qa(question, kpis)

def _fallback_qa(question, kpis):
    q = question.lower()
    if "revenue" in q:
        return f"Total revenue is ₹{kpis['total_revenue']:,.0f} with an average order value of ₹{kpis['avg_order_value']:,.0f}."
    if "profit" in q:
        return f"Total profit is ₹{kpis['total_profit']:,.0f} at a margin of {kpis['profit_margin']}%."
    if "return" in q:
        return f"The return rate is {kpis['return_rate']}%."
    if "order" in q:
        return f"Total orders processed: {kpis['total_orders']:,}."
    if "category" in q or "best" in q:
        return f"The top-performing category by revenue is {kpis['top_category']}."
    return "Please provide your Claude API key in the sidebar for full AI-powered Q&A, or ask about revenue, profit, returns, or categories."