# RetailIQ — Automated Retail Sales Intelligence

An end-to-end automated data insights, dashboard, and forecasting platform for retail organisations.

## Features
- Auto data cleaning & quality scoring
- 7-tab interactive Streamlit dashboard
- Revenue, profit, category, city, channel analysis
- 90-day Prophet forecasting with confidence intervals
- Anomaly detection (Isolation Forest)
- AI-generated insights (Claude API or rule-based fallback)
- Natural language Q&A on your data

## Project structure
```
retail_insights/
├── app.py               ← Streamlit dashboard (run this)
├── data_processor.py    ← data cleaning & KPIs
├── forecasting.py       ← Prophet time-series forecasting
├── ai_insights.py       ← AI insights via Claude API
├── generate_data.py     ← dummy data generator
├── retail_sales_data.csv← generated dataset
└── requirements.txt
```

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dummy data (if not already generated)
```bash
python generate_data.py
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

### 4. Open your browser
Go to http://localhost:8501

## Optional: Enable AI insights
1. Get a free Claude API key from https://console.anthropic.com
2. Enter it in the sidebar of the dashboard
3. Click "Generate insights" in the AI Insights tab

## Dashboard tabs
| Tab | What it shows |
|-----|--------------|
| Overview | KPIs, channel mix, segment breakdown |
| Sales Trends | Monthly revenue, seasonality, YoY growth |
| Category & Brand | Category performance, brand treemap |
| Geographic | City revenue, channel preference by city |
| Forecasting | 90-day Prophet forecast with confidence interval |
| AI Insights | Claude-powered insights + Q&A |
| Anomaly Detection | Isolation Forest anomaly flagging |

## Using your own data
Upload any CSV with these columns (or a subset):
- `date` (YYYY-MM-DD)
- `revenue`, `profit`, `quantity`
- `category`, `brand`, `city`, `channel`
- `customer_segment`, `payment_method`
- `returned` (0/1), `customer_rating`

## Tech stack
- **Streamlit** — dashboard UI
- **Pandas + NumPy** — data processing
- **Plotly** — interactive charts
- **Prophet** — time-series forecasting
- **Scikit-learn** — anomaly detection
- **Claude API** — AI insights