import pandas as pd
import numpy as np

def forecast_revenue(df, periods=90):
    """
    Forecast daily revenue for next `periods` days using Prophet.
    Falls back to a simple linear trend if Prophet not available.
    """
    daily = (
        df.groupby("date")["revenue"]
          .sum()
          .reset_index()
          .rename(columns={"date": "ds", "revenue": "y"})
    )
    daily["ds"] = pd.to_datetime(daily["ds"])

    try:
        from prophet import Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.80,
            changepoint_prior_scale=0.05
        )
        model.fit(daily)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        result = forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(periods)
        result.columns = ["date","predicted","lower","upper"]
        return result, "prophet"
    except Exception:
        x = np.arange(len(daily))
        y = daily["y"].values
        coeffs = np.polyfit(x, y, 1)
        future_x = np.arange(len(daily), len(daily) + periods)
        predicted = np.polyval(coeffs, future_x)
        std = y.std()
        last_date = daily["ds"].max()
        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods)
        result = pd.DataFrame({
            "date": dates,
            "predicted": predicted.clip(0),
            "lower": (predicted - std).clip(0),
            "upper": predicted + std
        })
        return result, "linear"


def growth_metrics(df):
    """Month-over-month and year-over-year growth."""
    monthly = df.groupby("month")["revenue"].sum().reset_index().sort_values("month")
    monthly["mom_growth"] = monthly["revenue"].pct_change() * 100
    monthly["rolling_3m"] = monthly["revenue"].rolling(3).mean()

    years = df.groupby("year")["revenue"].sum()
    yoy = {}
    for yr in sorted(years.index)[1:]:
        prev = years.get(yr - 1, None)
        if prev and prev > 0:
            yoy[yr] = round((years[yr] - prev) / prev * 100, 2)
    return monthly, yoy


def detect_seasonality(df):
    """Return average revenue by month number."""
    df = df.copy()
    df["month_num"] = df["date"].dt.month
    seasonal = df.groupby("month_num")["revenue"].mean().reset_index()
    seasonal["month_label"] = pd.to_datetime(seasonal["month_num"], format="%m").dt.strftime("%b")
    return seasonal


def trend_summary(df):
    """Plain English trend summary."""
    monthly = df.groupby("month")["revenue"].sum().sort_index()
    if len(monthly) < 3:
        return "Not enough data for trend analysis."

    recent   = monthly.iloc[-3:].mean()
    previous = monthly.iloc[-6:-3].mean() if len(monthly) >= 6 else monthly.iloc[:-3].mean()

    pct = (recent - previous) / previous * 100 if previous else 0
    direction = "up" if pct > 0 else "down"
    magnitude = "significantly" if abs(pct) > 15 else "slightly"

    peak_month   = monthly.idxmax()
    lowest_month = monthly.idxmin()

    summary = (
        f"Revenue is trending {direction} {magnitude} by {abs(pct):.1f}% "
        f"compared to the previous 3-month period. "
        f"Peak revenue was in {peak_month} and the lowest was in {lowest_month}."
    )
    return summary