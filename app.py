import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .hero {
        background: linear-gradient(120deg, #e8f7ff 0%, #fff7ec 100%);
        border: 1px solid #d8e6f5; border-radius: 16px;
        padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .hero h2 { color: #0f172a !important; }
    .hero p  { color: #1f2937 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h2 style="margin:0;">📈 Stock Price Prediction — LSTM Model</h2>
        <p style="margin:0.35rem 0 0 0;">
            EC6301 Artificial Intelligence Mini Project &nbsp;|&nbsp;
            University of Ruhuna &nbsp;|&nbsp; LSTM-based deep learning for stock price forecasting
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

LOOKBACK       = 60
DATA_PATH      = "all_stocks_5yr.csv"
GENERAL_MODEL  = "models/lstm_general.keras"
ORIGINAL_MODEL = "lstm_model.keras"


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_csv_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["Name", "date"]).drop_duplicates().reset_index(drop=True)
    return df


@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
    except ImportError:
        return None, None
    for path, label in [
        (GENERAL_MODEL,  "General Model (50 Companies)"),
        (ORIGINAL_MODEL, "AAPL Model"),
    ]:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path), label
            except Exception:
                pass
    return None, None


@st.cache_data(ttl=3600)
def fetch_live_data(ticker: str):
    try:
        import yfinance as yf
        raw = yf.download(ticker, start="2013-01-01", progress=False)
        if raw is None or raw.empty:
            return None
        raw = raw.reset_index()
        # Flatten multi-level columns (yfinance 1.3+)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        # Normalise column names to lowercase
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        # Handle "adj close" if present
        if "adj close" in raw.columns:
            raw = raw.rename(columns={"adj close": "close"})
        raw["date"] = pd.to_datetime(raw["date"])
        raw["Name"] = ticker
        cols = ["date", "open", "high", "low", "close", "volume", "Name"]
        cols = [c for c in cols if c in raw.columns]
        return raw[cols].sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def predict_future_days(model, scaled, scaler, days: int):
    seq = list(scaled[-LOOKBACK:, 0])
    prices = []
    for _ in range(days):
        x = np.array(seq[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        p = float(model.predict(x, verbose=0)[0][0])
        seq.append(p)
        prices.append(float(scaler.inverse_transform([[p]])[0][0]))
    return prices


# ── Load resources ────────────────────────────────────────────────────────────
csv_df            = load_csv_data(DATA_PATH)
model, model_label = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    companies    = sorted(csv_df["Name"].unique().tolist())
    default_idx  = companies.index("AAPL") if "AAPL" in companies else 0
    selected_company = st.selectbox("Company", companies, index=default_idx)

    st.divider()
    use_live = st.checkbox(
        "Use Live Data (yfinance)",
        value=False,
        help="Fetch fresh data from Yahoo Finance — 2013 to today",
    )

    forecast_days = 1
    if use_live:
        forecast_days = st.slider("Future Forecast Days", min_value=1, max_value=30, value=7)

    company_df = csv_df[csv_df["Name"] == selected_company].copy()
    min_date   = company_df["date"].min().date()
    max_date   = company_df["date"].max().date()

    if not use_live:
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date, end_date = min_date, max_date

    st.divider()
    if model is not None:
        st.success(f"Loaded: {model_label}")
    else:
        st.warning("No model found.\nRun the notebook or:\npython run_train_all.py")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Explorer", "🤖 LSTM Prediction", "📋 Raw Data"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Explorer
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    if use_live:
        with st.spinner(f"Fetching live data for {selected_company}..."):
            live_df = fetch_live_data(selected_company)
        if live_df is None:
            st.error(f"Could not fetch live data for {selected_company}. Check internet connection.")
            st.stop()
        filtered = live_df.copy()
        st.info(f"Live data: {filtered['date'].min().date()} — {filtered['date'].max().date()}")
    else:
        filtered = company_df[
            (company_df["date"].dt.date >= start_date) &
            (company_df["date"].dt.date <= end_date)
        ].copy()

    if filtered.empty:
        st.warning("No data found for the selected range.")
        st.stop()

    filtered["MA_20"]        = filtered["close"].rolling(20).mean()
    filtered["MA_50"]        = filtered["close"].rolling(50).mean()
    filtered["daily_return"] = filtered["close"].pct_change()

    latest_close  = filtered["close"].iloc[-1]
    first_close   = filtered["close"].iloc[0]
    period_change = ((latest_close - first_close) / first_close) * 100
    avg_volume    = filtered["volume"].mean()
    volatility    = filtered["daily_return"].std()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close",         f"${latest_close:,.2f}")
    c2.metric("Period Change",        f"{period_change:+.2f}%")
    c3.metric("Avg Daily Volume",     f"{avg_volume:,.0f}")
    c4.metric("Volatility (Std Ret)", f"{volatility:.4f}")

    chart_df = filtered[["date", "close", "MA_20", "MA_50"]].melt(
        id_vars="date", var_name="Series", value_name="Price"
    )
    fig_ma = px.line(
        chart_df, x="date", y="Price", color="Series",
        title=f"{selected_company}: Closing Price with 20-Day and 50-Day Moving Averages",
        color_discrete_map={"close": "#0f172a", "MA_20": "#0ea5e9", "MA_50": "#f97316"},
    )
    fig_ma.update_layout(legend_title_text="", hovermode="x unified")
    st.plotly_chart(fig_ma, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        vol_fig = px.bar(
            filtered, x="date", y="volume",
            title=f"{selected_company}: Trading Volume",
            color_discrete_sequence=["#6366f1"],
        )
        st.plotly_chart(vol_fig, use_container_width=True)
    with col_r:
        ret_fig = px.histogram(
            filtered.dropna(subset=["daily_return"]),
            x="daily_return", nbins=60,
            title=f"{selected_company}: Daily Return Distribution",
            color_discrete_sequence=["#14b8a6"],
        )
        ret_fig.update_layout(bargap=0.05)
        st.plotly_chart(ret_fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — LSTM Prediction
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    if model is None:
        st.error(
            "**LSTM model not found.**\n\n"
            "Run the notebook OR train the general model:\n"
            "```\npython run_train_all.py\n```"
        )
    else:
        st.markdown(f"### LSTM Prediction — {selected_company}")

        if use_live:
            with st.spinner(f"Fetching live data for {selected_company}..."):
                pred_src = fetch_live_data(selected_company)
            if pred_src is None:
                st.error(f"Could not fetch live data for {selected_company}.")
                st.stop()
            data_label = f"Live data: 2013 — {pred_src['date'].max().date()}"
        else:
            pred_src   = csv_df[csv_df["Name"] == selected_company].copy()
            data_label = "Historical CSV: 2013 — 2018"

        st.caption(f"{model_label}  |  Lookback: {LOOKBACK} days  |  {data_label}")

        stock_data = pred_src[["date", "close"]].sort_values("date").reset_index(drop=True)

        if len(stock_data) < LOOKBACK + 20:
            st.warning(
                f"Not enough data for {selected_company}. "
                f"Need at least {LOOKBACK + 20} trading days."
            )
        else:
            close_vals = stock_data["close"].values.reshape(-1, 1)
            scaler     = MinMaxScaler(feature_range=(0, 1))
            scaled     = scaler.fit_transform(close_vals)

            X, y = [], []
            for i in range(LOOKBACK, len(scaled)):
                X.append(scaled[i - LOOKBACK:i, 0])
                y.append(scaled[i, 0])
            X = np.array(X).reshape(-1, LOOKBACK, 1)
            y = np.array(y)

            split  = int(len(X) * 0.80)
            X_test = X[split:]
            y_test = y[split:]

            with st.spinner("Running LSTM predictions..."):
                y_pred_scaled = model.predict(X_test, verbose=0)

            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

            mse  = mean_squared_error(y_true, y_pred)
            rmse = math.sqrt(mse)
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MSE",              f"{mse:.4f}")
            m2.metric("RMSE",             f"${rmse:.2f}")
            m3.metric("MAPE",             f"{mape:.2f}%")
            m4.metric("Avg Actual Price", f"${float(y_true.mean()):.2f}")

            test_dates = stock_data["date"].values[LOOKBACK + split:]

            # Predicted vs Actual
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=test_dates, y=y_true.flatten(),
                name="Actual Price", line=dict(color="#0f172a", width=2),
            ))
            fig_pred.add_trace(go.Scatter(
                x=test_dates, y=y_pred.flatten(),
                name="LSTM Predicted", line=dict(color="#ef4444", width=2, dash="dash"),
            ))
            fig_pred.update_layout(
                title=f"{selected_company} — LSTM: Predicted vs Actual Closing Price (Test Set)",
                xaxis_title="Date", yaxis_title="Price (USD)",
                hovermode="x unified", legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Full timeline
            all_dates    = stock_data["date"].values
            train_actual = scaler.inverse_transform(scaled[:split + LOOKBACK])

            fig_full = go.Figure()
            fig_full.add_trace(go.Scatter(
                x=all_dates[:split + LOOKBACK], y=train_actual.flatten(),
                name="Training Data (Actual)", line=dict(color="#94a3b8", width=1), opacity=0.7,
            ))
            fig_full.add_trace(go.Scatter(
                x=test_dates, y=y_true.flatten(),
                name="Test Actual", line=dict(color="#0f172a", width=1.5),
            ))
            fig_full.add_trace(go.Scatter(
                x=test_dates, y=y_pred.flatten(),
                name="LSTM Predicted", line=dict(color="#ef4444", width=1.5, dash="dash"),
            ))
            fig_full.add_shape(
                type="line",
                x0=pd.Timestamp(test_dates[0]), x1=pd.Timestamp(test_dates[0]),
                y0=0, y1=1, xref="x", yref="paper",
                line=dict(dash="dot", color="#64748b"),
            )
            fig_full.add_annotation(
                x=pd.Timestamp(test_dates[0]), y=1.02, xref="x", yref="paper",
                text="Train / Test Split", showarrow=False, font=dict(color="#64748b"),
            )
            fig_full.update_layout(
                title=f"{selected_company} — Full Timeline: Training Data + LSTM Predictions",
                xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified",
            )
            st.plotly_chart(fig_full, use_container_width=True)

            # ── Forecast section ──────────────────────────────────────────
            st.divider()

            if use_live and forecast_days > 1:
                st.markdown(f"#### Future {forecast_days}-Day Price Forecast")
                with st.spinner(f"Forecasting next {forecast_days} days..."):
                    future_prices = predict_future_days(model, scaled, scaler, forecast_days)

                last_date    = pd.Timestamp(stock_data["date"].iloc[-1])
                future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
                last_price   = float(close_vals[-1][0])

                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=stock_data["date"].values[-60:],
                    y=close_vals[-60:].flatten(),
                    name="Last 60 Days (Actual)",
                    line=dict(color="#0f172a", width=2),
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_dates, y=future_prices,
                    name=f"{forecast_days}-Day Forecast",
                    line=dict(color="#22c55e", width=2.5, dash="dash"),
                    mode="lines+markers",
                ))
                fig_future.update_layout(
                    title=f"{selected_company} — {forecast_days}-Day Future Price Forecast",
                    xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified",
                )
                st.plotly_chart(fig_future, use_container_width=True)

                st.markdown("##### Forecast Table")
                forecast_df = pd.DataFrame({
                    "Day":              [f"Day {i + 1}" for i in range(forecast_days)],
                    "Date":             [d.strftime("%Y-%m-%d") for d in future_dates],
                    "Predicted Price":  [f"${p:.2f}" for p in future_prices],
                    "Change from Last": [f"{((p - last_price) / last_price) * 100:+.2f}%" for p in future_prices],
                    "Direction":        ["▲ UP" if p > last_price else "▼ DOWN" for p in future_prices],
                })
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            else:
                st.markdown("#### Next Trading Day Forecast")
                last_seq    = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
                next_scaled = model.predict(last_seq, verbose=0)
                next_price  = float(scaler.inverse_transform(next_scaled)[0][0])
                last_price  = float(close_vals[-1][0])
                change      = next_price - last_price
                pct         = (change / last_price) * 100

                fa, fb, fc = st.columns(3)
                fa.metric("Last Known Close",     f"${last_price:.2f}")
                fb.metric(
                    "Predicted Next Close",
                    f"${next_price:.2f}",
                    f"{change:+.2f} ({pct:+.2f}%)",
                )
                fc.metric("Direction", "▲ UP" if change > 0 else "▼ DOWN")

                if use_live:
                    st.info("Enable the forecast slider in the sidebar to predict 7 or 30 days ahead.")

            st.caption("For academic use only — not financial advice.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Raw Data
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"### Raw Data — {selected_company}")

    if use_live:
        with st.spinner(f"Fetching live data for {selected_company}..."):
            raw_live = fetch_live_data(selected_company)
        if raw_live is not None:
            display_df = raw_live.sort_values("date", ascending=False)
            st.dataframe(display_df, use_container_width=True)
            st.caption(
                f"{len(display_df):,} rows  |  "
                f"Source: Yahoo Finance (yfinance)  |  "
                f"2013 — {raw_live['date'].max().date()}"
            )
        else:
            st.error("Could not fetch live data. Showing CSV data instead.")
            display_df = company_df.sort_values("date", ascending=False)
            st.dataframe(display_df, use_container_width=True)
    else:
        display_df = company_df[
            (company_df["date"].dt.date >= start_date) &
            (company_df["date"].dt.date <= end_date)
        ].copy().sort_values("date", ascending=False)
        st.dataframe(display_df, use_container_width=True)
        st.caption(
            f"{len(display_df):,} rows shown  |  "
            "Source: S&P 500 5-Year Dataset (Kaggle)"
        )
