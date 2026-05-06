import math
import os

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

LOOKBACK   = 60
MODEL_PATH = "lstm_model.keras"
DATA_PATH  = "all_stocks_5yr.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["Name", "date"]).drop_duplicates().reset_index(drop=True)
    return df


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


def _has_yfinance() -> bool:
    try:
        import yfinance  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_live_data(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        symbol,
        period="max",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    # yfinance can return MultiIndex columns even for a single ticker depending on version/config.
    if isinstance(raw.columns, pd.MultiIndex):
        level_vals = raw.columns.get_level_values(-1)
        if symbol in set(level_vals):
            raw = raw.xs(symbol, axis=1, level=-1, drop_level=True)
        else:
            raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:, ~raw.columns.duplicated()]

    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else ("Datetime" if "Datetime" in raw.columns else None)
    if not date_col:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    def _series_1d(col_name: str):
        val = raw.get(col_name, np.nan)
        if isinstance(val, pd.DataFrame):
            if val.shape[1] == 0:
                return np.nan
            return val.iloc[:, 0]
        return val

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]),
            "open": _series_1d("Open"),
            "high": _series_1d("High"),
            "low": _series_1d("Low"),
            "close": _series_1d("Close"),
            "volume": _series_1d("Volume"),
        }
    )
    out = (
        out.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )
    return out


df    = load_data(DATA_PATH)
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    companies    = sorted(df["Name"].unique().tolist())
    default_idx  = companies.index("AAPL") if "AAPL" in companies else 0
    selected_company = st.selectbox("Company", companies, index=default_idx)

    has_yf = _has_yfinance()
    use_live = st.toggle(
        "Use live prices (Yahoo Finance)",
        value=has_yf,
        disabled=not has_yf,
        help="Downloads latest market data on demand. Requires `yfinance`.",
    )
    if not has_yf:
        st.caption("Install `yfinance` to enable live prices.")

    ticker = selected_company
    if use_live:
        ticker = st.text_input("Ticker symbol", value=selected_company).strip().upper() or selected_company
        with st.spinner(f"Downloading latest data for {ticker}..."):
            live_df = load_live_data(ticker)
        if live_df.empty:
            st.error(f"No live data found for ticker `{ticker}`. Falling back to CSV dataset.")
            use_live = False

    if use_live:
        company_df = live_df.copy()
        min_date = company_df["date"].min().date()
        max_date = company_df["date"].max().date()
    else:
        company_df = df[df["Name"] == selected_company].copy()
        min_date   = company_df["date"].min().date()
        max_date   = company_df["date"].max().date()

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

    st.divider()
    if model is not None:
        st.success("LSTM model loaded")
    else:
        st.warning(
            "Model not found.  \n"
            "Run the notebook first to train and save `lstm_model.keras`."
        )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Explorer", "🤖 LSTM Prediction", "📋 Raw Data"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Explorer
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
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
            "Please run all cells in `stock-price-direction-forecasting-with-lstm.ipynb` "
            "to train the model. The notebook will save `lstm_model.keras` in this folder. "
            "Then refresh this page."
        )
    else:
        st.markdown(f"### LSTM Prediction — {selected_company}")
        st.caption(
            f"Model trained on AAPL. Lookback window: {LOOKBACK} days. "
            "Predictions use MinMax scaling and a 2-layer LSTM architecture."
        )

        stock_data = company_df[["date", "close"]].copy()
        stock_data = stock_data.sort_values("date").reset_index(drop=True)

        if len(stock_data) < LOOKBACK + 20:
            st.warning(
                f"Not enough trading days for {selected_company}. "
                f"Need at least {LOOKBACK + 20} days."
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
                name="Actual Price",
                line=dict(color="#0f172a", width=2),
            ))
            fig_pred.add_trace(go.Scatter(
                x=test_dates, y=y_pred.flatten(),
                name="LSTM Predicted",
                line=dict(color="#ef4444", width=2, dash="dash"),
            ))
            fig_pred.update_layout(
                title=f"{selected_company} — LSTM: Predicted vs Actual Closing Price (Test Set)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Full timeline
            all_dates    = stock_data["date"].values
            train_actual = scaler.inverse_transform(scaled[: split + LOOKBACK])

            fig_full = go.Figure()
            fig_full.add_trace(go.Scatter(
                x=all_dates[: split + LOOKBACK],
                y=train_actual.flatten(),
                name="Training Data (Actual)",
                line=dict(color="#94a3b8", width=1),
                opacity=0.7,
            ))
            fig_full.add_trace(go.Scatter(
                x=test_dates, y=y_true.flatten(),
                name="Test Actual",
                line=dict(color="#0f172a", width=1.5),
            ))
            fig_full.add_trace(go.Scatter(
                x=test_dates, y=y_pred.flatten(),
                name="LSTM Predicted",
                line=dict(color="#ef4444", width=1.5, dash="dash"),
            ))
            fig_full.add_shape(
                type="line",
                x0=pd.Timestamp(test_dates[0]),
                x1=pd.Timestamp(test_dates[0]),
                y0=0, y1=1, xref="x", yref="paper",
                line=dict(dash="dot", color="#64748b"),
            )
            fig_full.add_annotation(
                x=pd.Timestamp(test_dates[0]),
                y=1.02, xref="x", yref="paper",
                text="Train / Test Split",
                showarrow=False, font=dict(color="#64748b"),
            )
            fig_full.update_layout(
                title=f"{selected_company} — Full Timeline: Training Data + LSTM Predictions",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_full, use_container_width=True)

            # Next-day forecast
            st.divider()
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

            st.caption(
                "This forecast uses the last 60 trading days as input to the LSTM. "
                "For academic use only — not financial advice."
            )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Raw Data
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"### Raw Data — {selected_company}")
    display_df = company_df[
        (company_df["date"].dt.date >= start_date) &
        (company_df["date"].dt.date <= end_date)
    ].copy().sort_values("date", ascending=False)

    st.dataframe(display_df, use_container_width=True)
    st.caption(
        f"{len(display_df)} rows shown | "
        "Source: S&P 500 5-Year Dataset (Kaggle)"
    )
