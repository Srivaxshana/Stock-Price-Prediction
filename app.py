import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")

# Light styling for a cleaner look
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .hero {
        background: linear-gradient(120deg, #e8f7ff 0%, #fff7ec 100%);
        border: 1px solid #d8e6f5;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .hero h2 {
        color: #0f172a !important;
    }

    .hero p {
        color: #1f2937 !important;
    }

    .small-note {
        color: #4a5568;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h2 style="margin: 0;">Stock Price Explorer</h2>
        <p style="margin: 0.35rem 0 0 0;">Simple UI to explore trends, moving averages, returns, and volatility.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["Name", "date"]).drop_duplicates().reset_index(drop=True)
    return df


df = load_data("all_stocks_5yr.csv")

with st.sidebar:
    st.header("Filters")
    companies = sorted(df["Name"].unique().tolist())
    selected_company = st.selectbox("Company", companies, index=0)

    company_df = df[df["Name"] == selected_company].copy()
    min_date = company_df["date"].min().date()
    max_date = company_df["date"].max().date()

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

filtered = company_df[(company_df["date"].dt.date >= start_date) & (company_df["date"].dt.date <= end_date)].copy()

if filtered.empty:
    st.warning("No data found in selected range. Try a wider date range.")
    st.stop()

filtered["MA_20"] = filtered["close"].rolling(20).mean()
filtered["MA_50"] = filtered["close"].rolling(50).mean()
filtered["daily_return"] = filtered["close"].pct_change()

latest_close = filtered["close"].iloc[-1]
first_close = filtered["close"].iloc[0]
period_change = ((latest_close - first_close) / first_close) * 100
avg_volume = filtered["volume"].mean()
volatility = filtered["daily_return"].std()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close", f"${latest_close:,.2f}")
c2.metric("Period Change", f"{period_change:+.2f}%")
c3.metric("Avg Volume", f"{avg_volume:,.0f}")
c4.metric("Volatility (Std Return)", f"{volatility:.4f}")

chart_df = filtered[["date", "close", "MA_20", "MA_50"]].melt(
    id_vars="date", var_name="Series", value_name="Price"
)

fig = px.line(
    chart_df,
    x="date",
    y="Price",
    color="Series",
    title=f"{selected_company}: Close Price and Moving Averages",
    color_discrete_map={
        "close": "#0f172a",
        "MA_20": "#0ea5e9",
        "MA_50": "#f97316",
    },
)
fig.update_layout(legend_title_text="", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

returns_fig = px.histogram(
    filtered.dropna(subset=["daily_return"]),
    x="daily_return",
    nbins=60,
    title=f"{selected_company}: Daily Return Distribution",
    color_discrete_sequence=["#14b8a6"],
)
returns_fig.update_layout(bargap=0.05)
st.plotly_chart(returns_fig, use_container_width=True)

st.markdown("<p class='small-note'>Note: This UI is for analysis/visualization. Add your trained LSTM model file later to show forecast outputs here.</p>", unsafe_allow_html=True)

st.subheader("Recent Data")
st.dataframe(filtered.tail(20), use_container_width=True)
