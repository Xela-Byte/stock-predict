import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- 1. App Configuration ---
st.set_page_config(page_title="NGX Predictive System", layout="wide")
st.title("📈 NGX Stock Market Predictor")
st.subheader("Hybrid SARIMAX-XGBoost Architecture")

# --- 2. Load the Exported Data ---
@st.cache_data
def load_data():
    rmse_df = pd.read_csv('rmse_results.csv')
    stock_dict = joblib.load('stock_data_dict.pkl')
    return rmse_df, stock_dict

rmse_df, stock_data = load_data()

# --- 3. Sidebar Navigation ---
st.sidebar.header("Navigation Panel")
page = st.sidebar.radio(
    "Select a View:",
    ["Model Evaluation", "Stock Price Charts"]
)

# --- 4. Page 1: Model Evaluation ---
if page == "Model Evaluation":
    st.header("📊 System Accuracy & Evaluation")
    st.write(
        "Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and "
        "Mean Absolute Percentage Error (MAPE) for all 15 NGX equities. "
        "Lower values indicate better model accuracy."
    )

    # Full metrics table
    st.dataframe(rmse_df, use_container_width=True)

    st.divider()

    # --- RMSE Bar Chart ---
    st.subheader("📉 RMSE by Equity")
    st.bar_chart(data=rmse_df.set_index("Stock Ticker")[["RMSE"]])

    st.divider()

    # --- MAE Bar Chart ---
    st.subheader("📉 MAE by Equity")
    st.bar_chart(data=rmse_df.set_index("Stock Ticker")[["MAE"]])

    st.divider()

    # --- MAPE Bar Chart ---
    st.subheader("📉 MAPE (%) by Equity")
    st.bar_chart(data=rmse_df.set_index("Stock Ticker")[["MAPE (%)"]])

    st.divider()

    # --- Side-by-side RMSE vs MAE comparison ---
    st.subheader("🔁 RMSE vs MAE Comparison")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(rmse_df))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], rmse_df["RMSE"], width, label="RMSE", color="#4F8BF9")
    bars2 = ax.bar([i + width/2 for i in x], rmse_df["MAE"], width, label="MAE", color="#F97B4F")
    ax.set_xticks(list(x))
    ax.set_xticklabels(rmse_df["Stock Ticker"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Error Value")
    ax.set_title("RMSE vs MAE per Equity")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# --- 5. Page 2: Stock Price Charts ---
elif page == "Stock Price Charts":
    st.header("📉 Historical Closing Prices — All 15 Equities")
    st.write(
        "Historical closing price trends for each NGX equity over the study period. "
        "Select the chart type from the tabs below."
    )

    tab1, tab2 = st.tabs(["📊 Individual Charts", "🔍 Compare Two Stocks"])

    with tab1:
        cols = st.columns(2)
        for idx, ticker in enumerate(stock_data.keys()):
            df = stock_data[ticker]
            with cols[idx % 2]:
                st.subheader(ticker)
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(df.index, df["Closing_Price"], color="#4F8BF9", linewidth=1.2)
                ax.set_title(f"{ticker} — Closing Price", fontsize=10)
                ax.set_xlabel("Date", fontsize=8)
                ax.set_ylabel("Price (NGN)", fontsize=8)
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.tick_params(axis="y", labelsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with tab2:
        st.write("Pick two equities to overlay their closing prices.")
        col1, col2 = st.columns(2)
        with col1:
            stock_a = st.selectbox("First Equity:", list(stock_data.keys()), key="a")
        with col2:
            stock_b = st.selectbox("Second Equity:", list(stock_data.keys()), index=1, key="b")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(stock_data[stock_a].index, stock_data[stock_a]["Closing_Price"],
                label=stock_a, color="#4F8BF9", linewidth=1.5)
        ax.plot(stock_data[stock_b].index, stock_data[stock_b]["Closing_Price"],
                label=stock_b, color="#F97B4F", linewidth=1.5)
        ax.set_title(f"{stock_a} vs {stock_b} — Closing Prices", fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (NGN)")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
