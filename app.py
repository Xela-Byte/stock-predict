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
page = st.sidebar.radio("Select a View:", ["Model Evaluation (RMSE)", "Interactive Stock Trends"])

# --- 4. Page 1: The RMSE Scores ---
if page == "Model Evaluation (RMSE)":
    st.header("📊 System Accuracy & Evaluation")
    st.write("This table displays the Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) for all 15 equities.")
    st.write("*Note: A lower RMSE score indicates a more accurate prediction under tax policy shocks.*")
    
    # Display the table
    st.dataframe(rmse_df, use_container_width=True)
    
    # Draw a bar chart of the RMSE scores
    st.write("### Visual RMSE Comparison")
    st.bar_chart(data=rmse_df, x='Stock Ticker', y='RMSE')

# --- 5. Page 2: Interactive Charts ---
elif page == "Interactive Stock Trends":
    st.header("📉 Interactive Historical Data")
    st.write("Select a stock from the dropdown below to view its price volatility over the study period.")
    
    # Dropdown menu populated by your stock tickers
    selected_stock = st.selectbox("Select an Equity:", rmse_df['Stock Ticker'].tolist())
    
    # Fetch the specific dataset for the selected stock
    df = stock_data[selected_stock]
    
    # Plot the interactive chart
    st.line_chart(df['Closing_Price'])
    st.caption(f"Historical Closing Prices for {selected_stock} under recent macroeconomic conditions.")