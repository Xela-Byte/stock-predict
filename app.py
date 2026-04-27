import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from datetime import timedelta, date

# ── Optional ML deps ────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NGX Predictive System",
    page_icon="https://upload.wikimedia.org/wikipedia/en/8/8d/Nigerian_Exchange_Group_logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Metric card ── */
.kpi-card {
    background: #0f1c2e;
    border: 1px solid #1e3550;
    border-radius: 14px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    margin-bottom: 4px;
    transition: border-color .2s;
}
.kpi-card:hover { border-color: #3b6fa0; }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #3b82f6);
    border-radius: 14px 14px 0 0;
}
.kpi-icon {
    font-family: 'Material Icons Round';
    font-size: 22px;
    color: var(--accent, #3b82f6);
    display: block;
    margin-bottom: 10px;
    line-height: 1;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
    margin-bottom: 4px;
}
.kpi-sub {
    font-size: 12px;
    color: #64748b;
    font-weight: 500;
}
.kpi-sub.up   { color: #34d399; }
.kpi-sub.down { color: #f87171; }

/* ── Section heading ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 22px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e3550;
}
.sec-head .material-icons-round {
    font-size: 20px;
    color: #3b82f6;
    flex-shrink: 0;
}
.sec-head span.title {
    font-size: 15px;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: .3px;
}

/* ── Info / hint banner ── */
.info-banner {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #0c1e33;
    border: 1px solid #1e3550;
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #94a3b8;
    font-size: 13.5px;
    line-height: 1.6;
}
.info-banner .material-icons-round { color: #3b82f6; font-size: 20px; flex-shrink: 0; margin-top: 1px; }
.info-banner b { color: #cbd5e1; }

/* ── Rule card (format rules) ── */
.rule-card {
    background: #0f1c2e;
    border: 1px solid #1e3550;
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
}
.rule-card .rule-head {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; font-weight: 700; color: #e2e8f0;
    margin-bottom: 12px;
}
.rule-card .rule-head .material-icons-round { font-size: 18px; }
.rule-card .rule-head.ok   .material-icons-round { color: #34d399; }
.rule-card .rule-head.warn .material-icons-round { color: #fbbf24; }
.rule-card ul { margin: 0; padding-left: 18px; color: #94a3b8; font-size: 13px; line-height: 2; }
.rule-card li { margin-bottom: 2px; }
.rule-card li b { color: #cbd5e1; }

/* ── Template card ── */
.tmpl-card {
    background: #0f1c2e;
    border: 1px solid #1e3550;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    height: 100%;
}
.tmpl-card .material-icons-round { font-size: 36px; color: #3b82f6; display: block; margin-bottom: 10px; }
.tmpl-card .tmpl-title { font-size: 14px; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; }
.tmpl-card .tmpl-desc  { font-size: 12px; color: #64748b; line-height: 1.6; margin-bottom: 14px; }

/* ── Step badge ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #0c1e33;
    border: 1px solid #1e3550;
    border-radius: 999px;
    padding: 5px 14px 5px 6px;
    font-size: 12px;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: .3px;
    margin: 12px 0 8px;
}
.step-badge .num {
    background: #3b82f6;
    color: white;
    border-radius: 50%;
    width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px;
}

/* ── Tag pill ── */
.pill {
    display: inline-block;
    background: #1e3550;
    color: #93c5fd;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 11px;
    font-family: monospace;
    margin: 1px;
}

/* ── Streamlit overrides ── */
.stRadio > label { display: none; }
div[data-testid="stSidebarContent"] section { padding-top: 0 !important; }
.stButton > button {
    background: #1d4ed8;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 20px;
    transition: background .2s, transform .1s;
}
.stButton > button:hover { background: #2563eb; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }
.stDownloadButton > button {
    background: #0f1c2e;
    color: #93c5fd;
    border: 1px solid #1e3550;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    width: 100%;
    transition: border-color .2s, background .2s;
}
.stDownloadButton > button:hover { background: #1e3550; border-color: #3b82f6; }
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
.stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 600; }
.stAlert { border-radius: 10px !important; }

/* ── Sidebar nav ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    color: #94a3b8;
    margin-bottom: 2px;
    transition: background .15s, color .15s;
}
.nav-item:hover { background: #1e3550; color: #e2e8f0; }
.nav-item.active { background: #1d4ed8; color: white; }
.nav-item .material-icons-round { font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# ── Helper: icon heading ─────────────────────────────────────────────────────
def sec(icon, title):
    st.markdown(f"""<div class="sec-head">
        <span class="material-icons-round">{icon}</span>
        <span class="title">{title}</span>
    </div>""", unsafe_allow_html=True)

def kpi(icon, label, value, sub="", sub_cls="", accent="#3b82f6"):
    st.markdown(f"""<div class="kpi-card" style="--accent:{accent}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub {sub_cls}">{sub}</div>
    </div>""", unsafe_allow_html=True)

def info(icon, html):
    st.markdown(f"""<div class="info-banner">
        <span class="material-icons-round">{icon}</span>
        <div>{html}</div>
    </div>""", unsafe_allow_html=True)

def step(n, label):
    st.markdown(f"""<div class="step-badge">
        <span class="num">{n}</span>{label}
    </div>""", unsafe_allow_html=True)

# ── Chart defaults ───────────────────────────────────────────────────────────
BG       = "#080f1a"
BG_PANEL = "#0d1625"
BLUE     = "#3b82f6"
ORANGE   = "#f97316"
GREEN    = "#34d399"
RED      = "#f87171"
YELLOW   = "#fbbf24"
MUTED    = "#334155"
TEXT     = "#94a3b8"

def style_ax(ax, fig):
    ax.set_facecolor(BG_PANEL)
    fig.patch.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
    ax.grid(color=MUTED, linewidth=0.4, linestyle="--", alpha=0.5)

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    rmse_df     = pd.read_csv("rmse_results.csv")
    stock_dict  = joblib.load("stock_data_dict.pkl")
    return rmse_df, stock_dict

rmse_df, stock_data = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image(
            "https://upload.wikimedia.org/wikipedia/en/8/8d/Nigerian_Exchange_Group_logo.png",
            width=100, use_container_width=False,
        )
    except Exception:
        pass

    st.markdown("""
    <div style="margin:14px 0 4px">
        <div style="font-size:16px;font-weight:700;color:#f1f5f9">NGX Predictive System</div>
        <div style="font-size:11px;color:#64748b;margin-top:2px">Hybrid SARIMAX · XGBoost</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "nav",
        ["Overview", "Model Evaluation", "Stock Charts", "Predict"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"""
    <div style="font-size:12px;color:#475569;line-height:2">
        <span class="material-icons-round" style="font-size:14px;vertical-align:middle;color:#3b82f6">show_chart</span>
        &nbsp;Equities: <b style="color:#94a3b8">{len(stock_data)}</b><br>
        <span class="material-icons-round" style="font-size:14px;vertical-align:middle;color:#3b82f6">storage</span>
        &nbsp;Data: <b style="color:#94a3b8">NGX · CBN · IMF</b>
    </div>
    """, unsafe_allow_html=True)

# ── Feature engineering + model helpers ─────────────────────────────────────
def engineer_features(df, price_col="Closing_Price"):
    df = df.copy().sort_index()
    df["lag_1"]      = df[price_col].shift(1)
    df["lag_3"]      = df[price_col].shift(3)
    df["lag_5"]      = df[price_col].shift(5)
    df["lag_10"]     = df[price_col].shift(10)
    df["ma_5"]       = df[price_col].rolling(5).mean()
    df["ma_10"]      = df[price_col].rolling(10).mean()
    df["ma_20"]      = df[price_col].rolling(20).mean()
    df["std_5"]      = df[price_col].rolling(5).std()
    df["std_10"]     = df[price_col].rolling(10).std()
    df["momentum"]   = df[price_col] - df[price_col].shift(5)
    df["pct_chg"]    = df[price_col].pct_change()
    df["day_of_week"]= df.index.dayofweek
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    return df.dropna()

def train_and_forecast(df, price_col, forecast_days):
    df_feat      = engineer_features(df, price_col)
    feature_cols = [c for c in df_feat.columns if c != price_col]
    X = df_feat[feature_cols].values
    y = df_feat[price_col].values
    split = int(len(X) * 0.85)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X[:split])
    X_test_s  = scaler.transform(X[split:])

    if XGB_AVAILABLE:
        model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    else:
        model = Ridge(alpha=1.0)

    model.fit(X_train_s, y[:split])
    y_pred_test = model.predict(X_test_s)
    test_rmse   = np.sqrt(mean_squared_error(y[split:], y_pred_test))
    test_mae    = mean_absolute_error(y[split:], y_pred_test)
    test_dates  = df_feat.index[split:]

    last_known    = df[price_col].copy()
    future_prices = []
    last_date     = df.index[-1]

    for _ in range(forecast_days):
        temp_df  = pd.DataFrame({price_col: last_known}, index=last_known.index)
        feat_df  = engineer_features(temp_df, price_col)
        if feat_df.empty:
            break
        last_row = feat_df.iloc[[-1]][feature_cols].values
        pred     = model.predict(scaler.transform(last_row))[0]
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        last_known    = pd.concat([last_known, pd.Series([pred], index=[next_date])])
        future_prices.append((next_date, pred))
        last_date = next_date

    future_df = pd.DataFrame(future_prices, columns=["Date", "Forecast"]).set_index("Date")
    return dict(
        model_name   = "XGBoost" if XGB_AVAILABLE else "Ridge Regression",
        test_rmse    = test_rmse,
        test_mae     = test_mae,
        actual_train = df_feat[price_col][:split],
        actual_test  = df_feat[price_col][split:],
        pred_test    = pd.Series(y_pred_test, index=test_dates),
        forecast     = future_df,
        price_col    = price_col,
    )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div style="margin-bottom:6px">
        <div style="font-size:26px;font-weight:800;color:#f1f5f9;letter-spacing:-.3px">
            NGX Stock Market Predictor
        </div>
        <div style="font-size:14px;color:#64748b;margin-top:4px">
            Hybrid SARIMAX · XGBoost architecture &nbsp;·&nbsp; 15 listed equities
        </div>
    </div>
    """, unsafe_allow_html=True)

    avg_rmse  = rmse_df["RMSE"].mean()
    avg_mae   = rmse_df["MAE"].mean()
    best_stock = rmse_df.loc[rmse_df["RMSE"].idxmin(), "Stock Ticker"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("query_stats",    "Avg RMSE",      f"{avg_rmse:.4f}", "Across all equities")
    with c2: kpi("straighten",     "Avg MAE",        f"{avg_mae:.4f}",  "Mean Absolute Error", accent="#8b5cf6")
    with c3: kpi("military_tech",  "Best Model",     best_stock,        "Lowest RMSE",  "up", accent="#34d399")
    with c4: kpi("bar_chart",      "Equities",       "15",              "NGX listed",         accent="#f97316")

    st.divider()
    col_a, col_b = st.columns([3, 2])

    with col_a:
        sec("leaderboard", "Model Performance — RMSE vs MAE")
        fig, ax = plt.subplots(figsize=(9, 4))
        style_ax(ax, fig)
        tickers = rmse_df["Stock Ticker"].str.replace(" ", "\n", n=1)
        x = np.arange(len(tickers))
        ax.bar(x - 0.18, rmse_df["RMSE"], 0.32, label="RMSE", color=BLUE,   alpha=0.9, zorder=3)
        ax.bar(x + 0.18, rmse_df["MAE"],  0.32, label="MAE",  color=ORANGE, alpha=0.9, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, fontsize=7, rotation=45, ha="right", color=TEXT)
        ax.set_ylabel("Error value", fontsize=9)
        legend = ax.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=9, framealpha=1)
        legend.get_frame().set_edgecolor(MUTED)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        sec("table_chart", "Equity Summary")
        display_df = rmse_df[["Stock Ticker", "RMSE", "MAE"]].copy()
        display_df["RMSE"] = display_df["RMSE"].round(4)
        display_df["MAE"]  = display_df["MAE"].round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=330)

    info("lightbulb", "Navigate to <b>Predict</b> in the sidebar to upload your own stock data and generate instant forecasts.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    st.markdown("""
    <div style="font-size:22px;font-weight:800;color:#f1f5f9;margin-bottom:4px">System Accuracy &amp; Evaluation</div>
    <div style="font-size:13px;color:#64748b">RMSE · MAE · MAPE for all 15 NGX equities. Lower values indicate better accuracy.</div>
    """, unsafe_allow_html=True)

    st.dataframe(rmse_df, use_container_width=True, hide_index=True)
    st.divider()

    tab_rmse, tab_mae, tab_compare = st.tabs(["RMSE", "MAE", "RMSE vs MAE"])

    def eval_bar(col, color, ylabel):
        fig, ax = plt.subplots(figsize=(11, 4))
        style_ax(ax, fig)
        bars = ax.bar(rmse_df["Stock Ticker"], rmse_df[col], color=color, alpha=0.85, zorder=3, width=0.6)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7, color=TEXT)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticklabels(rmse_df["Stock Ticker"], rotation=45, ha="right", fontsize=8.5, color=TEXT)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_rmse:   eval_bar("RMSE", BLUE,   "RMSE")
    with tab_mae:    eval_bar("MAE",  ORANGE, "MAE")
    with tab_compare:
        fig, ax = plt.subplots(figsize=(12, 4))
        style_ax(ax, fig)
        x = np.arange(len(rmse_df)); w = 0.35
        ax.bar(x - w/2, rmse_df["RMSE"], w, label="RMSE", color=BLUE,   alpha=0.9, zorder=3)
        ax.bar(x + w/2, rmse_df["MAE"],  w, label="MAE",  color=ORANGE, alpha=0.9, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(rmse_df["Stock Ticker"], rotation=45, ha="right", fontsize=8.5, color=TEXT)
        ax.set_ylabel("Error Value", fontsize=9)
        legend = ax.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=9, framealpha=1)
        legend.get_frame().set_edgecolor(MUTED)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE — STOCK CHARTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Stock Charts":
    st.markdown("""
    <div style="font-size:22px;font-weight:800;color:#f1f5f9;margin-bottom:4px">Historical Closing Prices</div>
    <div style="font-size:13px;color:#64748b">Price trends for all 15 NGX equities over the study period.</div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Individual Charts", "Compare Two Stocks"])

    def mini_price_chart(df, ticker, ax, fig):
        style_ax(ax, fig)
        ax.plot(df.index, df["Closing_Price"], color=BLUE, linewidth=1.2, zorder=3)
        ax.fill_between(df.index, df["Closing_Price"], alpha=0.08, color=BLUE)
        ax.set_title(ticker, fontsize=9, fontweight="600", pad=8)
        ax.set_ylabel("NGN", fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    with tab1:
        cols = st.columns(2)
        for idx, ticker in enumerate(stock_data.keys()):
            df = stock_data[ticker]
            with cols[idx % 2]:
                fig, ax = plt.subplots(figsize=(6, 2.8))
                mini_price_chart(df, ticker, ax, fig)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with tab2:
        info("compare_arrows", "Overlay two equities' closing prices to compare performance directly.")
        c1, c2 = st.columns(2)
        tickers_list = list(stock_data.keys())
        with c1: stock_a = st.selectbox("First Equity",  tickers_list, key="cmp_a")
        with c2: stock_b = st.selectbox("Second Equity", tickers_list, index=1, key="cmp_b")

        fig, ax = plt.subplots(figsize=(12, 4))
        style_ax(ax, fig)
        ax.plot(stock_data[stock_a].index, stock_data[stock_a]["Closing_Price"],
                label=stock_a, color=BLUE,   lw=1.8, zorder=3)
        ax.plot(stock_data[stock_b].index, stock_data[stock_b]["Closing_Price"],
                label=stock_b, color=ORANGE, lw=1.8, zorder=3)
        ax.set_ylabel("Price (NGN)", fontsize=9)
        legend = ax.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=10, framealpha=1)
        legend.get_frame().set_edgecolor(MUTED)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE — PREDICT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown("""
    <div style="font-size:22px;font-weight:800;color:#f1f5f9;margin-bottom:4px">Real-Time Stock Prediction</div>
    <div style="font-size:13px;color:#64748b">Upload your own data or select a built-in NGX equity to generate a price forecast.</div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "source",
        ["Upload my own CSV", "Use built-in NGX equity"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # MODE A — Upload CSV
    # ══════════════════════════════════════════════════════════════════════
    if mode == "Upload my own CSV":

        # ── Template generator ─────────────────────────────────────────────
        def make_template(style):
            base_date = date(2022, 1, 3)
            all_dates = [base_date + timedelta(days=i) for i in range(400)]
            trading_dates = [d for d in all_dates if d.weekday() < 5][:120]
            np.random.seed(42)
            price = 100.0
            rows = []
            for d in trading_dates:
                open_  = round(price * (1 + np.random.uniform(-0.01, 0.01)), 2)
                high   = round(open_ * (1 + np.random.uniform(0, 0.015)), 2)
                low    = round(open_ * (1 - np.random.uniform(0, 0.015)), 2)
                close  = round((open_ + high + low) / 3 + np.random.uniform(-0.5, 0.5), 2)
                volume = int(np.random.uniform(500_000, 5_000_000))
                price  = close
                if style == "basic":
                    rows.append({"Date": d.strftime("%Y-%m-%d"), "Close": close})
                elif style == "ohlcv":
                    rows.append({"Date": d.strftime("%Y-%m-%d"), "Open": open_,
                                 "High": high, "Low": low, "Close": close, "Volume": volume})
                else:
                    rows.append({"Date": d.strftime("%Y-%m-%d"), "Closing_Price": close,
                                 "Open": open_, "High": high, "Low": low, "Volume": volume})
            buf = io.StringIO()
            pd.DataFrame(rows).to_csv(buf, index=False)
            return buf.getvalue().encode()

        # ── Step 1: Download a template ────────────────────────────────────
        step("1", "Download a template — fill it with your real data")

        tc1, tc2, tc3 = st.columns(3)

        template_specs = [
            ("tc1", "description",  "Basic",       "basic_template.csv",  "basic",
             "Minimum format — just Date and Close price. Perfect for simple price series."),
            ("tc2", "table_chart",  "OHLCV",       "ohlcv_template.csv",  "ohlcv",
             "Full Open / High / Low / Close / Volume format. Matches most broker exports."),
            ("tc3", "corporate_fare","NGX-Style",  "ngx_template.csv",    "ngx",
             "Matches the Closing_Price format used by this system's built-in equities."),
        ]

        for col_obj, icon, title, filename, style, desc in zip(
            [tc1, tc2, tc3],
            ["description", "table_chart", "corporate_fare"],
            ["Basic", "OHLCV", "NGX-Style"],
            ["basic_template.csv", "ohlcv_template.csv", "ngx_template.csv"],
            ["basic", "ohlcv", "ngx"],
            [
                "Minimum format — just Date and Close price. Perfect for simple price series.",
                "Full Open / High / Low / Close / Volume format. Matches most broker exports.",
                "Matches the Closing_Price format used by this system's built-in equities.",
            ],
        ):
            with col_obj:
                st.markdown(f"""<div class="tmpl-card">
                    <span class="material-icons-round">{icon}</span>
                    <div class="tmpl-title">{title} Template</div>
                    <div class="tmpl-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)
                st.download_button(
                    f"Download {filename}",
                    data=make_template(style),
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_{style}",
                )

        # ── Step 2: Format rules ───────────────────────────────────────────
        st.divider()
        step("2", "Format rules — read before uploading")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""<div class="rule-card">
                <div class="rule-head ok">
                    <span class="material-icons-round">check_circle</span>Required
                </div>
                <ul>
                    <li>One <b>date column</b> — any standard format
                        (<span class="pill">YYYY-MM-DD</span>
                         <span class="pill">DD/MM/YYYY</span>
                         <span class="pill">MM-DD-YYYY</span>)</li>
                    <li>One <b>numeric price column</b> (Close, Price, Last, Adj Close, Closing_Price, etc.)</li>
                    <li>Minimum <b>50 rows</b> for reliable model training</li>
                    <li>Column headers in <b>row 1</b> — one row per trading day</li>
                </ul>
            </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown("""<div class="rule-card">
                <div class="rule-head warn">
                    <span class="material-icons-round">warning</span>Common mistakes
                </div>
                <ul>
                    <li>Do <b>not</b> use commas inside numbers — write
                        <span class="pill">1500.00</span> not
                        <span class="pill">1,500.00</span></li>
                    <li>Do <b>not</b> include currency symbols — write
                        <span class="pill">245.5</span> not
                        <span class="pill">&#8358;245.5</span></li>
                    <li>Do <b>not</b> leave blank rows in the middle of the file</li>
                    <li>Do <b>not</b> include duplicate dates</li>
                </ul>
            </div>""", unsafe_allow_html=True)

        with st.expander("Accepted column name reference", expanded=False):
            ref = pd.DataFrame({
                "Column type":   ["Date", "Close / Price", "Open", "High", "Low", "Volume"],
                "Accepted names (case-insensitive)": [
                    "Date, date, Datetime, timestamp, Time, period",
                    "Close, Closing_Price, Last, Price, Adj Close, close_price",
                    "Open, open_price, Open_Price",
                    "High, high_price, High_Price",
                    "Low, low_price, Low_Price",
                    "Volume, vol, shares, traded",
                ],
                "Required?": ["Yes", "Yes (to predict)", "Optional", "Optional", "Optional", "Optional"],
            })
            st.dataframe(ref, use_container_width=True, hide_index=True)

        # ── Step 3: Upload ─────────────────────────────────────────────────
        st.divider()
        step("3", "Upload your completed CSV file")

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

        if uploaded:
            try:
                raw = pd.read_csv(uploaded)
                st.success(f"File loaded — {len(raw):,} rows × {len(raw.columns)} columns")

                with st.expander("Preview raw data", expanded=False):
                    st.dataframe(raw.head(10), use_container_width=True)

                st.divider()
                sec("tune", "Configure Columns")

                col_left, col_right, col_days = st.columns([2, 2, 1])
                with col_left:
                    date_col = st.selectbox("Date column", raw.columns.tolist())
                with col_right:
                    num_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
                    if not num_cols:
                        st.error("No numeric columns found. Check your file format.")
                        st.stop()
                    default_p = next(
                        (c for c in num_cols if any(k in c.lower() for k in ["close","price","last","adj"])),
                        num_cols[0],
                    )
                    price_col = st.selectbox("Price column to predict", num_cols,
                                             index=num_cols.index(default_p))
                with col_days:
                    forecast_days = st.number_input("Forecast days", 5, 90, 30)

                if st.button("Run Prediction", type="primary", use_container_width=True):
                    with st.spinner("Training model and generating forecast…"):
                        df = raw.copy()
                        df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
                        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
                        df = df[[price_col]].dropna()
                        if len(df) < 50:
                            st.error("Need at least 50 data points to train a reliable model.")
                            st.stop()
                        result = train_and_forecast(df, price_col, forecast_days)

                    st.divider()
                    sec("analytics", "Model Results")

                    last_price    = df[price_col].iloc[-1]
                    fc_last       = result["forecast"]["Forecast"].iloc[-1]
                    pct_change    = (fc_last - last_price) / last_price * 100
                    direction     = "up" if pct_change >= 0 else "down"
                    arrow         = "arrow_upward" if pct_change >= 0 else "arrow_downward"
                    accent_out    = GREEN if pct_change >= 0 else RED

                    m1, m2, m3, m4 = st.columns(4)
                    with m1: kpi("memory",       "Model",       result["model_name"], "", accent="#8b5cf6")
                    with m2: kpi("query_stats",  "Test RMSE",   f"{result['test_rmse']:.4f}")
                    with m3: kpi("straighten",   "Test MAE",    f"{result['test_mae']:.4f}", accent="#f97316")
                    with m4: kpi(arrow, f"{forecast_days}D Outlook", f"{fc_last:.2f}",
                                 f"{'+' if pct_change >= 0 else ''}{pct_change:.1f}% from last",
                                 direction, accent_out)

                    # Main chart
                    st.divider()
                    sec("show_chart", "Price History + Forecast")
                    fig, ax = plt.subplots(figsize=(13, 5))
                    style_ax(ax, fig)
                    ax.plot(result["actual_train"].index, result["actual_train"].values,
                            color=MUTED, lw=1, alpha=0.7, label="Training data", zorder=2)
                    ax.plot(result["actual_test"].index,  result["actual_test"].values,
                            color=BLUE,   lw=1.6, label="Actual (test)", zorder=3)
                    ax.plot(result["pred_test"].index,    result["pred_test"].values,
                            color=ORANGE, lw=1.6, linestyle="--", label="Predicted (test)", zorder=3)
                    ax.plot(result["forecast"].index, result["forecast"]["Forecast"].values,
                            color=GREEN,  lw=2.2, marker="o", markersize=3.5,
                            label=f"{forecast_days}-day forecast", zorder=4)
                    ax.axvline(df.index[-1], color=YELLOW, lw=1, linestyle=":", alpha=0.9)
                    ax.set_ylabel(price_col, fontsize=9)
                    legend = ax.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=9, framealpha=1, ncol=2)
                    legend.get_frame().set_edgecolor(MUTED)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Forecast table + download
                    st.divider()
                    sec("event_note", "Day-by-Day Forecast")
                    fc_display = result["forecast"].copy()
                    fc_display.index = fc_display.index.strftime("%Y-%m-%d")
                    fc_display["Forecast"] = fc_display["Forecast"].round(4)
                    fc_display["Daily Change"] = fc_display["Forecast"].diff().round(4)
                    fc_display["Daily Change %"] = (fc_display["Forecast"].pct_change() * 100).round(2)
                    st.dataframe(fc_display, use_container_width=True)
                    st.download_button(
                        "Download forecast CSV",
                        fc_display.to_csv().encode(),
                        file_name="forecast.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")

        else:
            info("upload_file", "Drag and drop your CSV file above to get started. "
                 "Not sure about the format? Download one of the templates in Step 1.")

    # ══════════════════════════════════════════════════════════════════════
    # MODE B — Built-in NGX equity
    # ══════════════════════════════════════════════════════════════════════
    else:
        sec("corporate_fare", "Select an NGX Equity")
        col_sel, col_fcast = st.columns([3, 1])
        with col_sel:
            ticker = st.selectbox("Equity", list(stock_data.keys()), label_visibility="collapsed")
        with col_fcast:
            forecast_days = st.number_input("Forecast days", 5, 90, 30, label_visibility="collapsed")

        df_sel     = stock_data[ticker][["Closing_Price"]].copy()
        last_price = df_sel["Closing_Price"].iloc[-1]
        first_price= df_sel["Closing_Price"].iloc[0]
        tot_return = (last_price - first_price) / first_price * 100
        ret_dir    = "up" if tot_return >= 0 else "down"
        rmse_row   = rmse_df[rmse_df["Stock Ticker"] == ticker]
        rmse_val   = f"{rmse_row['RMSE'].values[0]:.4f}" if not rmse_row.empty else "N/A"

        s1, s2, s3, s4 = st.columns(4)
        with s1: kpi("price_change",  "Last Price",     f"NGN {last_price:.2f}")
        with s2: kpi("trending_up" if tot_return >= 0 else "trending_down",
                     "Total Return",  f"{tot_return:.1f}%",
                     f"Since {df_sel.index[0].year}", ret_dir,
                     accent=GREEN if tot_return >= 0 else RED)
        with s3: kpi("calendar_today","Data Points",    f"{len(df_sel):,}", "Trading days", accent="#8b5cf6")
        with s4: kpi("query_stats",   "Hybrid RMSE",    rmse_val, accent="#f97316")

        if st.button("Generate Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Training on {ticker} — forecasting {forecast_days} days…"):
                result = train_and_forecast(df_sel, "Closing_Price", forecast_days)

            st.divider()
            fc_last  = result["forecast"]["Forecast"].iloc[-1]
            fc_pct   = (fc_last - last_price) / last_price * 100
            fc_dir   = "up" if fc_pct >= 0 else "down"
            fc_arrow = "arrow_upward" if fc_pct >= 0 else "arrow_downward"
            fc_accent= GREEN if fc_pct >= 0 else RED

            sec("analytics", "Forecast Results")
            r1, r2, r3, r4 = st.columns(4)
            with r1: kpi("memory",      "Model",     result["model_name"], accent="#8b5cf6")
            with r2: kpi("query_stats", "Test RMSE", f"{result['test_rmse']:.4f}")
            with r3: kpi("straighten",  "Test MAE",  f"{result['test_mae']:.4f}", accent="#f97316")
            with r4: kpi(fc_arrow, f"{forecast_days}D Target", f"NGN {fc_last:.2f}",
                         f"{'+' if fc_pct >= 0 else ''}{fc_pct:.1f}%", fc_dir, fc_accent)

            # Main 2-panel chart
            st.divider()
            sec("show_chart", f"{ticker} — Price History + Forecast")
            fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                                     gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
            fig.patch.set_facecolor(BG)

            ax = axes[0]
            style_ax(ax, fig)
            ax.plot(result["actual_train"].index, result["actual_train"].values,
                    color=MUTED, lw=1, alpha=0.6, label="Training", zorder=2)
            ax.plot(result["actual_test"].index,  result["actual_test"].values,
                    color=BLUE,   lw=1.6, label="Actual (test)", zorder=3)
            ax.plot(result["pred_test"].index,    result["pred_test"].values,
                    color=ORANGE, lw=1.6, linestyle="--", label="Predicted (test)", zorder=3)
            ax.plot(result["forecast"].index, result["forecast"]["Forecast"].values,
                    color=GREEN,  lw=2.4, marker="o", markersize=4, label=f"{forecast_days}d Forecast", zorder=4)
            ax.axvline(df_sel.index[-1], color=YELLOW, lw=1.2, linestyle=":", alpha=0.9, label="Today")
            ax.fill_between(result["forecast"].index,
                            result["forecast"]["Forecast"] * 0.95,
                            result["forecast"]["Forecast"] * 1.05,
                            alpha=0.12, color=GREEN, label="±5% band")
            ax.set_ylabel("Price (NGN)", fontsize=9)
            ax.set_xticklabels([])
            legend = ax.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=9, framealpha=1, ncol=3)
            legend.get_frame().set_edgecolor(MUTED)

            ax2 = axes[1]
            style_ax(ax2, fig)
            residuals  = result["actual_test"].values - result["pred_test"].values
            bar_colors = [GREEN if r >= 0 else RED for r in residuals]
            ax2.bar(result["actual_test"].index, residuals, color=bar_colors, alpha=0.75, width=1, zorder=3)
            ax2.axhline(0, color=YELLOW, lw=0.8, linestyle="--")
            ax2.set_ylabel("Residual", fontsize=8)
            ax2.set_title("Prediction Residuals (Actual − Predicted)", fontsize=9, pad=6)
            ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Table + download
            st.divider()
            sec("event_note", "Day-by-Day Forecast")
            fc_display = result["forecast"].copy()
            fc_display.index = fc_display.index.strftime("%Y-%m-%d")
            fc_display["Forecast (NGN)"] = fc_display["Forecast"].round(2)
            fc_display["Daily Change"]   = fc_display["Forecast"].diff().round(2)
            fc_display["Daily Change %"] = (fc_display["Forecast"].pct_change() * 100).round(2)
            fc_display = fc_display.drop(columns=["Forecast"])
            st.dataframe(fc_display, use_container_width=True)
            st.download_button(
                "Download forecast CSV",
                fc_display.to_csv().encode(),
                file_name=f"{ticker.replace(' ','_')}_forecast.csv",
                mime="text/csv",
            )

        else:
            # Historical chart placeholder
            st.divider()
            sec("candlestick_chart", "Historical Closing Price")
            fig, ax = plt.subplots(figsize=(13, 4))
            style_ax(ax, fig)
            ax.plot(df_sel.index, df_sel["Closing_Price"], color=BLUE, lw=1.4, zorder=3)
            ax.fill_between(df_sel.index, df_sel["Closing_Price"], alpha=0.08, color=BLUE)
            ax.set_ylabel("Price (NGN)", fontsize=9)
            ax.set_title(f"{ticker} — Historical Closing Price", fontsize=11, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            info("smart_toy", "Press <b>Generate Forecast</b> above to run the prediction model on this equity.")
