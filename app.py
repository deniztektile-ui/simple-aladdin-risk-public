import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Simple Aladdin Risk", page_icon="📊", layout="wide")

st.title("📊 Simple Aladdin Risk Dashboard")
st.caption("Educational prototype inspired by BlackRock Aladdin · Not for real trading")

st.sidebar.header("Portfolio Settings")

default_tickers = "AAPL, MSFT, GOOGL, AMZN, TSLA"
tickers_input = st.sidebar.text_input("Tickers (comma separated)", default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

lookback = st.sidebar.slider("Lookback period (days)", 90, 1000, 730)
risk_free = st.sidebar.number_input("Risk-free rate", 0.0, 0.1, 0.04, 0.005)

weights = np.array([1/len(tickers)] * len(tickers))

try:
    end = datetime.now()
    start = end - timedelta(days=lookback)
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"].dropna()
    returns = prices.pct_change().dropna()

    port_returns = returns.dot(weights)
    ann_return = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0

    var_95 = -np.percentile(port_returns, 5)
    var_99 = -np.percentile(port_returns, 1)
    cvar_95 = -port_returns[port_returns <= -var_95].mean()

    cumulative = (1 + port_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Return", f"{ann_return:.2%}")
    col2.metric("Volatility", f"{ann_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{max_dd:.2%}")

    col5, col6, col7 = st.columns(3)
    col5.metric("VaR 95%", f"{var_95:.2%}")
    col6.metric("VaR 99%", f"{var_99:.2%}")
    col7.metric("CVaR 95%", f"{cvar_95:.2%}")

    st.subheader("Portfolio Performance")
    fig_cum = px.line(cumulative, title="Cumulative Return")
    st.plotly_chart(fig_cum, use_container_width=True)

    st.subheader("Drawdown")
    fig_dd = px.area(drawdown, title="Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = returns.corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Return Distribution")
    fig_hist = px.histogram(port_returns, nbins=50, title="Daily Returns Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {e}")

st.sidebar.markdown("---")
st.sidebar.info("This is an educational prototype only.")
