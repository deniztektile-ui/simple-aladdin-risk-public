#!/usr/bin/env python3
"""
Simplified Portfolio Risk Analyzer
Educational prototype inspired by BlackRock Aladdin
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
WEIGHTS = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
LOOKBACK_DAYS = 365 * 2
RISK_FREE_RATE = 0.04

def download_data(tickers, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return data.dropna()

def calculate_returns(prices):
    return prices.pct_change().dropna()

def portfolio_performance(returns, weights):
    port_returns = returns.dot(weights)
    mean_return = port_returns.mean() * 252
    volatility = port_returns.std() * np.sqrt(252)
    return port_returns, mean_return, volatility

def historical_var(returns, confidence=0.95):
    return -np.percentile(returns, (1 - confidence) * 100)

def cvar(returns, confidence=0.95):
    var = historical_var(returns, confidence)
    return -returns[returns <= -var].mean()

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def sharpe_ratio(ann_return, ann_vol, risk_free=0.04):
    return (ann_return - risk_free) / ann_vol if ann_vol != 0 else 0

def main():
    print("=" * 65)
    print("  Simple Aladdin Risk Analyzer (Educational Prototype)")
    print("=" * 65)
    print(f"Portfolio: {TICKERS}")
    print(f"Weights:   {WEIGHTS.tolist()}")
    print()

    print("Downloading data...")
    prices = download_data(TICKERS, LOOKBACK_DAYS)
    returns = calculate_returns(prices)

    port_returns, ann_return, ann_vol = portfolio_performance(returns, WEIGHTS)

    print(f"Annual Return:          {ann_return:.2%}")
    print(f"Annual Volatility:      {ann_vol:.2%}")
    print(f"Sharpe Ratio:           {sharpe_ratio(ann_return, ann_vol):.2f}")
    print(f"Historical VaR (95%):   {historical_var(port_returns, 0.95):.2%}")
    print(f"Historical VaR (99%):   {historical_var(port_returns, 0.99):.2%}")
    print(f"CVaR (Expected Shortfall): {cvar(port_returns, 0.95):.2%}")
    print(f"Max Drawdown:           {max_drawdown(port_returns):.2%}")

    print("\nCorrelation Matrix:")
    print(returns.corr().round(2))

    print("\n" + "=" * 65)
    print("Educational prototype only. Do not use for real trading decisions.")
    print("=" * 65)

if __name__ == "__main__":
    main()
