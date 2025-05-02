import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

st.title("CAPM and Sharpe Ratio Calculator")

stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")
index_ticker = st.text_input("Enter index ticker (e.g., ^GSPC for S&P 500)", "^GSPC")

def fetch_yahoo_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch daily adjusted-close history for `ticker` from Yahoo Finance.
    """
    period1 = int(start.timestamp())
    period2 = int(end.timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    r = requests.get(url, params=params, headers=headers)
    if r.status_code != 200:
        st.error(f"HTTP {r.status_code} error fetching {ticker}")
        return pd.DataFrame()
    try:
        js = r.json()
    except ValueError:
        st.error(f"Invalid JSON for {ticker}")
        return pd.DataFrame()

    result = js.get("chart", {}).get("result")
    if not result:
        st.error(f"No data returned for {ticker}")
        return pd.DataFrame()

    data = result[0]
    timestamps = data["timestamp"]
    adj = data["indicators"]["adjclose"][0]["adjclose"]
    df = pd.DataFrame({"adjclose": adj}, index=pd.to_datetime(timestamps, unit="s"))
    return df

if st.button("Calculate"):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)

    with st.spinner("Downloading stock data..."):
        stock_data = fetch_yahoo_history(stock_ticker, start_date, end_date)
    with st.spinner("Downloading index data..."):
        index_data = fetch_yahoo_history(index_ticker, start_date, end_date)

    if stock_data.empty or index_data.empty:
        st.stop()

    # Align data by date
    combined_data = pd.concat([stock_data["adjclose"].rename('stock'), index_data["adjclose"].rename('index')], axis=1).dropna()
    
    # Calculate simple returns (for CAPM)
    stock_returns = combined_data['stock'].pct_change().dropna()
    index_returns = combined_data['index'].pct_change().dropna()

    # CAPM beta
    cov = np.cov(stock_returns, index_returns)[0][1]
    beta = cov / np.var(index_returns)

    # CAPM Expected Return (assuming risk-free rate = 0)
    market_return_annual = index_returns.mean() * 252
    capm_return = beta * market_return_annual

    # Calculate log returns for Sharpe (aligned dates)
    stock_log_returns = np.log(combined_data['stock'] / combined_data['stock'].shift(1)).dropna()
    
    # Sharpe Ratio (annualized)
    sharpe = (stock_log_returns.mean() / stock_log_returns.std()) * np.sqrt(252)

    # Display results
    st.metric("Beta", f"{beta:.4f}")
    st.metric("CAPM Expected Return", f"{capm_return * 100:.2f}%")
    st.metric("Sharpe Ratio", f"{sharpe:.4f}")
