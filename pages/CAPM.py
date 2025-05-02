import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.title("CAPM and Sharpe Ratio Calculator")

stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")
index_ticker = st.text_input("Enter index ticker (e.g., ^GSPC for S&P 500)", "^GSPC")

if st.button("Calculate"):
    try:
        # Use a 5-year period to ensure recent data
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5*365)
        
        # Fetch data with progress
        with st.spinner("Downloading stock data..."):
            stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
        with st.spinner("Downloading index data..."):
            index_data = yf.download(index_ticker, start=start_date, end=end_date)
        
        # Debug: Show downloaded data info
        st.write(f"Stock data rows: {len(stock_data)}, Index data rows: {len(index_data)}")
        
        if stock_data.empty:
            st.error(f"No stock data for {stock_ticker}. Check ticker on Yahoo Finance.")
            st.stop()
        if index_data.empty:
            st.error(f"No index data for {index_ticker}. Check ticker on Yahoo Finance.")
            st.stop()

        # Ensure 'Close' columns exist and are Series (not DataFrames)
        stock_close = stock_data['Close'].squeeze()  # Convert to Series if needed
        index_close = index_data['Close'].squeeze()

        # Calculate log returns
        stock_returns = np.log(1 + stock_close.pct_change().dropna())
        index_returns = np.log(1 + index_close.pct_change().dropna())

        # Align dates explicitly
        common_dates = stock_returns.index.intersection(index_returns.index)
        if len(common_dates) < 2:  # Need at least 2 data points for covariance
            st.error("Insufficient overlapping data points between stock and index.")
            st.stop()

        stock_returns_aligned = stock_returns.loc[common_dates]
        index_returns_aligned = index_returns.loc[common_dates]

        # Calculate covariance and beta (explicitly convert to float)
        covariance = np.cov(stock_returns_aligned, index_returns_aligned)[0, 1]
        market_variance = index_returns_aligned.var()
        beta = float(covariance / market_variance)  # Force scalar

        # CAPM components
        rf = 0.0137  # Risk-free rate
        market_return = float(index_returns_aligned.mean() * 252)  # Force scalar
        capm_return = float(rf + beta * (market_return - rf))

        # Sharpe Ratio (ensure scalar)
        stock_volatility = float(stock_returns_aligned.std() * np.sqrt(252))
        sharpe_ratio = float((capm_return - rf) / stock_volatility)

        # Display results
        st.success(f"""
        **Results for {stock_ticker}:**
        - Beta: {beta:.2f}
        - CAPM Return: {capm_return*100:.2f}%
        - Sharpe Ratio: {sharpe_ratio:.2f}
        """)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()
