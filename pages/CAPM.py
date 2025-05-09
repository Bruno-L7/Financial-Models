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
        # Simplified date calculation
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5*365)
        
        st.write(f"Debug - Start Date: {start_date.strftime('%Y-%m-%d')}")
        st.write(f"Debug - End Date: {end_date.strftime('%Y-%m-%d')}")

        # Removed custom session with User-Agent as it might cause issues
        # session = requests.Session()
        # session.headers.update({
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        # })

        # Use the built-in download function without custom session
        with st.spinner("Downloading stock data..."):
            stock_data = yf.download(
                stock_ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            # Debug: Show raw stock data
            st.write("Debug - Stock Data Head:", stock_data.head(3))
            st.write("Debug - Stock Data Tail:", stock_data.tail(3))

        with st.spinner("Downloading index data..."):
            index_data = yf.download(
                index_ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            st.write("Debug - Index Data Head:", index_data.head(3))
            st.write("Debug - Index Data Tail:", index_data.tail(3))

        st.write(f"Debug - Stock data rows: {len(stock_data)}, Index data rows: {len(index_data)}")
        
        if stock_data.empty:
            st.error(f"No stock data for {stock_ticker}. Check ticker on Yahoo Finance.")
            st.stop()
        if index_data.empty:
            st.error(f"No index data for {index_ticker}. Check ticker on Yahoo Finance.")
            st.stop()

        # Rest of your code remains the same...
        stock_close = stock_data['Close'].squeeze()
        index_close = index_data['Close'].squeeze()

        stock_returns = np.log(1 + stock_close.pct_change().dropna())
        index_returns = np.log(1 + index_close.pct_change().dropna())

        common_dates = stock_returns.index.intersection(index_returns.index)
        if len(common_dates) < 2:  
            st.error("Insufficient overlapping data points between stock and index.")
            st.stop()

        stock_returns_aligned = stock_returns.loc[common_dates]
        index_returns_aligned = index_returns.loc[common_dates]
        covariance = np.cov(stock_returns_aligned, index_returns_aligned)[0, 1]
        market_variance = index_returns_aligned.var()
        beta = float(covariance / market_variance) 
        rf = 0.0137
        market_return = float(index_returns_aligned.mean() * 252)
        capm_return = float(rf + beta * (market_return - rf))
        stock_volatility = float(stock_returns_aligned.std() * np.sqrt(252))
        sharpe_ratio = float((capm_return - rf) / stock_volatility)

        st.success(f"""
        **Results for {stock_ticker}:**
        - Beta: {beta:.2f}
        - CAPM Return: {capm_return*100:.2f}%
        - Sharpe Ratio: {sharpe_ratio:.2f}
        """)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()
