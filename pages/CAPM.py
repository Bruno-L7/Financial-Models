import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# Streamlit application
st.title("CAPM and Sharpe Ratio Calculator")

# Input for stock symbol
stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "SSNLF")

# Input for index symbol
index_ticker = st.text_input("Enter index ticker (e.g., S&P = ^GSPC, Nasdaq = ^IXIC, Dow Jones = DJIA)", "^GSPC")

# Button to calculate metrics
if st.button("Calculate"):
    # Download data for stock and index
    data = {}
    for ticker in [stock_ticker, index_ticker]:
        try:
            data[ticker] = yf.download(ticker, start='2013-01-01')
            if data[ticker].empty:
                st.error(f"No data found for {ticker}. Please check the ticker symbol.")
                break
            
            # Display the downloaded data for debugging
            st.write(f"Data for {ticker}:")
            st.write(data[ticker])  # Show the DataFrame
            
            # Check if 'Adj Close' column exists
            if 'AdjClose' not in data[ticker].columns:
                st.error(f"'AdjClose' column not found for {ticker}.")
                break
            
        except Exception as e:
            st.error(f"Error downloading data for {ticker}: {e}")
            break

    # Proceed only if data is available
    if stock_ticker in data and index_ticker in data:
        # Calculate log returns
        log_returns = {}
        for ticker in [stock_ticker, index_ticker]:
            log_returns[ticker] = np.log(1 + data[ticker]['AdjClose'].pct_change())

        # Calculate covariance matrix
        cov = pd.concat(log_returns, axis=1).cov() * 252

        # Calculate Beta
        cov_market = cov.iloc[0, 1]  # Covariance between stock and index
        market_var = log_returns[index_ticker].var() * 252  # Variance of the index
        stock_beta = cov_market / market_var

        # Calculate CAPM return
        rf = 0.0137  # Risk-free rate
        riskpremium = (log_returns[index_ticker].mean() * 252) - rf
        stock_capm_return = rf + stock_beta * riskpremium

        # Calculate Sharpe ratio
        sharpe = (stock_capm_return - rf) / (log_returns[stock_ticker].std() * 252 ** 0.5)

        # Display results
        st.write(f"The Beta of {stock_ticker} is: {round(stock_beta, 3)}")
        st.write(f"The CAPM return of {stock_ticker} is: {round(stock_capm_return * 100, 3)}%")
        st.write(f"The Sharpe ratio of {stock_ticker} is: {round(sharpe, 3)}")

        # Risk assessment based on Beta
        if stock_beta > 1:
            st.warning("Beta = RISKY")
        elif stock_beta < 1:
            st.success("Beta = NOT Risky")
        else:
            st.info("Beta = BUY! BUY! BUY!")
