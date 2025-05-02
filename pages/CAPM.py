import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import pandas_datareader.data as web
import yfinance as yf  # We'll keep yfinance as a fallback

st.title("CAPM and Sharpe Ratio Calculator")

stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")
index_ticker = st.text_input("Enter index ticker (e.g., ^GSPC for S&P 500)", "^GSPC")

# Add a disclaimer about data sources
st.markdown("""
<small>Note: This app fetches data using pandas-datareader with Yahoo Finance as the data source. 
If the main data source fails, it will try yfinance as a fallback.</small>
""", unsafe_allow_html=True)

if st.button("Calculate"):
    try:
        # Use a 5-year period to ensure recent data
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5*365)
        
        # Fetch data with progress and error handling
        def get_stock_data_with_retry(ticker, start_date, end_date, max_retries=3):
            # Try all available methods until one works
            methods = [
                lambda: web.DataReader(ticker, 'yahoo', start_date, end_date),  # pandas_datareader
                lambda: yf.download(ticker, start=start_date, end=end_date, progress=False)  # yfinance
            ]
            
            errors = []
            for method_index, method in enumerate(methods):
                for attempt in range(max_retries):
                    try:
                        data = method()
                        if data is not None and not data.empty:
                            method_name = "pandas_datareader" if method_index == 0 else "yfinance"
                            st.info(f"Successfully fetched data for {ticker} using {method_name}")
                            return data
                    except Exception as e:
                        error_msg = f"Method {method_index+1}, attempt {attempt+1} failed: {str(e)}"
                        errors.append(error_msg)
                        if attempt < max_retries - 1:
                            wait_time = random.uniform(1, 3) * (attempt + 1)
                            time.sleep(wait_time)
            
            # If we get here, all methods failed
            error_details = "\n".join(errors)
            raise Exception(f"Failed to fetch data for {ticker} after trying all methods:\n{error_details}")
        
        with st.spinner(f"Downloading data for {stock_ticker}..."):
            stock_data = get_stock_data_with_retry(stock_ticker, start_date, end_date)
        with st.spinner(f"Downloading data for {index_ticker}..."):
            index_data = get_stock_data_with_retry(index_ticker, start_date, end_date)
        
        # Debug: Show downloaded data info
        st.write(f"Stock data rows: {len(stock_data)}, Index data rows: {len(index_data)}")
        
        if stock_data.empty:
            st.error(f"No stock data for {stock_ticker}. Check ticker on Yahoo Finance.")
            st.stop()
        if index_data.empty:
            st.error(f"No index data for {index_ticker}. Check ticker on Yahoo Finance.")
            st.stop()

        # Handle different column naming conventions between libraries
        def get_close_column(df):
            if 'Close' in df.columns:  # Both methods should now use uppercase
                return df['Close']
            elif 'close' in df.columns:  # Just in case
                return df['close']
            else:
                # Display available columns for debugging
                st.error(f"Available columns: {df.columns.tolist()}")
                raise ValueError("Could not find close price column in data")
        
        # Get close prices
        stock_close = get_close_column(stock_data).squeeze()  # Convert to Series if needed
        index_close = get_close_column(index_data).squeeze()

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
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        st.stop()
