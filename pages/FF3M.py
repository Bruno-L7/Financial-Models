import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web
import yfinance as yf
import streamlit as st
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Streamlit App Title
st.title("Fama-French Three Factor Model Analysis")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('2020-12-31'))
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
use_robust_errors = st.sidebar.checkbox("Use Robust Standard Errors (HC3)", value=False)

# Fetch Fama-French data
@st.cache_data  # Cache data to improve performance
def fetch_fama_french(start_date, end_date):
    try:
        ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start_date, end_date)
        ff_factors = ff_data[0]
        ff_factors.index = ff_factors.index.to_timestamp(how='e')  # Convert to month-end dates
        ff_factors.index = ff_factors.index.normalize()  # Remove time component (if any)
        return ff_factors
    except Exception as e:
        st.error(f"Fama-French Error: {e}")
        return pd.DataFrame()

# Fetch stock data using yfinance instead of pandas_datareader
@st.cache_data
def fetch_stock_data(stock_symbol, start_date, end_date):
    try:
        # Use yfinance to download the data
        stock_data = yf.download(
            stock_symbol, 
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Check if we got valid data
        if stock_data is None or stock_data.empty:
            st.warning(f"No data found for {stock_symbol} in the given date range.")
            return pd.DataFrame()  # Return empty DataFrame instead of Series
        
        # Process data
        stock_returns = stock_data['Close'].resample('M').last().pct_change().dropna()
        # Create a DataFrame with the returns
        returns_df = pd.DataFrame(stock_returns)
        returns_df.columns = ['Stock_Return']  # Name the column
        returns_df.index = returns_df.index.to_period('M').to_timestamp('M')
        return returns_df
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of Series

# Main App Logic
with st.spinner('Fetching Fama-French factors...'):
    ff_factors = fetch_fama_french(start_date, end_date)

with st.spinner(f'Fetching stock data for {stock_symbol}...'):
    stock_returns_df = fetch_stock_data(stock_symbol, start_date, end_date)

# Merge datasets
if not ff_factors.empty and not stock_returns_df.empty:
    merged_data = pd.merge(
        stock_returns_df,  # Already a DataFrame, no need for to_frame()
        ff_factors,
        left_index=True,
        right_index=True,
        how='inner'
    )
    merged_data.dropna(inplace=True)
    
    if not merged_data.empty:
        st.write("### Merged Data (First 5 Rows)")
        st.dataframe(merged_data.head())
        
        # Prepare variables
        merged_data['Excess_Return'] = merged_data['Stock_Return'] - merged_data['RF'] / 100
        X = merged_data[['Mkt-RF', 'SMB', 'HML']] / 100
        X = sm.add_constant(X)
        y = merged_data['Excess_Return']
        
        # Run regression
        if use_robust_errors:
            results = sm.OLS(y, X).fit(cov_type='HC3')
        else:
            results = sm.OLS(y, X).fit()
        
        # Display regression results
        st.write("### Regression Results")
        st.write(results.summary())
        
        # Display key metrics
        st.write("### Key Metrics")
        st.write(f"- **Alpha (Intercept):** {results.params['const']:.4f}")
        st.write(f"- **Market Beta (Mkt-RF):** {results.params['Mkt-RF']:.4f}")
        st.write(f"- **SMB Beta (Size):** {results.params['SMB']:.4f}")
        st.write(f"- **HML Beta (Value):** {results.params['HML']:.4f}")
        st.write(f"- **R-squared:** {results.rsquared:.4f}")
        
        # Add interpretation
        st.write("### Interpretation")
        alpha = results.params['const']
        if alpha > 0:
            st.write(f"The stock has a positive alpha of {alpha:.4f}, suggesting it outperformed the market on a risk-adjusted basis.")
        elif alpha < 0:
            st.write(f"The stock has a negative alpha of {alpha:.4f}, suggesting it underperformed the market on a risk-adjusted basis.")
        else:
            st.write("The stock's performance is in line with what would be expected based on its risk factors.")
            
        market_beta = results.params['Mkt-RF']
        if market_beta > 1:
            st.write(f"With a market beta of {market_beta:.4f}, this stock tends to be more volatile than the market.")
        elif market_beta < 1:
            st.write(f"With a market beta of {market_beta:.4f}, this stock tends to be less volatile than the market.")
        else:
            st.write(f"With a market beta of {market_beta:.4f}, this stock moves in line with the market.")
            
        smb = results.params['SMB']
        if abs(smb) > 0.2:
            size_exposure = "strong" if abs(smb) > 0.5 else "moderate"
            size_direction = "small-cap" if smb > 0 else "large-cap"
            st.write(f"The stock shows a {size_exposure} {size_direction} tilt (SMB beta: {smb:.4f}).")
            
        hml = results.params['HML']
        if abs(hml) > 0.2:
            value_exposure = "strong" if abs(hml) > 0.5 else "moderate"
            value_direction = "value" if hml > 0 else "growth"
            st.write(f"The stock shows a {value_exposure} {value_direction} tilt (HML beta: {hml:.4f}).")
    else:
        st.error("No overlapping dates. Please check the date range or try a different stock symbol.")
else:
    if ff_factors.empty:
        st.error("Failed to load Fama-French data. Please check your date range.")
    if stock_returns_df.empty:
        st.error("Failed to load stock data. Please check your ticker symbol and date range.")