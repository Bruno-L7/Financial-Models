import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web
from yahoo_fin import stock_info as si
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
        ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date, end=end_date)
        ff_factors = ff_data[0]
        ff_factors.index = ff_factors.index.to_timestamp(how='e')  # Convert to month-end dates
        ff_factors.index = ff_factors.index.normalize()  # Remove time component (if any)
        return ff_factors
    except Exception as e:
        st.error(f"Fama-French Error: {e}")
        return pd.DataFrame()

# Fetch stock data
@st.cache_data  # Cache data to improve performance
def fetch_stock_data(stock_symbol, start_date, end_date):
    try:
        stock_data = si.get_data(stock_symbol, start_date=start_date, end_date=end_date)
        if not stock_data.empty:
            stock_data.index = stock_data.index.tz_localize(None)  # Remove timezone
            # Resample to calendar month-end (force alignment)
            stock_returns = stock_data['adjclose'].resample('M').last().pct_change().dropna()
            stock_returns.index = stock_returns.index.to_period('M').to_timestamp('M')  # Force month-end
            stock_returns.name = 'Stock_Return'
            return stock_returns
        else:
            st.warning(f"Stock data for {stock_symbol} is empty.")
            return pd.Series()
    except Exception as e:
        st.error(f"Stock Error: {e}")
        return pd.Series()

# Main App Logic
ff_factors = fetch_fama_french(start_date, end_date)
stock_returns = fetch_stock_data(stock_symbol, start_date, end_date)

# Merge datasets
if not ff_factors.empty and not stock_returns.empty:
    merged_data = pd.merge(
        stock_returns.to_frame(),
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
    else:
        st.error("No overlapping dates. Please check the date range or try a different stock symbol.")
else:
    st.error("Failed to load Fama-French or stock data. Please check your inputs.")
