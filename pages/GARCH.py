import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from yahoo_fin.stock_info import get_data
import plotly.express as px

st.set_page_config(page_title="GARCH Volatility Forecast", layout="wide")

# Sidebar for user inputs
with st.sidebar:
    st.header("Model Parameters")
    ticker = st.text_input("Stock Ticker", value="SPY")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    p = st.slider("GARCH p (lag order)", 1, 5, 1)
    q = st.slider("GARCH q (error terms)", 1, 5, 1)
    model_type = st.selectbox("Volatility Model", ["GARCH", "EGARCH", "HARCH"])
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 5)

def load_data(ticker, start_date, end_date):
    """Fetch stock data using yahoo_fin"""
    try:
        data = get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True)
        return data['adjclose']
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def check_stationarity(series, sig_level=0.05):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value < sig_level  # Return True if stationary

def make_stationary(series):
    """Make series stationary by differencing if necessary"""
    if check_stationarity(series):
        return series, False  # Already stationary, no differencing
    else:
        return series.diff().dropna(), True  # Differenced series

# Main app
st.title("Volatility Forecasting with GARCH Models")
prices = load_data(ticker, start_date, end_date)

if prices is not None:
    # Convert to returns
    returns = 100 * prices.pct_change().dropna()
    
    # Make returns stationary if necessary
    stationary_returns, was_differenced = make_stationary(returns)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data Diagnostics", "Model Analysis", "Forecasts"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{ticker} Price Data")
            st.line_chart(prices)
        with col2:
            st.subheader("Daily Returns")
            st.line_chart(returns)
        
        st.markdown("### Stationarity Check")
        st.write(f"Original returns stationarity: {'Stationary' if check_stationarity(returns) else 'Non-stationary'}")
        if was_differenced:
            st.warning("⚠️ Returns were non-stationary - applied first-order differencing")
            st.write(f"Differenced returns stationarity: {'Stationary' if check_stationarity(stationary_returns) else 'Non-stationary'}")
    
with tab2:
    if check_stationarity(stationary_returns):
        # Fit GARCH model
        model = arch_model(stationary_returns, vol=model_type, p=p, q=q)
        model_fit = model.fit(disp='off')
        
        st.subheader("Model Summary")
        
        # Get the full model summary
        summary = model_fit.summary()
        
        # Display each table in the summary
        for i, table in enumerate(summary.tables):
            st.markdown(f"**Table {i + 1}**")
            table_df = pd.DataFrame(table.data[1:], columns=table.data[0])
            
            # Identify numeric columns for formatting
            numeric_cols = table_df.columns[table_df.dtypes.apply(lambda x: np.issubdtype(x, np.number))]
            
            # Apply formatting only to numeric columns
            styled_df = table_df.style.format({col: "{:.4f}" for col in numeric_cols})
            
            # Display the styled DataFrame
            st.dataframe(styled_df)
        
        st.subheader("Conditional Volatility")
        
        # Create a DataFrame for the conditional volatility
        volatility_df = pd.DataFrame({
            'Date': stationary_returns.index,
            'Conditional Volatility': model_fit.conditional_volatility
        })
        
        # Create an interactive Plotly line plot
        fig = px.line(
            volatility_df,
            x='Date',
            y='Conditional Volatility',
            title='Conditional Volatility Over Time',
            labels={'Conditional Volatility': 'Volatility', 'Date': 'Date'},
            hover_data={'Conditional Volatility': ':.4f'}  # Format hover values
        )
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Volatility',
            hovermode='x unified'  # Show hover info for all points on the x-axis
        )
        
        # Display the Plotly graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Store model fit in session state for Tab 3
        st.session_state.model_fit = model_fit
    else:
        st.error("""
        ❌ Cannot model GARCH: 
        Returns series is still non-stationary after differencing.
        Try a different time period or asset.
        """)

    with tab3:
        if 'model_fit' in st.session_state:
            st.subheader(f"{forecast_horizon}-Day Volatility Forecast")
            
            # Generate forecasts
            forecasts = st.session_state.model_fit.forecast(horizon=forecast_horizon)
            forecast_var = forecasts.variance.iloc[-1].values
            forecast_vol = np.sqrt(forecast_var)
            
            # Create forecast DataFrame
            forecast_dates = pd.date_range(stationary_returns.index[-1], periods=forecast_horizon+1)[1:]
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Volatility': forecast_vol
            }).set_index('Date')
            
            # Plot forecasts
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(forecast_df, marker='o', linestyle='--', color='red')
            ax.set_title("Volatility Forecast")
            st.pyplot(fig)
            
            # Display forecast table
            st.write("### Forecast Values")
            st.dataframe(forecast_df.style.format("{:.4f}"))
        else:
            st.warning("Generate model in Tab 2 first")

st.sidebar.markdown("""
**Instructions:**
1. Enter valid stock ticker
2. Adjust parameters in sidebar
3. View diagnostics in Tab 1
4. Generate model in Tab 2
5. View forecasts in Tab 3
""")