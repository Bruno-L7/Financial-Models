import numpy as np
import streamlit as st
from yahoo_fin.stock_info import get_data
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_var_parametric(returns, confidence_level, portfolio_value, time_horizon=1, use_mean=True):
    returns = np.asarray(returns)
    if len(returns) < 2 or np.all(returns == returns[0]):
        return np.nan
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    T = time_horizon
    mu_T = mu * T if use_mean else 0.0
    sigma_T = sigma * np.sqrt(T)
    try:
        Z = norm.ppf(1 - confidence_level)
    except:
        return np.nan
    quantile_return = mu_T + Z * sigma_T
    return portfolio_value * max(-quantile_return, 0)

def calculate_var_historical(returns, confidence_level, portfolio_value):
    returns = np.asarray(returns)
    if len(returns) < 1:
        return np.nan
    try:
        q = np.percentile(returns, 100 * (1 - confidence_level))
    except:
        return np.nan
    return portfolio_value * max(-q, 0)

# Streamlit interface
st.title("Value at Risk (VaR) Calculator")

# Sidebar inputs
st.sidebar.header("Input Parameters")
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", value=1000000.0, step=100000.0)
confidence_level = st.sidebar.slider("Confidence Level", 0.85, 0.99, 0.95)
time_horizon = st.sidebar.number_input("Time Horizon (days)", 1, 365, 1)
use_mean = st.sidebar.checkbox("Include Mean in Parametric Calculation", value=True)

# Data source selection
data_source = st.sidebar.radio("Data Source", ["Generate Random Data", "Fetch from Yahoo Finance", "Upload CSV"])

returns = None
valid_data = False

if data_source == "Generate Random Data":
    st.sidebar.subheader("Random Data Parameters")
    daily_mu = st.sidebar.number_input("Daily Mean Return", 0.0001, 0.1, 0.0001)
    daily_sigma = st.sidebar.number_input("Daily Volatility", 0.01, 0.5, 0.01)
    num_days = st.sidebar.number_input("Number of Days", 100, 10000, 1000)
    returns = np.random.normal(daily_mu, daily_sigma, num_days)
    valid_data = True

elif data_source == "Fetch from Yahoo Finance":
    st.sidebar.subheader("Yahoo Finance Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    
    if start_date >= end_date:
        st.error("End date must be after start date")
    else:
        try:
            # Fetch data using yahoo_fin
            data = get_data(
                ticker,
                start_date=start_date.strftime("%m/%d/%Y"),
                end_date=end_date.strftime("%m/%d/%Y"),
                index_as_date=True,
                interval="1d"
            )
            
            if len(data) < 5:
                st.error("Need at least 5 trading days to calculate returns")
            else:
                data['Returns'] = data['close'].pct_change().dropna()
                if len(data['Returns']) < 1:
                    st.error("Could not calculate returns from the data")
                else:
                    returns = data['Returns'].replace([np.inf, -np.inf], np.nan).dropna().values
                    if len(returns) < 10:
                        st.error("Insufficient valid return data (need ≥10 observations)")
                    else:
                        valid_data = True
                        
                        st.subheader(f"{ticker} Price History")
                        st.line_chart(data['close'])
                        
                        st.subheader("Daily Returns")
                        st.line_chart(data['Returns'])
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if 'returns' not in data.columns:
                st.error("CSV file must contain a 'returns' column")
            else:
                returns = data['returns'].replace([np.inf, -np.inf], np.nan).dropna().values
                if len(returns) < 10:
                    st.error("Need at least 10 valid returns in uploaded file")
                else:
                    valid_data = True
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if valid_data and returns is not None:
    # Calculate VaR
    var_parametric = calculate_var_parametric(
        returns, confidence_level, portfolio_value, time_horizon, use_mean
    )
    var_historical = calculate_var_historical(
        returns, confidence_level, portfolio_value
    )

    # Display results
    if not np.isnan(var_parametric) and not np.isnan(var_historical):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parametric VaR", f"${var_parametric:,.2f}",
                    f"at {confidence_level*100:.1f}% confidence")
        with col2:
            st.metric("Historical VaR", f"${var_historical:,.2f}",
                    f"at {confidence_level*100:.1f}% confidence")

        # Plot returns distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(returns, bins=50, density=True, alpha=0.6, color='blue', label='Historical Returns')
        
        # Plot normal distribution if valid parameters
        if data_source != "Upload CSV":
            try:
                x = np.linspace(returns.min(), returns.max(), 100)
                mu = np.mean(returns)
                sigma = np.std(returns)
                ax.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Distribution')
            except:
                pass
        
        # Plot VaR lines
        try:
            ax.axvline(-var_parametric/portfolio_value, color='red', linestyle='--', 
                    label=f'Parametric VaR ({confidence_level*100:.1f}%)')
            ax.axvline(-var_historical/portfolio_value, color='green', linestyle='--',
                    label=f'Historical VaR ({confidence_level*100:.1f}%)')
        except:
            pass
        
        ax.set_title("Returns Distribution with VaR Thresholds")
        ax.set_xlabel("Daily Returns")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Graph Interpretation")

        # Calculate stats for interpretation
        returns_pct = returns * 100  # Convert to percentage
        stable_threshold = 1.0  # Define stability threshold as ±1%
        stable_pct = np.mean((returns_pct > -stable_threshold) & (returns_pct < stable_threshold)) * 100

        # 1. Stability analysis
        interpretation = []
        interpretation.append("### Key Observations")

        if stable_pct > 60:
            interpretation.append(f"📊 **Majority around 0**: {stable_pct:.1f}% of daily returns are between ±{stable_threshold}%. "
                               "This suggests the asset was relatively stable during this period.")
        else:
            interpretation.append(f"📊 **Dispersed Returns**: Only {stable_pct:.1f}% of returns are between ±{stable_threshold}%. "
                               "This indicates significant price movements during the period.")

        # 2. VaR comparison
        var_parametric_pct = abs(var_parametric/portfolio_value)*100
        var_historical_pct = abs(var_historical/portfolio_value)*100

        interpretation.append(f"🔴 **Parametric VaR Line (Red)**: Predicts a worst-case loss of {var_parametric_pct:.1f}% "
                            f"({confidence_level*100:.0f}% confidence) based on normal distribution assumptions.")

        interpretation.append(f"🟢 **Historical VaR Line (Green)**: Shows actual worst {100*(1-confidence_level):.0f}% of days had "
                            f"losses exceeding {var_historical_pct:.1f}% based on historical data.")

        # 3. Risk assessment
        var_diff = var_historical_pct - var_parametric_pct
        if var_diff > 1:
            interpretation.append("⚠️ **Risk Warning**: The parametric method underestimates risk compared to historical data. "
                               "This suggests:")
            interpretation.append("   - Historical data has 'fat tails' (more extreme losses than normal distribution predicts)")
            interpretation.append("   - Actual risk might be higher than parametric model suggests")
        elif abs(var_diff) <= 1:
            interpretation.append("✅ **Model Alignment**: Parametric and historical VaR are similar. This suggests:")
            interpretation.append("   - Return distribution is close to normal")
            interpretation.append("   - Parametric model reasonably captures risk")
        else:
            interpretation.append("📉 **Conservative Model**: Parametric VaR shows higher risk than historical VaR. This suggests:")
            interpretation.append("   - Recent volatility may be lower than historical patterns")
            interpretation.append("   - Normal distribution assumptions may be too pessimistic")

        # Display interpretation
        for paragraph in interpretation:
            st.markdown(paragraph)

        # Add statistical disclaimer
        st.markdown("""
        ---
        **Statistical Notes**:
        - Analysis based on {} trading days
        - Stability threshold: ±{:.1f}% daily returns
        - VaR calculations assume independent daily returns
        """.format(len(returns), stable_threshold))

    else:
        st.error("Could not calculate VaR - check input parameters and ensure sufficient market data")

elif returns is not None:
    st.error("Invalid data for VaR calculation")

else:
    st.warning("Please select a data source and provide required inputs")