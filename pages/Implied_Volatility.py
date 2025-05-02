import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go

# Black-Scholes Implied Volatility Calculation
def black_scholes_implied_volatility(S, K, T, r, q, market_price, option_type='call'):
    if T <= 0:
        return np.nan  # Option is expired
    
    # Precompute discounted terms
    S_disc = S * np.exp(-q * T)
    K_disc = K * np.exp(-r * T)
    
    # Calculate intrinsic value and theoretical price bounds
    if option_type == 'call':
        intrinsic = max(S_disc - K_disc, 0)
        upper_bound = S_disc  # Call price approaches S_disc as σ → ∞
    else:
        intrinsic = max(K_disc - S_disc, 0)
        upper_bound = K_disc  # Put price approaches K_disc as σ → ∞
    
    # Check if market price is outside valid bounds (with a small tolerance)
    eps = 1e-8
    if market_price < intrinsic - eps or market_price > upper_bound + eps:
        return np.nan
    
    # If market price equals intrinsic, return 0 volatility
    if abs(market_price - intrinsic) < eps:
        return 0.0
    
    # Handle near-zero volatility cases to avoid division by zero
    if abs(S - K) < eps and T < eps:
        return 0.0

    def bs_price(sigma):
        # Handle extreme volatilities to avoid numerical errors
        if sigma > 1000:  # Treat σ > 1000% as "infinite"
            return upper_bound
        if sigma < 1e-6:  # Avoid division-by-zero in d1/d2
            return intrinsic
        
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        if option_type == 'call':
            price = S_disc * norm.cdf(d1) - K_disc * norm.cdf(d2)
        else:
            price = K_disc * norm.cdf(-d2) - S_disc * norm.cdf(-d1)
        return price

    # Dynamically adjust the upper volatility bound
    low, high = 1e-6, 5.0  # Initial bounds
    max_iter = 100
    
    # Ensure the root is bracketed: bs_price(low) < target < bs_price(high)
    for _ in range(max_iter):
        price_high = bs_price(high)
        if price_high < market_price:
            high *= 2.0  # Increase upper bound until BS price exceeds market price
        else:
            break
    
    # Check if bracketing succeeded
    if bs_price(high) < market_price:
        return np.nan  # No solution exists
    
    try:
        implied_vol = brentq(
            lambda sigma: bs_price(sigma) - market_price,
            a=low, b=high, xtol=1e-6, maxiter=1000
        )
        return implied_vol
    except ValueError:
        return np.nan  # Root-finding failed

# Streamlit App
st.title("Implied Volatility Surface")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")
spot_price = st.sidebar.number_input("Spot Price", value=100.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=5.0) / 100
dividend_yield = st.sidebar.number_input("Dividend Yield (%)", value=2.0) / 100
min_strike_pct = st.sidebar.number_input("Minimum Strike Price (% of Spot Price)", value=80.0) * 0.01  # 80% → 0.8
max_strike_pct = st.sidebar.number_input("Maximum Strike Price (% of Spot Price)", value=120.0) * 0.01  # 120% → 1.2
min_maturity = st.sidebar.number_input("Minimum Time to Maturity (years)", value=0.1, min_value=0.01)
max_maturity = st.sidebar.number_input("Maximum Time to Maturity (years)", value=2.0, min_value=0.01)
market_price = st.sidebar.number_input("Market Price for Options", value=10.0)  # Adjusted to a reasonable value
y_axis_choice = st.sidebar.radio("Y-Axis", ["Strike Price", "Moneyness (K/S)"])

# Generate data for the surface plot
strike_prices = np.linspace(spot_price * min_strike_pct, spot_price * max_strike_pct, 20)
maturities = np.linspace(min_maturity, max_maturity, 20)
implied_vols = np.zeros((len(strike_prices), len(maturities)))

for i, K in enumerate(strike_prices):
    for j, T in enumerate(maturities):
        iv = black_scholes_implied_volatility(
            S=spot_price, K=K, T=T, r=risk_free_rate, q=dividend_yield,
            market_price=market_price, option_type='call'
        )
        implied_vols[i, j] = iv if not np.isnan(iv) else 1e-6  # Replace NaN with a small positive number

# Debugging: Print shapes and data ranges
st.write("Shape of implied_vols:", implied_vols.shape)
st.write("Shape of maturities:", maturities.shape)
st.write("Min implied volatility:", np.min(implied_vols))
st.write("Max implied volatility:", np.max(implied_vols))

# Prepare data for the 3D plot
if y_axis_choice == "Strike Price":
    y_values = strike_prices  # Use strike prices directly
    y_label = "Strike Price"
else:
    y_values = strike_prices / spot_price  # Calculate moneyness (K/S)
    y_label = "Moneyness (K/S)"

# Debugging: Print y_values
st.write("Y Values:", y_values)

# Create the 3D surface plot
fig = go.Figure(data=[go.Surface(
    z=implied_vols,
    x=maturities,
    y=y_values,
    colorscale='Viridis',
    colorbar=dict(title="Implied Volatility")
)])

# Update layout for better visualization
fig.update_layout(
    title="Implied Volatility Surface",
    scene=dict(
        xaxis_title="Time to Maturity (years)",
        yaxis_title=y_label,
        zaxis_title="Implied Volatility"
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

# Display the plot in Streamlit
st.plotly_chart(fig)