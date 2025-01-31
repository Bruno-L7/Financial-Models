import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """Calculate the Black-Scholes call option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate the Black-Scholes put option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return put_price

# Streamlit application
st.title("Black-Scholes Option Pricing Model")

# Sidebar for input fields
with st.sidebar:
    S = st.number_input("Stock Price (S)", min_value=0.0, value=100.0)
    K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0)
    T = st.number_input("Time to Expiry (T in years)", min_value=0.0, value=1.0)
    r = st.number_input("Risk-Free Interest Rate (r)", min_value=0.0, value=0.05)
    sigma = st.number_input("Volatility (σ)", min_value=0.0, value=0.2)

# Sidebar for heatmap inputs
st.sidebar.header("Heatmap Parameters")
min_spot_price = st.sidebar.number_input("Min Spot Price", min_value=0.0, value=10.0)
max_spot_price = st.sidebar.number_input("Max Spot Price", min_value=0.0, value=150.0)
min_volatility = st.sidebar.number_input("Min Volatility", min_value=0.0, value=0.1)
max_volatility = st.sidebar.number_input("Max Volatility", min_value=0.0, value=0.5)
num_points = st.sidebar.number_input("Number of Points", min_value=1, value=10)

# Calculate option prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

# Display option prices in colored rectangles
st.markdown(f"""
<div style='display: flex; justify-content: center; gap: 20px;'>
    <div style='background-color: #90EE90; padding: 8px; border-radius: 5px; color: black; font-size: 20px; width: 500px; text-align: center;'>
        Call Option Price:<br><strong>${call_price:.2f}</strong>
    </div>
    <div style='background-color: #FFCCCB; padding: 8px; border-radius: 5px; color: black; font-size: 20px; width: 500px; text-align: center;'>
        Put Option Price:<br><strong>${put_price:.2f}</strong>
    </div>
</div>
""", unsafe_allow_html=True)

st.title("Options Price Heatmap")

# Generate heatmap data
spot_prices = np.linspace(min_spot_price, max_spot_price, num_points)
volatilities = np.linspace(min_volatility, max_volatility, num_points)

call_prices = np.zeros((num_points, num_points))
put_prices = np.zeros((num_points, num_points))

for i, S in enumerate(spot_prices):
    for j, sigma in enumerate(volatilities):
        call_prices[j, i] = black_scholes_call(S, K, T, r, sigma)  # Transpose the assignment
        put_prices[j, i] = black_scholes_put(S, K, T, r, sigma)    # Transpose the assignment

# Use Streamlit columns to align the heatmaps side by side
col1, col2 = st.columns(2)

with col1:
    plt.figure(figsize=(13, 10))  # Reduced heatmap size
    ax = sns.heatmap(call_prices, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2), cmap='viridis', annot=True, fmt=".2f")
    plt.title(f'Call Option Price Heatmap\n(K = {K:.2f})', fontsize=14)
    plt.xlabel('Stock Price (S)', fontsize=12)
    plt.ylabel('Volatility (σ)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt)

with col2:
    plt.figure(figsize=(13, 10))  # Reduced heatmap size
    ax = sns.heatmap(put_prices, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2), cmap='viridis', annot=True, fmt=".2f")
    plt.title(f'Put Option Price Heatmap\n(K = {K:.2f})', fontsize=14)
    plt.xlabel('Stock Price (S)', fontsize=12)
    plt.ylabel('Volatility (σ)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt)