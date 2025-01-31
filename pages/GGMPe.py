import streamlit as st
import numpy as np

st.title("Gordon Growth Model Estimated Return")

def cagr(start_value, end_value, num_years):
    """Calculate the Compound Annual Growth Rate (CAGR)."""
    return (end_value / start_value) ** (1 / num_years) - 1

def estimate_price(dividend_1988, dividend_2022, return_market, num_years):
    """Estimate the price using the Gordon Growth Model."""
    cagr_rate = cagr(dividend_1988, dividend_2022, num_years)
    price = dividend_2022 / (return_market - cagr_rate)
    return price

# Streamlit application
st.title("Gordon Growth Model Price Estimator")

# Input fields for user inputs
dividend_1988 = st.number_input("Dividend in 1988 ($)", value=0.4, format="%.2f")
dividend_2022 = st.number_input("Dividend in 2022 ($)", value=0.91, format="%.2f")
return_market = st.number_input("Expected Market Return (as a decimal)", value=0.07, format="%.2f")
num_years = st.number_input("Number of Years", value=34, min_value=1)

# Button to calculate the estimated price
if st.button("Estimate Price"):
    estimated_price = estimate_price(dividend_1988, dividend_2022, return_market, num_years)
    st.success(f"Estimated Price: ${estimated_price:.2f}")
