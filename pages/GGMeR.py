import streamlit as st
import numpy as np


def cagr(start_value, end_value, num_years):
    """Calculate the Compound Annual Growth Rate (CAGR)."""
    return (end_value / start_value) ** (1 / num_years) - 1

def estimate_return(dividend_1988, dividend_2022, stock_price, num_years):
    """Estimate the return based on dividends and stock price."""
    cagr_rate = cagr(dividend_1988, dividend_2022, num_years)
    return_rate = (dividend_2022 / stock_price) + cagr_rate
    return return_rate

# Input fields for user inputs
dividend_1988 = st.number_input("First Dividend ($)", value=0.4, format="%.2f")
dividend_2022 = st.number_input("Latested Dividend ($)", value=0.91, format="%.2f")
stock_price = st.number_input("Current Stock Price ($)", value=129.0, format="%.2f")
num_years = st.number_input("Number of Years", value=34, min_value=1)

# Button to calculate the estimated return
if st.button("Estimate Return"):
    estimated_return = estimate_return(dividend_1988, dividend_2022, stock_price, num_years)
    st.success(f"Estimated Return: {estimated_return:.2%}")
