import streamlit as st
import numpy as np

# Function to calculate the Compound Annual Growth Rate (CAGR)
def cagr(start_value, end_value, num_years):
    return (end_value / start_value) ** (1 / num_years) - 1

# Function to estimate the return based on dividends and stock price
def estimate_return(dividend_1988, dividend_2022, stock_price, num_years):
    cagr_rate = cagr(dividend_1988, dividend_2022, num_years)
    return_rate = (dividend_2022 / stock_price) + cagr_rate
    return return_rate

# Function to estimate the price using the Gordon Growth Model
def estimate_price(dividend_1988, dividend_2022, return_market, num_years):
    cagr_rate = cagr(dividend_1988, dividend_2022, num_years)
    price = dividend_2022 / (return_market - cagr_rate)
    return price

# Streamlit application
st.title("Gordon Growth Model Calculator")

# User selects between estimating return or price
calculation_type = st.radio("Select Calculation Type", ("Estimated Return", "Price Estimator"))

# Common input fields
dividend_1988 = st.number_input("First Dividend ($)", value=0.4, format="%.2f")
dividend_2022 = st.number_input("Latest Dividend ($)", value=0.91, format="%.2f")
num_years = st.number_input("Number of Years", value=34, min_value=1)

if calculation_type == "Estimated Return":
    # Input field for current stock price
    stock_price = st.number_input("Current Stock Price ($)", value=129.0, format="%.2f")
    
    # Button to calculate the estimated return
    if st.button("Estimate Return"):
        estimated_return = estimate_return(dividend_1988, dividend_2022, stock_price, num_years)
        st.success(f"Estimated Return: {estimated_return:.2%}")

elif calculation_type == "Price Estimator":
    # Input field for expected market return
    return_market = st.number_input("Expected Market Return (as a decimal)", value=0.07, format="%.2f")
    
    # Button to calculate the estimated price
    if st.button("Estimate Price"):
        estimated_price = estimate_price(dividend_1988, dividend_2022, return_market, num_years)
        st.success(f"Estimated Price: ${estimated_price:.2f}")
