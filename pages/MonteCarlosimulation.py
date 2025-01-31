import numpy as np
import streamlit as st

def monte_carlo_spread_option(S0, sigma, r, T, K, corr_matrix, num_sims, spread_weights, seed=None):

    n_assets = len(S0)
    
    # Validate inputs
    if len(sigma) != n_assets:
        raise ValueError("sigma must match the length of S0")
    if corr_matrix.shape != (n_assets, n_assets):
        raise ValueError(f"corr_matrix must be {n_assets}x{n_assets}, but got {corr_matrix.shape}")
    if len(spread_weights) != n_assets:
        raise ValueError("spread_weights must match the length of S0")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Correlation matrix must be symmetric and positive definite")

    # Generate correlated random variables
    Z = np.random.normal(size=(num_sims, n_assets))
    correlated_Z = Z @ L.T  # Shape: (num_sims, n_assets)

    # Vectorized simulation
    drift = (r - 0.5 * np.array(sigma)**2) * T
    vol_scaled = np.outer(np.array(sigma), np.sqrt(T))  # Shape: (n_assets, 1)
    
    # Explicitly reshape S0 for broadcasting
    S0_array = np.array(S0).reshape(-1, 1)  # Shape: (n_assets, 1)
    
    # Simulate terminal prices
    S_T = S0_array * np.exp(drift.reshape(-1, 1) + vol_scaled * correlated_Z.T)
    S_T = S_T.T  # Final shape: (num_sims, n_assets)

    # Calculate payoff
    spread = np.dot(S_T, spread_weights) - K
    payoff = np.maximum(spread, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    
    return option_price

def main():
    st.title("Multi-Asset Spread Option Pricing with Cholesky Decomposition (Linear Correlation)")

    # Input: Number of assets
    n_assets = st.number_input("Enter the number of assets", min_value=1, value=2, step=1)

    # Input: Initial prices and volatilities
    st.subheader("Asset Parameters")
    S0 = []
    sigma = []
    for i in range(n_assets):
        col1, col2 = st.columns(2)
        with col1:
            S0.append(st.number_input(f"Initial price for asset {i+1}", value=100.0, step=0.1))
        with col2:
            sigma.append(st.number_input(f"Volatility for asset {i+1}", value=0.2, step=0.01))

    # Input: Other parameters
    st.subheader("Option Parameters")
    r = st.number_input("Risk-free rate", value=0.05, step=0.01)
    T = st.number_input("Time to maturity (years)", value=1.0, step=0.1)
    K = st.number_input("Strike price", value=5.0, step=0.1)
    num_sims = st.number_input("Number of simulations", value=100000, step=1000)

    # Input: Spread weights
    st.subheader("Spread Weights")
    spread_weights_input = st.text_input("Spread weights (comma-separated)", value="1,-1")
    spread_weights = list(map(float, spread_weights_input.split(',')))
    if len(spread_weights) != n_assets:
        st.error(f"Expected {n_assets} spread weights, got {len(spread_weights)}")

    # Input: Correlation matrix
    st.subheader("Correlation Matrix")
    st.write(f"Enter the {n_assets}x{n_assets} correlation matrix (one row at a time):")
    corr_matrix = []
    for i in range(n_assets):
        row_input = st.text_input(f"Row {i+1} (comma-separated)", value="1.0,0.5" if i == 0 else "0.5,1.0")
        row = list(map(float, row_input.split(',')))
        if len(row) != n_assets:
            st.error(f"Row {i+1} must have {n_assets} elements.")
        corr_matrix.append(row)
    corr_matrix = np.array(corr_matrix)

    # Input: Seed option
    st.subheader("Random Seed")
    use_seed = st.radio("Do you want to use a fixed seed?", ("Yes", "No"), index=1)
    if use_seed == "Yes":
        seed = st.number_input("Enter the seed value (integer)", value=42, step=1)
    else:
        seed = None

    # Run simulation
    if st.button("Run Simulation"):
        try:
            price = monte_carlo_spread_option(S0, sigma, r, T, K, corr_matrix, num_sims, spread_weights, seed)
            st.success(f"Estimated option price: {price:.4f}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()