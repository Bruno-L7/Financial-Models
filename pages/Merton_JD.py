import streamlit as st
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def merton_american_lsmc(S0, K, T, r, sigma, lambda_j, mu_j, sigma_j, 
                        num_simulations, num_steps, option_type='call', degree=2):

    # Initialize parameters and arrays
    np.random.seed(42)
    dt = T / num_steps
    m = np.exp(mu_j + 0.5 * sigma_j**2) - 1  # Jump compensator
    discount_factors = np.exp(-r * dt * np.arange(1, num_steps + 1))
    
    # Simulate all paths with jump-diffusion
    stock_paths = np.zeros((num_simulations, num_steps + 1))
    stock_paths[:, 0] = S0
    
    for t in range(num_steps):
        # Diffusion component
        Z = np.random.normal(0, 1, num_simulations)
        drift = (r - 0.5 * sigma**2 - lambda_j * m) * dt
        diffusion = drift + sigma * np.sqrt(dt) * Z
        
        # Jump component
        jump_counts = np.random.poisson(lambda_j * dt, num_simulations)
        jump_sizes = np.zeros(num_simulations)
        mask = jump_counts > 0
        
        if np.any(mask):
            # Vectorized jump size calculation
            jump_sizes[mask] = np.random.normal(
                mu_j * jump_counts[mask],
                sigma_j * np.sqrt(jump_counts[mask])
            )
        
        # Update stock paths
        stock_paths[:, t + 1] = stock_paths[:, t] * np.exp(diffusion + jump_sizes)
    
    # Initialize cash flow matrix
    cash_flows = np.zeros_like(stock_paths)
    
    # Set terminal payoff
    if option_type == 'call':
        cash_flows[:, -1] = np.maximum(stock_paths[:, -1] - K, 0)
    else:
        cash_flows[:, -1] = np.maximum(K - stock_paths[:, -1], 0)
    
    # Backward induction with vectorized operations
    for t in range(num_steps - 1, 0, -1):
        # Identify in-the-money paths
        if option_type == 'call':
            in_the_money = stock_paths[:, t] > K
        else:
            in_the_money = stock_paths[:, t] < K
            
        if not np.any(in_the_money):
            continue
            
        X = stock_paths[in_the_money, t].reshape(-1, 1)
        future_cash_flows = cash_flows[in_the_money, t + 1] * np.exp(-r * dt)
        
        # Fit continuation value regression
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, future_cash_flows)
        continuation = model.predict(X_poly)
        
        # Calculate immediate exercise value
        if option_type == 'put':
            exercise = np.maximum(K - X[:, 0], 0)
        else:
            exercise = np.maximum(X[:, 0] - K, 0)
            
        # Update cash flows where exercise is optimal
        exercise_mask = exercise > continuation
        cash_flows[in_the_money, t] = np.where(exercise_mask, exercise, 0)
        cash_flows[in_the_money, t+1:] = 0
    
    # Calculate discounted payoff
    discounted_payoffs = np.sum(cash_flows[:, 1:] * discount_factors, axis=1)
    return np.mean(discounted_payoffs)

def main():
    st.title("Merton's Jump-Diffusion Option Pricing for American Options")

    # Sidebar controls
    with st.sidebar:
        st.header("Model Parameters")
        option_style = st.selectbox("Call/Put", ["call", "put"])
        
        st.subheader("Financial Parameters")
        S0 = st.number_input("Spot Price (S₀)", min_value=1.0, value=100.0)
        K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0)
        T = st.slider("Time to Maturity (years)", min_value=0.1, max_value=5.0, value=1.0)
        r = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=5.0) / 100
        
        st.subheader("Diffusion Parameters")
        sigma = st.slider("Volatility (%)", min_value=1.0, max_value=100.0, value=20.0) / 100
        
        st.subheader("Jump Parameters")
        lambda_j = st.slider("Jump Intensity (λ)", min_value=0.0, max_value=5.0, value=0.5)
        mu_j = st.slider("Mean Jump Size (%)", min_value=-50.0, max_value=50.0, value=-10.0) / 100
        sigma_j = st.slider("Jump Volatility (%)", min_value=1.0, max_value=100.0, value=30.0) / 100
        
        st.subheader("Simulation Parameters")
        num_simulations = st.selectbox("Number of Simulations", [1000, 5000, 10000, 25000], index=2)
        num_steps = st.selectbox("Number of Time Steps", [30, 60, 125, 252], index=3)

    # Pricing button
    if st.button("Calculate American Option Price"):
        with st.spinner("Running Monte Carlo Simulations..."):
            try:
                # Calculate American option price using LSMC
                price = merton_american_lsmc(S0, K, T, r, sigma, lambda_j, mu_j, sigma_j,
                                           num_simulations, num_steps, option_style)
                
                st.success(f"American {option_style.capitalize()} Option Price: ${price:.4f}")
                
            except Exception as e:
                st.error(f"Error in calculation: {str(e)}")

    # Theory section
    with st.expander("Model Explanation"):
        st.markdown("""
        **Merton's Jump-Diffusion Model** extends the Black-Scholes model by incorporating:
        - Geometric Brownian Motion (continuous price movements)
        - Poisson jump process (sudden price jumps)
        
        **American Options** allow early exercise, and their pricing uses:
        - **Least Squares Monte Carlo (LSMC)**: Estimates optimal exercise strategy by regressing future cash flows against current stock prices.
        
        Key parameters:
        - λ (Jump intensity): Expected number of jumps per year
        - μ_j (Mean jump size): Average log-return of jumps
        - σ_j (Jump volatility): Volatility of jump sizes
        """)

if __name__ == "__main__":
    main()