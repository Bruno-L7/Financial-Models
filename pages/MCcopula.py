import numpy as np
from scipy.stats import multivariate_normal, norm, t
import streamlit as st

def generate_copula_samples(copula_type, n_assets, n_samples, **kwargs):
    if copula_type == 'Gaussian':
        rho = kwargs['rho']
        samples = multivariate_normal(mean=np.zeros(n_assets), cov=rho).rvs(n_samples)
        uniforms = norm.cdf(samples)
    elif copula_type == 't':
        rho = kwargs['rho']
        nu = kwargs['nu']
        cov_matrix = np.array(rho)
        d = len(cov_matrix)
        g = np.random.chisquare(nu, n_samples) / nu
        z = multivariate_normal(mean=np.zeros(d), cov=cov_matrix).rvs(n_samples)
        t_samples = z / np.sqrt(g)[:, None]
        uniforms = t.cdf(t_samples, df=nu)
    elif copula_type == 'Clayton':
        if n_assets != 2:
            raise ValueError("Clayton copula is only supported for two assets.")
        theta = kwargs['theta']
        v = np.random.gamma(shape=1/theta, scale=1, size=n_samples)
        u = np.random.uniform(size=(n_samples, 2))
        u1 = u[:, 0]
        u2 = u[:, 1]
        x1 = (1 - theta * np.log(u1) / v) ** (-1/theta)
        x2 = (1 - theta * np.log(u2) / v) ** (-1/theta)
        uniforms = np.column_stack((x1, x2))
    elif copula_type == 'Gumbel':
        if n_assets != 2:
            raise ValueError("Gumbel copula is only supported for two assets.")
        theta = kwargs['theta']
        u = np.random.uniform(size=(n_samples, 2))
        t_val = (-np.log(u[:, 0])) ** theta
        w = np.random.exponential(size=n_samples)
        x = (t_val + w) ** (1/theta) / (t_val ** (1/theta) + (-np.log(u[:, 1])) ** (theta/(theta - 1))) ** ((theta - 1)/theta)
        v1 = np.exp(-t_val / x)
        v2 = np.exp(-w / x)
        uniforms = np.column_stack((v1, v2))
    else:
        raise ValueError(f"Copula type {copula_type} not supported.")
    return uniforms

def monte_carlo_spread_option(S0, sigma, r, T, K, n_sims, n_steps, copula_type, **copula_params):
    n_assets = len(S0)
    dt = T / n_steps
    discount = np.exp(-r * T)
    
    total_samples = n_sims * n_steps
    copula_uniforms = generate_copula_samples(copula_type, n_assets, total_samples, **copula_params)
    normals = norm.ppf(copula_uniforms)
    normals = normals.reshape(n_sims, n_steps, n_assets)
    
    S = np.zeros((n_sims, n_steps + 1, n_assets))
    S[:, 0, :] = S0
    
    for t in range(n_steps):
        z = normals[:, t, :]
        for i in range(n_assets):
            drift = (r - 0.5 * sigma[i] ** 2) * dt
            diffusion = sigma[i] * np.sqrt(dt) * z[:, i]
            S[:, t+1, i] = S[:, t, i] * np.exp(drift + diffusion)
    
    if n_assets == 2:
        spread = S[:, -1, 0] - S[:, -1, 1]
    else:
        spread = S[:, -1, 0] - np.sum(S[:, -1, 1:], axis=1)
    payoffs = np.maximum(spread - K, 0)
    option_price = discount * np.mean(payoffs)
    return option_price

def main():
    st.title("Multi-Asset Spread Option Pricing using Monte Carlo Simulation with different Copulas")

    # Input parameters in the main page
    st.header("Asset Parameters")
    col1, col2 = st.columns(2)
    with col1:
        S0_1 = st.number_input("Initial price for Asset 1", value=100.0, min_value=0.01)
        sigma_1 = st.number_input("Volatility for Asset 1 (e.g., 0.2 for 20%)", value=0.2, min_value=0.01)
    with col2:
        S0_2 = st.number_input("Initial price for Asset 2", value=100.0, min_value=0.01)
        sigma_2 = st.number_input("Volatility for Asset 2 (e.g., 0.3 for 30%)", value=0.3, min_value=0.01)
    S0 = [S0_1, S0_2]
    sigma = [sigma_1, sigma_2]

    st.header("General Parameters")
    r = st.number_input("Risk-free rate (e.g., 0.05 for 5%)", value=0.05, min_value=0.0)
    T = st.number_input("Time to maturity (years)", value=1.0, min_value=0.01)
    K = st.number_input("Strike price", value=5.0, min_value=0.0)
    n_sims = st.number_input("Number of simulations", value=10000, min_value=1000, step=1000)
    n_steps = st.number_input("Number of time steps", value=100, min_value=10, step=10)

    st.header("Copula Selection")
    copula_type = st.selectbox("Copula type", ["Gaussian", "t", "Clayton", "Gumbel"])

    # Copula-specific parameters
    if copula_type == "Gaussian":
        rho = st.number_input("Correlation coefficient (rho)", value=0.5, min_value=-1.0, max_value=1.0)
        copula_params = {'rho': [[1.0, rho], [rho, 1.0]]}
    elif copula_type == "t":
        rho = st.number_input("Correlation coefficient (rho)", value=0.5, min_value=-1.0, max_value=1.0)
        nu = st.number_input("Degrees of freedom (nu > 2)", value=5.0, min_value=2.1)
        copula_params = {'rho': [[1.0, rho], [rho, 1.0]], 'nu': nu}
    elif copula_type == "Clayton":
        theta = st.number_input("Theta parameter (theta > 0)", value=2.0, min_value=0.01)
        copula_params = {'theta': theta}
    elif copula_type == "Gumbel":
        theta = st.number_input("Theta parameter (theta >= 1)", value=2.0, min_value=1.0)
        copula_params = {'theta': theta}

    # Run simulation when the user clicks the button
    if st.button("Calculate Option Price"):
        with st.spinner("Running Monte Carlo simulation..."):
            price = monte_carlo_spread_option(
                S0, sigma, r, T, K, n_sims, n_steps,
                copula_type, **copula_params
            )
        st.success(f"Spread option price using {copula_type} copula: **{price:.4f}**")

if __name__ == "__main__":
    main()