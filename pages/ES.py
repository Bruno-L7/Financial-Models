import streamlit as st
import numpy as np
import pandas as pd
import yahoo_fin.stock_info as yf
import plotly.express as px

def historical_es(returns, confidence_level=0.95):
    returns = np.asarray(returns)
    var = np.percentile(returns, 100 * (1 - confidence_level))
    tail_losses = returns[returns <= var]
    return -tail_losses.mean() if len(tail_losses) > 0 else 0.0

def monte_carlo_es(mu, sigma, time_horizon=1, num_simulations=10000, confidence_level=0.95):
    np.random.seed(42)
    dt = time_horizon / 252
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=num_simulations)
    simple_returns = np.exp(log_returns) - 1
    var = np.percentile(simple_returns, 100 * (1 - confidence_level))
    tail_losses = simple_returns[simple_returns <= var]
    return -tail_losses.mean() if len(tail_losses) > 0 else 0.0, simple_returns

def estimate_gbm_parameters(prices):
    log_returns = np.log(prices[1:]/prices[:-1])
    dt = 1/252
    mu = (np.mean(log_returns) + 0.5 * np.var(log_returns)) / dt
    sigma = np.sqrt(np.var(log_returns) / dt)
    return mu, sigma

def main():
    st.set_page_config(page_title="ES Calculator", layout="wide")
    st.title("Expected Shortfall Calculator")
    st.markdown("### Risk Analysis with Historical Data and Monte Carlo Simulations")

    # Sidebar inputs
    with st.sidebar:
        st.header("Input Parameters")
        ticker = st.text_input("Stock Ticker", "AAPL")
        start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95)
        time_horizon = st.slider("Time Horizon (days)", 1, 30, 1)
        num_simulations = st.selectbox("Number of Simulations", [1000, 10000, 50000], index=1)

    try:
        # Fetch data
        with st.spinner("Fetching market data..."):
            data = yf.get_data(ticker, start_date=start_date, end_date=end_date)
            if data.empty:
                raise ValueError("No data retrieved - check ticker and dates")
        
        # Calculate returns and parameters
        prices = data['adjclose'].values
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        mu, sigma = estimate_gbm_parameters(prices)
        
        # Calculate ES values
        es_hist = historical_es(returns, confidence_level)
        es_mc, mc_returns = monte_carlo_es(mu, sigma, time_horizon, num_simulations, confidence_level)

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Results", "Historical Analysis", "Monte Carlo Simulation"])

        with tab1:
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Historical ES", f"{es_hist:.2%}", 
                         help="Average loss beyond VaR using historical returns")
            with col2:
                st.metric("Monte Carlo ES", f"{es_mc:.2%}",
                         help="Forward-looking ES using GBM simulations")
            
            st.subheader("GBM Parameters")
            st.write(f"Drift (μ): {mu:.4f}")
            st.write(f"Volatility (σ): {sigma:.4f}")

        with tab2:
            # Historical price and returns visualizations
            fig1 = px.line(data, x=data.index, y='adjclose', 
                          title=f"{ticker} Historical Prices")
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)

            # Returns distribution with VaR/ES markers
            fig2 = px.histogram(x=returns, nbins=100, 
                               title="Returns Distribution with Risk Metrics",
                               labels={'x': 'Daily Returns'})
            var_level = np.percentile(returns, 100 * (1 - confidence_level))
            fig2.add_vline(x=var_level, line_dash="dash", line_color="red",
                          annotation_text=f"VaR ({confidence_level:.0%})")
            fig2.add_vline(x=-es_hist, line_dash="dash", line_color="orange",
                          annotation_text=f"Historical ES")
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            # Monte Carlo simulation visualization
            st.subheader("Simulated Returns Distribution")
            fig3 = px.histogram(x=mc_returns, nbins=100,
                               title="Monte Carlo Returns Distribution",
                               labels={'x': 'Simulated Returns'})
            var_mc = np.percentile(mc_returns, 100 * (1 - confidence_level))
            fig3.add_vline(x=var_mc, line_dash="dash", line_color="red",
                          annotation_text=f"VaR ({confidence_level:.0%})")
            fig3.add_vline(x=-es_mc, line_dash="dash", line_color="orange",
                          annotation_text=f"Monte Carlo ES")
            st.plotly_chart(fig3, use_container_width=True)

            # Show sample paths
            st.subheader("Example Simulation Paths")
            num_paths = st.slider("Number of Paths to Display", 1, 50, 5)
            paths = np.random.choice(mc_returns, size=num_paths)
            fig4 = px.line(pd.DataFrame(paths).T, 
                          title="Sample Monte Carlo Paths",
                          labels={'index': 'Steps', 'value': 'Return'})
            st.plotly_chart(fig4, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()