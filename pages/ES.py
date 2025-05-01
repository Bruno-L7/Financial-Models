import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import traceback
import plotly.express as px

def historical_es(returns, confidence_level=0.95):
    returns = np.asarray(returns).flatten()
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
    prices = np.asarray(prices).flatten()
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
        # Fetch data using yfinance
        with st.spinner("Fetching market data..."):
            data = yf.download(
                ticker, 
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty:
                st.error(f"No data found for {ticker}. Check symbol on [Yahoo Finance](https://finance.yahoo.com/)")
                st.stop()

        # Validate required columns exist
        if 'Close' not in data.columns:
            st.error("Missing 'Close' column in downloaded data")
            st.stop()

        # Calculate returns and parameters
        prices = data['Close'].values.flatten()
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
            
            # Add summary statistics
            st.subheader("Historical Summary Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean Return', 'Standard Deviation', 'Min Return', 'Max Return', 
                           'VaR (95%)', 'Expected Shortfall'],
                'Value': [
                    f"{np.mean(returns):.2%}",
                    f"{np.std(returns):.2%}",
                    f"{np.min(returns):.2%}",
                    f"{np.max(returns):.2%}",
                    f"{-np.percentile(returns, 100 * (1 - confidence_level)):.2%}",
                    f"{es_hist:.2%}"
                ]
            })
            st.table(stats_df)

        with tab2:  # 'tab2' is the second tab in your Streamlit app
            st.subheader(f"{ticker} Historical Price")
            
            # Reset the index so that the dates become a column
            data = data.reset_index()

            # Plot using the Date column for x and the Close column for y
            fig1 = px.line(x=data.index, y=data['Close'].values.flatten())

            fig1.update_layout(
                title="Historical Stock Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                hovermode="x unified"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Display a sample of the data
            st.subheader("Historical Data Sample")
            display_df = data.copy()
            display_df['Daily Return'] = display_df['Close'].pct_change() * 100
            st.dataframe(display_df.tail(10).style.format({
                'Open': '${:.2f}', 
                'High': '${:.2f}', 
                'Low': '${:.2f}', 
                'Close': '${:.2f}',
                'Daily Return': '{:.2f}%'
            }))

            # Returns distribution with VaR/ES markers (unchanged)
            st.subheader("Returns Distribution with Risk Metrics")
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                marker_color='lightblue',
                opacity=0.75,
                name="Returns Distribution"
            ))
            var_level = np.percentile(returns, 100 * (1 - confidence_level))
            fig2.add_vline(x=var_level, line_dash="dash", line_color="red", line_width=2,
                        annotation=dict(text=f"VaR ({confidence_level:.0%}): {-var_level:.2%}",
                                        font=dict(color="red"), bordercolor="red", borderwidth=1, bgcolor="white"))
            fig2.add_vline(x=-es_hist, line_dash="dash", line_color="orange", line_width=2,
                        annotation=dict(text=f"ES: {es_hist:.2%}", font=dict(color="orange"),
                                        bordercolor="orange", borderwidth=1, bgcolor="white"))
            fig2.update_layout(height=400, margin=dict(l=40, r=40, t=40, b=40),
                            xaxis_title="Daily Returns", yaxis_title="Frequency", bargap=0.1, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            # Monte Carlo simulation visualization
            st.subheader("Simulated Returns Distribution")
            
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=mc_returns,
                nbinsx=50,
                marker_color='lightgreen',
                opacity=0.75,
                name="MC Returns Distribution"
            ))
            
            var_mc = np.percentile(mc_returns, 100 * (1 - confidence_level))
            
            # Add VaR line
            fig3.add_vline(
                x=var_mc, 
                line_dash="dash", 
                line_color="red",
                line_width=2,
                annotation=dict(
                    text=f"VaR ({confidence_level:.0%}): {-var_mc:.2%}",
                    font=dict(color="red"),
                    bordercolor="red",
                    borderwidth=1,
                    bgcolor="white"
                )
            )
            
            # Add ES line
            fig3.add_vline(
                x=-es_mc, 
                line_dash="dash", 
                line_color="orange",
                line_width=2,
                annotation=dict(
                    text=f"ES: {es_mc:.2%}",
                    font=dict(color="orange"),
                    bordercolor="orange",
                    borderwidth=1,
                    bgcolor="white"
                )
            )
            
            fig3.update_layout(
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis_title="Simulated Returns",
                yaxis_title="Frequency",
                bargap=0.1,
                showlegend=False
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Show sample paths
            st.subheader("Example Simulation Paths")
            num_paths = st.slider("Number of Paths to Display", 1, 50, 5)
            
            # Generate proper simulation paths over time
            days = np.arange(time_horizon + 1)
            
            fig4 = go.Figure()
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i in range(num_paths):
                np.random.seed(42 + i)  # Different seed for each path
                path_returns = np.zeros(time_horizon + 1)
                path_returns[0] = 0  # Start at 0
                
                for t in range(1, time_horizon + 1):
                    dt = t / 252
                    log_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()
                    path_returns[t] = np.exp(log_return) - 1
                
                fig4.add_trace(go.Scatter(
                    x=days,
                    y=path_returns,
                    mode='lines',
                    name=f'Path {i+1}',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig4.update_layout(
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis_title="Days",
                yaxis_title="Return",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig4, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()