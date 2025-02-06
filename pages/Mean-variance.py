import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from yahoo_fin import stock_info as si
from pypfopt import EfficientFrontier, risk_models, expected_returns


def main():
    st.title("Robust Portfolio Optimizer")
    st.markdown("## Proper Weight Handling Implementation")

    # Sidebar inputs
    st.sidebar.header("Parameters")
    start_date = st.sidebar.date_input("Start date", pd.to_datetime("2018-01-01"))
    end_date = st.sidebar.date_input("End date", pd.to_datetime("2023-01-01"))
    default_tickers = "AAPL, MSFT, AMZN, GOOG, TSLA"
    tickers = st.sidebar.text_input("Tickers (comma-separated)", default_tickers).upper().split(', ')
    min_alloc = st.sidebar.slider("Min Allocation (%)", 0.1, 10.0, 1.0) / 100

    # Initialize session state
    if 'weights' not in st.session_state:
        st.session_state.weights = {ticker: 1/len(tickers) for ticker in tickers}

    @st.cache_data
    def load_data(tickers, start, end):
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = si.get_data(ticker, start_date=start, end_date=end)['adjclose']
            except Exception as e:
                st.error(f"Failed to load data for {ticker}: {str(e)}")
        return pd.DataFrame(data)

    try:
        data = load_data(tickers, start_date, end_date)
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)

        # Convert weights to proper format
        def convert_weights(w):
            if isinstance(w, np.ndarray):
                return {ticker: w[i] for i, ticker in enumerate(tickers)}
            return w

        # Optimization
        if st.sidebar.button("Optimize Portfolio"):
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(min_alloc, 1))
                ef.max_sharpe()
                raw_weights = ef.weights
                cleaned_weights = {ticker: max(w, min_alloc) 
                                   for ticker, w in zip(tickers, raw_weights)}
                total = sum(cleaned_weights.values())
                st.session_state.weights = {k: v/total for k, v in cleaned_weights.items()}
                st.sidebar.success("Optimization successful!")
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")

        # Manual adjustments
        st.sidebar.header("Weight Adjustments")
        new_weights = {}
        total = 0.0

        for ticker in tickers:
            current = st.session_state.weights.get(ticker, min_alloc)
            new = st.sidebar.slider(
                f"{ticker} (%)",
                min_alloc*100,
                100.0,
                float(current*100),
                0.1
            ) / 100
            new_weights[ticker] = new
            total += new

        # Normalization
        normalized = {k: v/total for k, v in new_weights.items()}
        st.session_state.weights = normalized

        # Performance calculation
        def get_performance(weights):
            w_array = np.array([weights[t] for t in tickers])
            ret = mu.dot(w_array)
            vol = np.sqrt(w_array.T @ S @ w_array)
            return ret, vol

        # Efficient frontier calculation
        ef = EfficientFrontier(mu, S, weight_bounds=(min_alloc, 1))
        fig = go.Figure()

        # Generate frontier
        rets = np.linspace(mu.min(), mu.max(), 50)
        frontier = []
        for r in rets:
            try:
                ef.efficient_return(r)
                w = np.array([ef.weights[i] for i in range(len(tickers))])
                vol = np.sqrt(w.T @ S @ w)
                frontier.append((vol, r))
            except:
                continue

        # Create plot
        fig.add_trace(go.Scatter(
            x=[f[0] for f in frontier],
            y=[f[1] for f in frontier],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue')
        ))

        # Add optimized point
        opt_ret, opt_vol = get_performance(st.session_state.weights)
        fig.add_trace(go.Scatter(
            x=[opt_vol],
            y=[opt_ret],
            mode='markers',
            marker=dict(color='green', size=12),
            name='Optimized'
        ))

        # Current portfolio
        user_ret, user_vol = get_performance(normalized)
        fig.add_trace(go.Scatter(
            x=[user_vol],
            y=[user_ret],
            mode='markers',
            marker=dict(color='red', size=12),
            name='Current'
        ))

        # Display
        st.subheader("Allocation")
        weights_df = pd.DataFrame.from_dict(normalized, orient='index', columns=['Weight'])
        st.dataframe(weights_df.style.format("{:.2%}"))

        st.subheader("Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Return", f"{user_ret:.2%}")
        col2.metric("Volatility", f"{user_vol:.2%}")
        col3.metric("Sharpe", f"{(user_ret-0.02)/user_vol:.2f}")

        st.subheader("Efficient Frontier")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
