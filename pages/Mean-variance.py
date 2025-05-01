import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns


def main():
    st.title("Robust Portfolio Optimizer")
    st.markdown("## Proper Weight Handling Implementation")

    # Sidebar inputs
    st.sidebar.header("Parameters")
    start_date = st.sidebar.date_input("Start date", pd.to_datetime("2018-01-01"))
    end_date = st.sidebar.date_input("End date", pd.to_datetime("2023-01-01"))
    default_tickers = "AAPL,MSFT,AMZN,GOOG,TSLA"
    tickers_input = st.sidebar.text_input("Tickers (comma-separated)", default_tickers)
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    min_alloc = st.sidebar.slider("Min Allocation (%)", 0.1, 10.0, 1.0) / 100

    # Initialize session state variables if not already set
    if 'optimized_weights' not in st.session_state:
        st.session_state.optimized_weights = {}
    if 'manual_weights' not in st.session_state:
        st.session_state.manual_weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    @st.cache_data
    def load_data(tickers, start, end):
        data = {}
        for ticker in tickers:
            try:
                ticker_data = yf.download(ticker, start=start, end=end)
                if not ticker_data.empty:
                    data[ticker] = ticker_data['Close']
                else:
                    st.warning(f"No data available for {ticker}")
            except Exception as e:
                st.warning(f"Failed to load data for {ticker}: {str(e)}")
        
        if not data:
            st.error("No valid data found for any of the tickers")
            return pd.DataFrame(index=pd.date_range(start=start, end=end))
        
        df = pd.concat(data.values(), axis=1, keys=data.keys())
        df_clean = df.dropna()
        if len(df_clean) < 10:
            st.error(f"Not enough valid data points after removing NaN values. Only {len(df_clean)} rows remain.")
            return pd.DataFrame(index=pd.date_range(start=start, end=end))
        
        return df_clean

    try:
        data = load_data(tickers, start_date, end_date)
        if data.empty:
            st.error("No data available for the selected tickers and date range.")
            return

        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)

        # Optimization button and process
        if st.sidebar.button("Optimize Portfolio"):
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(min_alloc, 1))
                ef.max_sharpe()
                clean_weights = ef.clean_weights()

                # Enforce the minimum allocation for each ticker
                for ticker in tickers:
                    if ticker not in clean_weights or clean_weights[ticker] < min_alloc:
                        clean_weights[ticker] = min_alloc

                total = sum(clean_weights.values())
                st.session_state.optimized_weights = {k: v / total for k, v in clean_weights.items()}
                st.sidebar.success("Optimization successful!")
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")

        st.sidebar.header("Weight Source")
        use_opt = st.sidebar.radio("Use weights from:", ("Optimized", "Manual"))

        # Update manual weights only if user chooses Manual
        if use_opt == "Manual":
            manual_weights = {}
            total = 0.0
            for ticker in tickers:
                # Use optimized weight as default if available, otherwise equal weight
                default_val = st.session_state.optimized_weights.get(ticker, 1.0/len(tickers)) * 100
                new_val = st.sidebar.slider(f"{ticker} (%)", min_alloc * 100, 100.0, float(default_val), 0.1) / 100
                manual_weights[ticker] = new_val
                total += new_val
            st.session_state.manual_weights = {k: v / total for k, v in manual_weights.items()}

        # Choose which weights to use
        if use_opt == "Optimized" and st.session_state.optimized_weights:
            current_weights = st.session_state.optimized_weights
        else:
            current_weights = st.session_state.manual_weights

        weights_df = pd.DataFrame.from_dict(current_weights, orient='index', columns=['Weight'])
        weights_df.index.name = 'Ticker'

        # Performance calculation
        def calculate_performance(weights_dict):
            # Ensure weights order aligns with mu index
            weight_arr = np.array([weights_dict.get(ticker, 0.0) for ticker in mu.index])
            port_return = np.dot(mu, weight_arr)
            port_vol = np.sqrt(np.dot(weight_arr.T, np.dot(S, weight_arr)))
            sharpe = (port_return - 0.02) / port_vol if port_vol > 0 else 0
            return port_return, port_vol, sharpe

        curr_return, curr_volatility, curr_sharpe = calculate_performance(current_weights)

        # Display allocation and performance
        st.subheader("Allocation")
        st.dataframe(weights_df.style.format("{:.2%}"))

        st.subheader("Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Return", f"{curr_return:.2%}")
        col2.metric("Volatility", f"{curr_volatility:.2%}")
        col3.metric("Sharpe", f"{curr_sharpe:.2f}")

        # Efficient Frontier calculation and plotting
        st.subheader("Efficient Frontier")
        try:
            target_returns = np.linspace(mu.min() * 0.7, mu.max() * 1.3, 40)
            efficient_frontier_points = []
            for target_return in target_returns:
                try:
                    temp_ef = EfficientFrontier(mu, S, weight_bounds=(min_alloc, 1))
                    temp_ef.efficient_return(target_return)
                    ret, vol, _ = temp_ef.portfolio_performance()
                    efficient_frontier_points.append((vol, ret))
                except Exception:
                    continue
            if efficient_frontier_points:
                efficient_frontier_points.sort(key=lambda x: x[0])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in efficient_frontier_points],
                    y=[p[1] for p in efficient_frontier_points],
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=[curr_volatility],
                    y=[curr_return],
                    mode='markers',
                    marker=dict(color='red', size=12),
                    name='Current Portfolio'
                ))
                fig.update_layout(
                    title='Efficient Frontier',
                    xaxis_title='Volatility',
                    yaxis_title='Expected Return',
                    xaxis=dict(tickformat='.2%'),
                    yaxis=dict(tickformat='.2%')
                )
                st.plotly_chart(fig)
            else:
                st.warning("Could not generate efficient frontier.")
        except Exception as e:
            st.error(f"Error plotting frontier: {str(e)}")

    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        import traceback
        st.write("Traceback:", traceback.format_exc())


if __name__ == "__main__":
    main()
