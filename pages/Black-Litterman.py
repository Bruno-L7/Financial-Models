import streamlit as st
import pandas as pd
import numpy as np
from pypfopt import BlackLittermanModel, risk_models
from pypfopt import EfficientFrontier, objective_functions
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import warnings
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random, retry_if_exception_type
warnings.filterwarnings('ignore')

st.title("Black-Litterman Portfolio Optimization")

# Cache market cap data to avoid repeated API calls
@st.cache_data(ttl=3600*6)  # Cache for 6 hours
def get_market_caps_cached(tickers_str):
    """Cache market caps for multiple tickers"""
    tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]
    return get_all_market_caps(tickers)

# Improved retry decorator with jitter and longer waits
@retry(
    stop=stop_after_attempt(5),  # Increased attempts
    wait=wait_exponential(multiplier=1, min=4, max=30) + wait_random(0, 2),  # Exponential backoff with jitter
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception))
)
def get_market_cap(ticker):
    """Fetch market cap with improved retry logic"""
    try:
        # Add random delay to prevent rate limiting
        time.sleep(random.uniform(0.5, 2.0))
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Prioritize direct market cap
        market_cap = info.get('marketCap', info.get('totalMarketCap', np.nan))
        
        # Fallback: Calculate from shares outstanding and price
        if pd.isna(market_cap) or market_cap <= 0:
            shares = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', np.nan))
            price = info.get('regularMarketPrice', info.get('currentPrice', np.nan))
            if not pd.isna(shares) and not pd.isna(price) and shares > 0 and price > 0:
                market_cap = shares * price
            else:
                market_cap = np.nan
        
        return market_cap / 1e9  # Convert to billions
    
    except Exception as e:
        st.warning(f"Temporary error fetching data for {ticker}, retrying...")
        raise e

def get_all_market_caps(tickers):
    """Fetch market caps with better rate limiting and error handling"""
    market_caps = {}
    
    with st.spinner('Fetching market data (this may take a moment)...'):
        progress_bar = st.progress(0)
        for i, ticker in enumerate(tickers):
            try:
                market_caps[ticker] = get_market_cap(ticker)
                progress_bar.progress((i + 1) / len(tickers))
            except Exception as e:
                # If we fail after retries, assign NaN but continue with other tickers
                st.warning(f"Could not fetch market cap for {ticker}: {str(e)}")
                market_caps[ticker] = np.nan
                continue
    
    # Check if we have enough valid market caps
    valid_caps = sum(1 for cap in market_caps.values() if not np.isnan(cap))
    if valid_caps < 2:
        raise ValueError("Could not fetch enough valid market cap data. Please try again later or use different tickers.")
    
    return market_caps

def preprocess_returns(returns):
    """Preprocess returns data"""
    returns = returns.iloc[1:]
    lower_quantile = returns.quantile(0.01)
    upper_quantile = returns.quantile(0.99)
    for col in returns.columns:
        returns[col] = returns[col].clip(lower=lower_quantile[col], upper=upper_quantile[col])
    return returns.clip(lower=-0.25, upper=0.25)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(0, 2),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception))
)
def get_price_data(tickers, start_date, end_date):
    """Fetch price data with better retry logic"""
    # Add a random delay
    time.sleep(random.uniform(0.5, 1.5))
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    
    min_required_data = 252
    valid_columns = prices.count() >= min_required_data
    prices = prices.loc[:, valid_columns]
    
    if prices.empty:
        raise ValueError("No valid price data found")
    
    return prices, list(prices.columns)

def calculate_robust_covariance(returns):
    """Calculate robust covariance matrix"""
    returns_np = returns.values.astype(np.float64)
    returns_centered = returns_np - np.mean(returns_np, axis=0)
    n = returns_centered.shape[0]
    sample_cov = np.dot(returns_centered.T, returns_centered) / (n - 1)
    shrinkage_target = np.diag(np.diag(sample_cov))
    shrinkage_lambda = 0.5
    cov_matrix = (1 - shrinkage_lambda) * sample_cov + shrinkage_lambda * shrinkage_target
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    cov_matrix += np.eye(len(cov_matrix)) * 1e-8
    return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

def calculate_market_implied_returns(market_caps, risk_aversion, cov_matrix):
    """Calculate market implied returns"""
    total_market_cap = sum(market_caps.values())
    market_weights = {ticker: cap/total_market_cap for ticker, cap in market_caps.items()}
    weights = np.array([market_weights[ticker] for ticker in cov_matrix.columns])
    implied_returns = risk_aversion * np.dot(cov_matrix, weights)
    return pd.Series(implied_returns, index=cov_matrix.columns)

# Main page layout
st.write("### 1. Portfolio Setup")
col1, col2 = st.columns(2)

with col1:
    st.write("#### Stock Selection")
    tickers_input = st.text_input(
        "Enter stock tickers",
        "AAPL,MSFT,AMZN,GOOG,TSLA",
        help="Enter stock symbols separated by commas (e.g., AAPL,MSFT,AMZN)"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

with col2:
    st.write("#### Time Period")
    default_start = pd.Timestamp.now() - pd.DateOffset(years=3)
    default_end = pd.Timestamp.now()
    start_date = st.date_input("Start date", default_start)
    end_date = st.date_input("End date", default_end)

st.write("### 2. Risk Parameters")
risk_aversion = st.slider(
    "Risk Aversion Level",
    1.0, 5.0, 3.0, 0.1,
    help="Higher values indicate more risk-averse investment strategy"
)

st.write("### 3. Investment Views")
st.write("Define your market expectations:")
num_views = st.number_input("Number of Investment Views", 1, 5, 2)

views = []
confidences = []

for i in range(num_views):
    st.write(f"#### Investment View {i+1}")
    cols = st.columns(3)
    
    with cols[0]:
        view_type = st.radio(
            "View Type",
            ["Absolute", "Relative"],
            key=f"view_type_{i}",
            help="Absolute: direct return expectation for a stock\nRelative: expected outperformance between stocks"
        )
    
    with cols[1]:
        if view_type == "Absolute":
            ticker = st.selectbox("Select Stock", tickers, key=f"abs_ticker_{i}")
            return_view = st.slider(
                "Expected Annual Return (%)",
                -10.0, 30.0, 5.0,
                key=f"abs_return_{i}"
            ) / 100
            views.append(("absolute", ticker, return_view))
        else:
            ticker1 = st.selectbox("Outperforming Stock", tickers, key=f"rel_ticker1_{i}")
            ticker2 = st.selectbox(
                "Underperforming Stock",
                [t for t in tickers if t != ticker1],
                key=f"rel_ticker2_{i}"
            )
            outperformance = st.slider(
                "Outperformance (%)",
                0.1, 10.0, 3.0,
                key=f"rel_outperf_{i}"
            ) / 100
            views.append(("relative", ticker1, ticker2, outperformance))
    
    with cols[2]:
        confidence = st.slider(
            "Confidence Level (%)",
            1, 100, 70,
            key=f"conf_{i}",
            help="How confident are you in this view?"
        ) / 100
        confidences.append(confidence)

# Add fallback option
st.write("### 4. Fallback Options")
use_cached_data = st.checkbox("Use cached market data if available", value=True, 
                             help="This can help avoid rate limit errors by using previously fetched data")

run_btn = st.button("Optimize Portfolio", type="primary")

if run_btn:
    try:
        with st.spinner('Optimizing portfolio...'):
            # Try to get market caps and filter valid tickers
            try:
                # If caching is enabled, try to get cached data first
                if use_cached_data:
                    try:
                        market_caps = get_market_caps_cached(tickers_input)
                        st.success("Successfully loaded cached market data")
                    except Exception:
                        market_caps = get_all_market_caps(tickers)
                else:
                    market_caps = get_all_market_caps(tickers)
            except Exception as e:
                st.error(f"Failed to fetch market caps: {str(e)}")
                st.error("This is likely due to rate limiting by Yahoo Finance. Try enabling the cache option or waiting before retrying.")
                st.stop()

            valid_tickers = [t for t, cap in market_caps.items() if not np.isnan(cap)]
            
            if len(valid_tickers) < 2:
                st.error("Need at least 2 valid tickers with market cap data")
                st.stop()
            
            # Get price data and process
            st.info(f"Processing data for {len(valid_tickers)} valid tickers: {', '.join(valid_tickers)}")
            
            try:
                price_data, valid_tickers = get_price_data(valid_tickers, start_date, end_date)
            except Exception as e:
                st.error(f"Failed to fetch price data: {str(e)}")
                st.error("This is likely due to rate limiting. Try again later or with fewer tickers.")
                st.stop()
                
            daily_returns = price_data.pct_change()
            cleaned_returns = preprocess_returns(daily_returns)
            cov_matrix = calculate_robust_covariance(cleaned_returns)
            
            # Calculate market implied returns
            prior_returns = calculate_market_implied_returns(
                {t: market_caps[t] for t in valid_tickers},
                risk_aversion,
                cov_matrix
            )
            
            # Filter and process views
            filtered_views = []
            filtered_confidences = []
            for view, confidence in zip(views, confidences):
                if view[0] == "absolute":
                    if view[1] in valid_tickers:
                        filtered_views.append(view)
                        filtered_confidences.append(confidence)
                else:
                    if view[1] in valid_tickers and view[2] in valid_tickers:
                        filtered_views.append(view)
                        filtered_confidences.append(confidence)
            
            if not filtered_views:
                st.error("No valid views remaining after filtering")
                st.stop()
            
            # Prepare views matrices
            P = []
            Q = []
            omega = np.zeros((len(filtered_views), len(filtered_views)))
            
            for i, (view, confidence) in enumerate(zip(filtered_views, filtered_confidences)):
                view_vector = np.zeros(len(valid_tickers))
                if view[0] == "absolute":
                    idx = valid_tickers.index(view[1])
                    view_vector[idx] = 1
                    Q.append(view[2])
                else:
                    idx1 = valid_tickers.index(view[1])
                    idx2 = valid_tickers.index(view[2])
                    view_vector[idx1] = 1
                    view_vector[idx2] = -1
                    Q.append(view[3])
                P.append(view_vector)
                omega[i, i] = (1 - confidence) / confidence
            
            # Create and run Black-Litterman model
            bl = BlackLittermanModel(
                cov_matrix=cov_matrix,
                pi=prior_returns,
                absolute_views=None,
                P=np.array(P),
                Q=np.array(Q),
                omega=omega,
                tau=0.05
            )
            
            posterior_rets = bl.bl_returns()
            posterior_cov = bl.bl_cov()
            
            # Set risk-free rate
            risk_free_rate = min(0.02, posterior_rets.min() - 0.0001)
            
            # Portfolio optimization
            ef = EfficientFrontier(posterior_rets, posterior_cov)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.add_constraint(lambda x: x >= 0)
            ef.add_constraint(lambda x: sum(x) == 1)
            
            try:
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                optimization_method = "Maximum Sharpe Ratio"
            except Exception:
                weights = ef.min_volatility()
                optimization_method = "Minimum Volatility"
            
            cleaned_weights = ef.clean_weights(cutoff=0.01)
            
            # Display results
            st.write("## Optimization Results")
            
            # Create three columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Portfolio Allocation")
                weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
                st.dataframe(weights_df.style.format("{:.2%}"))
                
                # Add pie chart for portfolio weights
                fig_pie = px.pie(
                    values=weights_df['Weight'],
                    names=weights_df.index,
                    title='Portfolio Allocation',
                    hole=0.3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.write("### Expected Performance")
                perf = ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
                metrics = {
                    "Expected Annual Return": f"{perf[0]:.2%}",
                    "Annual Volatility": f"{perf[1]:.2%}",
                    "Sharpe Ratio": f"{perf[2]:.2f}"
                }
                for label, value in metrics.items():
                    st.metric(label, value)
                
                # Create risk-return scatter plot
                # Calculate individual asset metrics
                asset_returns = posterior_rets
                asset_volatilities = np.sqrt(np.diag(posterior_cov))
                
                # Create scatter plot
                fig_scatter = go.Figure()
                
                # Add individual assets
                fig_scatter.add_trace(go.Scatter(
                    x=asset_volatilities,
                    y=asset_returns,
                    mode='markers+text',
                    name='Individual Assets',
                    text=valid_tickers,
                    textposition="top center",
                    marker=dict(size=10, color='blue')
                ))
                
                # Add optimized portfolio
                fig_scatter.add_trace(go.Scatter(
                    x=[perf[1]],
                    y=[perf[0]],
                    mode='markers+text',
                    name='Optimized Portfolio',
                    text=['Portfolio'],
                    textposition="top center",
                    marker=dict(size=15, color='red', symbol='star')
                ))
                
                # Update layout
                fig_scatter.update_layout(
                    title='Risk-Return Profile',
                    xaxis_title='Annual Volatility',
                    yaxis_title='Expected Annual Return',
                    xaxis_tickformat='.1%',
                    yaxis_tickformat='.1%',
                    showlegend=True,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Add historical performance visualization
            st.write("### Historical Performance Analysis")
            
            # Calculate historical portfolio value
            portfolio_weights = pd.Series(cleaned_weights)
            historical_prices = price_data[portfolio_weights.index]
            normalized_prices = historical_prices / historical_prices.iloc[0]
            portfolio_value = (normalized_prices * portfolio_weights).sum(axis=1)
            
            # Create line plot
            fig_historical = go.Figure()
            
            # Add portfolio value line
            fig_historical.add_trace(go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='red', width=2)
            ))
            
            # Add individual asset lines
            for ticker in portfolio_weights.index:
                fig_historical.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[ticker].values,
                    mode='lines',
                    name=ticker,
                    line=dict(width=1),
                    opacity=0.5
                ))
            
            # Update layout
            fig_historical.update_layout(
                title='Historical Price Performance (Normalized)',
                xaxis_title='Date',
                yaxis_title='Normalized Value',
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_historical, use_container_width=True)

    except Exception as e:
        st.error(f"Portfolio optimization failed: {str(e)}")
        st.stop()
