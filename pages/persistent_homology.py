import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import gudhi as gd
import matplotlib.pyplot as plt

st.set_page_config(page_title="TopoFinance Analyzer", layout="wide")

# Sidebar controls
st.sidebar.header("Analysis Parameters")
tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", 
                              "AAPL,MSFT,AMZN,GOOG,TSLA,XOM,JPM,WMT,GE,BAC").split(',')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))

@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Close']
    return data.dropna(axis=1)

@st.cache_data
def process_data(data):
    returns = data.pct_change().dropna()
    corr_matrix = returns.corr().values
    distance_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(distance_matrix, 0)
    return returns, corr_matrix, distance_matrix

@st.cache_data
def compute_persistence(distance_matrix, tickers):
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=np.max(distance_matrix))
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()
    
    intervals = []
    persistence_pairs = simplex_tree.persistence_pairs()
    
    for i, (dim, (birth, death)) in enumerate(persistence):
        creator = []
        if i < len(persistence_pairs):
            pair = persistence_pairs[i]
            if pair and len(pair[0]) > 0:
                creator = [tickers[v] for v in pair[0]]
        
        # Use consistent column names
        intervals.append({
            'Dimension': dim,
            'Formation': birth,  # Changed from 'Birth' to 'Formation'
            'Dissolution': death if not np.isinf(death) else np.inf,
            'Persistence': death - birth if not np.isinf(death) else np.inf,
            'Stocks': " & ".join(creator) if creator else "Market-wide"
        })
    
    return pd.DataFrame(intervals), persistence

# Main app
st.title("Topological Analysis of Stock Market Correlations")

# Data loading and processing
with st.spinner("Loading market data..."):
    data = load_data(tickers, start_date, end_date)
    
if data.empty:
    st.error("No data retrieved - check your ticker symbols and date range")
    st.stop()

returns, corr_matrix, distance_matrix = process_data(data)

# Persistent homology computation
try:
    with st.spinner("Computing persistent homology..."):
        persistence_df, persistence_data = compute_persistence(distance_matrix, tickers)
except IndexError as e:
    st.error(f"Error computing persistence: {str(e)}")
    st.stop()

# Visualization
st.header("Market Structure Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(ticks=range(len(tickers)), labels=tickers, rotation=90)
    plt.yticks(ticks=range(len(tickers)), labels=tickers)
    st.pyplot(fig)

with col2:
    st.subheader("Distance Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar(label='Distance')
    plt.xticks(ticks=range(len(tickers)), labels=tickers, rotation=90)
    plt.yticks(ticks=range(len(tickers)), labels=tickers)
    st.pyplot(fig)

# Helper function to generate explanations
def generate_insights(persistence_df, persistence_diagram):
    insights = []
    
    # Analyze H0 features (clusters)
    h0 = persistence_df[persistence_df['Dimension'] == 0]
    h0_persistent = h0[h0['Persistence'] > np.quantile(h0['Persistence'], 0.75)]
    
    insights.append("### Market Structure Insights")
    
    if len(h0_persistent) > 0:
        cluster_stability = " ".join([
            f"We identified {len(h0_persistent)} persistent stock clusters.",
            "These groups maintained strong correlations throughout the analysis period.",
            "Key clusters involve:"
        ])
        insights.append(cluster_stability)
        
        # Get top 3 most persistent clusters
        for _, row in h0_persistent.nlargest(3, 'Persistence').iterrows():
            insights.append(f"- **{row['Stocks']}** (Duration: {row['Persistence']})")
    else:
        insights.append("The market shows **low cluster stability** - no long-lasting stock groups were identified.")
    
    # Analyze H1 features (cycles)
    h1 = persistence_df[persistence_df['Dimension'] == 1]
    
    if len(h1) > 0:
        cycle_insight = [
            "\n### Cyclic Relationships Detected",
            f"We found {len(h1)} cyclical patterns in stock correlations:",
            "These indicate rotating leadership or feedback loops between:"
        ]
        insights.extend(cycle_insight)
        
        for _, row in h1.nlargest(3, 'Persistence').iterrows():
            insights.append(f"- **{row['Stocks']}** (Duration: {row['Persistence']})")
    else:
        insights.append("\nNo significant cyclic relationships were found between stocks.")
    
    # Diagram shape analysis
    diagram_points = np.array([(birth, death) for _, (birth, death) in persistence_diagram])
    if len(diagram_points) > 0:
        x_max = diagram_points[:, 0].max()
        y_min = diagram_points[:, 1].min()
        
        if y_min > 0.4:
            insights.append("\n### Market Stability Note")
            insights.append("The market shows **strong persistent structure** - correlations remain stable across different time horizons.")
        elif x_max < 0.2:
            insights.append("\n### Market Stability Note")
            insights.append("The market exhibits **fragile relationships** - most correlations break down quickly.")
    
    return "\n\n".join(insights)

def main():
    st.title("Market Structure Analysis")
    st.write("""
    This tool analyzes stock market correlations through topological data analysis.
    It automatically explains market structure using persistent homology.
    """)
    
    # Data loading and processing
    tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", 
                                  "AAPL,MSFT,AMZN,GOOG,TSLA,XOM,JPM,WMT,GE,BAC").split(',')
    
    with st.spinner("Analyzing market structure..."):
        # Data processing
        data = yf.download(tickers, period="3y")['Close'].dropna(axis=1)
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr().values
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        np.fill_diagonal(distance_matrix, 0)
        
        # Compute persistence
        persistence_df, persistence_diagram = compute_persistence(distance_matrix, data.columns.tolist())
        
        # Visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            gd.plot_persistence_diagram(persistence_diagram, axes=ax, legend=True)
            plt.title("Topological Feature Lifespan Diagram")
            st.pyplot(fig)
            
        with col2:
            st.markdown("## Diagram Guide")
            st.markdown("""
            - **Blue Points (H₀):** Stock clusters forming/merging
            - **Orange Points (H₁):** Cyclic relationships
            - **Position Meaning:**
              - X-axis: Formation time (correlation strength threshold)
              - Y-axis: Dissolution time
            - **Distance from diagonal** = Persistence duration
            """)
        
        # Generate and display insights
        st.markdown("---")
        st.markdown(generate_insights(persistence_df, persistence_diagram))
        
        # Raw data tables
        with st.expander("View Persistence Intervals"):
            st.dataframe(persistence_df.style.format({
                'Birth': "{:.3f}",
                'Death': lambda x: "∞" if np.isinf(x) else "{:.3f}".format(x),
                'Persistence': lambda x: "∞" if np.isinf(x) else "{:.3f}".format(x)
            }))

if __name__ == "__main__":
    main()



# Interpretation
st.markdown("""
**Interpretation Guide:**
- **Dimension 0:** Stock clusters forming connected components
- **Dimension 1:** Cyclic relationships between stocks
- **Long durations:** Stable market patterns
- **Infinite (∞) persistence:** Features that never disappear
""")