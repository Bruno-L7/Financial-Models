import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def heston_model():
    st.title("Heston Model Simulation")
    
    # Sidebar parameters
    st.sidebar.header("Model Parameters")
    mu = st.sidebar.slider("Drift (μ)", 0.0, 0.1, 0.05, 0.001)
    kappa = st.sidebar.slider("Mean Reversion Rate (κ)", 0.1, 5.0, 2.0, 0.1)
    theta = st.sidebar.slider("Long-term Variance (θ)", 0.01, 0.2, 0.04, 0.01)
    sigma = st.sidebar.slider("Vol of Vol (σ)", 0.1, 1.0, 0.2, 0.01)
    rho = st.sidebar.slider("Correlation (ρ)", -1.0, 1.0, -0.7, 0.01)
    v0 = st.sidebar.slider("Initial Variance", 0.01, 0.2, 0.04, 0.01)
    S0 = st.sidebar.slider("Initial Price", 50.0, 200.0, 100.0, 1.0)
    T = st.sidebar.slider("Time Horizon (Years)", 0.1, 5.0, 1.0, 0.1)
    N = st.sidebar.slider("Time Steps", 10, 1000, 252)
    M = st.sidebar.slider("Simulations", 1, 50, 5)

    dt = T / N
    
    # Initialize arrays
    S_paths = np.zeros((M, N + 1))
    v_paths = np.zeros((M, N + 1))
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0

    # Simulation
    for m in range(M):
        logS = np.log(S0)
        v = v0
        for n in range(N):
            Z1 = np.random.normal()
            Z2 = np.random.normal()
            dW1 = np.sqrt(dt) * Z1
            dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
            
            v_new = v + kappa * (theta - v) * dt + sigma * np.sqrt(v) * dW2
            v_new = max(v_new, 0)
            
            logS += (mu - 0.5 * v) * dt + np.sqrt(v) * dW1
            
            S_paths[m, n + 1] = np.exp(logS)
            v_paths[m, n + 1] = v_new
            v = v_new

    # Create interactive plot
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=("Asset Price Paths", "Variance Paths"),
                       vertical_spacing=0.15)

    # Add price traces
    t = np.linspace(0, T, N + 1)
    for m in range(M):
        fig.add_trace(go.Scatter(
            x=t,
            y=S_paths[m],
            mode='lines',
            name=f'Sim {m+1}',
            hovertemplate="<b>Time</b>: %{x:.2f} yrs<br><b>Price</b>: %{y:.2f}<br>Simulation: %{name}",
            line=dict(width=1)
        ), row=1, col=1)

    # Add variance traces
    for m in range(M):
        fig.add_trace(go.Scatter(
            x=t,
            y=v_paths[m],
            mode='lines',
            name=f'Sim {m+1}',
            hovertemplate="<b>Time</b>: %{x:.2f} yrs<br><b>Variance</b>: %{y:.4f}<br>Simulation: %{name}",
            line=dict(width=1),
            showlegend=False
        ), row=2, col=1)

    # Update layout
    fig.update_layout(
        height=800,
        hovermode='x unified',
        title_text="Heston Model Simulations",
        margin=dict(t=80, b=80)
    )
    
    # Add axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Variance", row=2, col=1)
    fig.update_xaxes(title_text="Time (Years)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Display parameters
    st.subheader("Current Parameters:")
    st.write(f"""
    - Drift (μ): {mu}
    - Mean Reversion Rate (κ): {kappa}
    - Long-term Variance (θ): {theta}
    - Vol of Vol (σ): {sigma}
    - Correlation (ρ): {rho}
    - Initial Variance: {v0}
    - Initial Price: {S0}
    - Time Horizon: {T} years
    - Time Steps: {N}
    - Simulations: {M}
    """)

if __name__ == "__main__":
    heston_model()