import numpy as np
import matplotlib.pyplot as plt
import ot
import streamlit as st
import numpy as np

# Generate synthetic return distributions
np.random.seed(42)
n_samples = 50

# Market regime distributions
returns1 = np.random.normal(0.08, 0.15, n_samples)  # Bull market
returns2 = np.concatenate([                        # Volatile market
    np.random.normal(-0.05, 0.25, n_samples//2),
    np.random.normal(0.12, 0.10, n_samples//2)
])

# Uniform probability distributions
a = np.ones(n_samples)/n_samples  # Source distribution (current portfolio)
b = np.ones(n_samples)/n_samples  # Target distribution (desired allocation)

# Cost matrix and transport computation
X_source = returns1.reshape((-1, 1))
X_target = returns2.reshape((-1, 1))
C = ot.dist(X_source, X_target, metric='sqeuclidean')  # Transportation cost matrix
epsilon = 0.01  # Entropy regularization parameter

# Compute optimal transport plan
transport_plan = ot.sinkhorn(a, b, C, reg=epsilon, numItermax=10000)

# Calculate metrics
# 1. Regularized Wasserstein Distance
wasserstein_dist = np.sum(transport_plan * C)

# 2. Total Expected Transaction Cost (using bid-ask spread model)
spread_rate = 0.02  # 2% transaction cost
transaction_cost = spread_rate * (1 - np.trace(transport_plan))

# 3. Sinkhorn Divergence
def compute_sinkhorn_divergence(a, b, X_a, X_b, epsilon):
    C_ab = ot.dist(X_a, X_b, metric='sqeuclidean')
    C_aa = ot.dist(X_a, X_a, metric='sqeuclidean')
    C_bb = ot.dist(X_b, X_b, metric='sqeuclidean')
    
    W_ab = ot.sinkhorn2(a, b, C_ab, reg=epsilon)
    W_aa = ot.sinkhorn2(a, a, C_aa, reg=epsilon)
    W_bb = ot.sinkhorn2(b, b, C_bb, reg=epsilon)
    
    return W_ab - 0.5*(W_aa + W_bb)

sinkhorn_div = compute_sinkhorn_divergence(a, b, X_source, X_target, epsilon)

# Configure Streamlit and Page Layout
st.set_page_config(page_title="Entropy-Regularized Optimal Transport", layout="wide")
st.title("Entropy-Regularized Optimal Transport (Sinkhorn algorithm) to a portfolio rebalancing")

# Parameter Configuration Sidebar
with st.sidebar:
    st.header("Portfolio Configuration")
    seed = st.slider("Random Seed (Reproducibility)", 0, 100, 42, key="seed")
    n_samples = st.slider("Number of Return Samples", 20, 200, 50, key="n_samples")
    entropy_reg = st.slider("Entropy Regularization (ε)", 0.001, 0.2, 0.01, key="entropy")
    spread_rate = st.slider("Bid-Ask Spread (%)", 0.0, 5.0, 2.0, step=0.1, key="spread")

# Synthetic Portfolio Generation
np.random.seed(seed)
current_returns = np.random.normal(0.08, 0.15, n_samples)  # Stable growth period
target_returns = np.concatenate([
    np.random.normal(-0.05, 0.25, n_samples//2),  # Economic downturn
    np.random.normal(0.12, 0.10, n_samples//2)    # Recovery phase
])
prob_current = np.ones(n_samples)/n_samples 
prob_target = np.ones(n_samples)/n_samples 

# Compute Optimal Transport
X_src = current_returns.reshape((-1, 1))
X_tgt = target_returns.reshape((-1, 1))
cost_matrix = ot.dist(X_src, X_tgt, metric="sqeuclidean")
transport_plan = ot.sinkhorn(prob_current, prob_target, cost_matrix, reg=entropy_reg)

# Metrics Calculation
wasserstein = np.sum(transport_plan * cost_matrix)
sinkhorn_div = compute_sinkhorn_divergence(prob_current, prob_target, X_src, X_tgt, entropy_reg)
txn_cost = spread_rate / 100 * (1 - np.trace(transport_plan))

# Define Sinkhorn Divergence Function
def compute_sinkhorn_divergence(a, b, X_a, X_b, epsilon):
    C_ab = ot.dist(X_a, X_b, metric="sqeuclidean")
    C_aa = ot.dist(X_a, X_a, metric="sqeuclidean")
    C_bb = ot.dist(X_b, X_b, metric="sqeuclidean")
    
    W_ab = ot.sinkhorn2(a, b, C_ab, reg=epsilon)
    W_aa = ot.sinkhorn2(a, a, C_aa, reg=epsilon)
    W_bb = ot.sinkhorn2(b, b, C_bb, reg=epsilon)
    
    return W_ab - 0.5 * (W_aa + W_bb)

# Visualization with Individual Plots
plt.style.use('bmh')  # Note: Requires installation of the 'seaborn-bmh' style

# Return Distribution Analysis
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.hist(current_returns, bins=30, alpha=0.5, label="Current Portfolio", color='#2ca02c')
ax1.hist(target_returns, bins=30, alpha=0.5, label="Target Portfolio", color='#1f77b4')
ax1.set_xlabel("Daily Return (%)")
ax1.set_ylabel("Frequency")
ax1.set_title("Return Distribution Comparison")
ax1.legend(fontsize=10)

# Optimal Rebalancing Plan
fig2, ax2 = plt.subplots(figsize=(8, 6))
cax = ax2.imshow(transport_plan, cmap='inferno', aspect='auto')
fig2.colorbar(cax, ax=ax2).set_label("Fraction of Capital Allocated")
ax2.set_xlabel("Target Assets")
ax2.set_ylabel("Current Assets")
ax2.set_title("Optimal Rebalancing Matrix")

# Capital Redistribution Flow
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(np.cumsum(transport_plan.sum(axis=0)), label="Target Weights", color='#1f77b4')
ax3.plot(np.cumsum(transport_plan.sum(axis=1)), label="Source Weights", color='#2ca02c')
ax3.set_xlabel("Asset Index")
ax3.set_ylabel("Cumulative Allocation")
ax3.set_title("Cumulative Capital Flows")
ax3.legend(fontsize=10)

# Graph Analysis and Insight Generation
def analyze_distribution():
    mean_diff = abs(np.mean(current_returns) - np.mean(target_returns))
    overlap = ot.emd2(prob_current, prob_target, cost_matrix)  # Earth Mover's Distance

    if mean_diff < 0.05:
        return ("The current and target portfolios have similar "
                "return profiles with minor deviations in their distributions.")
    else:
        return (f"The portfolios diverge significantly (mean difference: {mean_diff:.3f}). "
                "Rebalancing is necessary to align performance characteristics.")

def analyze_transport():
    avg_flow = np.mean(transport_plan)
    max_flow = np.max(transport_plan)

    if max_flow > 3 * avg_flow:
        return ("Capital is concentrated between a few assets, indicating "
                "a focused rebalancing strategy.")
    else:
        return ("Capital is redistributed fairly evenly across assets, "
                "reflecting a diversified strategy.")

def analyze_flow():
    flow_balance = np.abs(np.cumsum(transport_plan.sum(axis=0)).mean() -
                          np.cumsum(transport_plan.sum(axis=1)).mean())
    
    if flow_balance < 0.05:
        return "Capital flows are well-balanced during the rebalancing process."
    else:
        return (f"Imbalance detected in capital flows (Δ: {flow_balance:.3f}). "
                "Asymmetric allocation may be strategic or risky.")

# Display Analysis and Metrics
st.subheader("Return Distribution Analysis")
st.pyplot(fig1)
st.write(analyze_distribution())

st.subheader("Optimal Rebalancing Plan")
st.pyplot(fig2)
st.write(analyze_transport())

st.subheader("Capital Redistribution Flow")
st.pyplot(fig3)
st.write(analyze_flow())

# Display Portfolio Rebalancing Metrics Summary
st.markdown("---")
st.subheader("Rebalancing Metrics Summary")
st.markdown(f"**Wasserstein Distance**: {wasserstein:.4f} (Aggregation cost measure)")
st.markdown(f"**Sinkhorn Divergence**: {sinkhorn_div:.4f} (Portfolio discrepancy metric)")
st.markdown(f"**Expected Transaction Cost**: {txn_cost:.2%} (Based on spread rate: {spread_rate}%)")
st.markdown(f"**Entropy Regularization**: {entropy_reg} (Optimization smoothness factor)")

# Show Transport Plan Example
st.subheader("Sample Rebalancing Matrix (First 5x5 Assets)")
st.dataframe(transport_plan[:5, :5].round(3))