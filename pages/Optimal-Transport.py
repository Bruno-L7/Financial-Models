import numpy as np
import ot  # POT library for Optimal Transport
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

def analyze_transport_plan(transport_plan, cost_matrix):
    """Generate textual analysis of the transport plan"""
    analysis = []
    num_assets = transport_plan.shape[0]
    
    # Find largest single transfer
    max_transfer = np.max(transport_plan)
    max_indices = np.where(transport_plan == max_transfer)
    for i, j in zip(max_indices[0], max_indices[1]):
        analysis.append(
            f"Largest transfer: {max_transfer:.2%} from Current Asset {i+1} to Target Asset {j+1}"
        )
    
    # Calculate self-transitions (diagonal elements)
    self_transfers = np.diag(transport_plan).sum()
    analysis.append(
        f"Self-transitions (no change): {self_transfers:.2%} of total portfolio remains unchanged"
    )
    
    # Identify zero transfers
    zero_transfers = np.sum(transport_plan == 0)
    if zero_transfers > 0:
        zero_percent = zero_transfers / (num_assets**2) * 100
        analysis.append(
            f"{zero_transfers} zero transfers ({zero_percent:.1f}% of possible transitions)"
        )
    else:
        analysis.append("All possible transitions are being used")
    
    # Calculate cost efficiency
    total_cost = np.sum(transport_plan * cost_matrix)
    cost_per_transfer = total_cost / (1 - self_transfers) if (1 - self_transfers) > 0 else 0
    analysis.append(
        f"Cost efficiency: ${total_cost:.4f} total cost, ${cost_per_transfer:.4f} per unit reallocated"
    )
    
    return analysis

# Function to create a cost matrix
def create_cost_matrix(num_assets):
    st.subheader("Cost Matrix")
    st.write("Define the cost of transitioning between assets.")
    
    # Initialize the cost matrix
    cost_matrix = np.zeros((num_assets, num_assets))
    
    # Create a grid of input fields
    cols = st.columns(num_assets)
    for i in range(num_assets):
        with cols[i]:
            st.write(f"Row {i+1}")
            for j in range(num_assets):
                cost_matrix[i, j] = st.number_input(
                    f"Cost from Asset {i+1} → Asset {j+1}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1 if i != j else 0.0,  # Default: No cost for self-transitions
                    step=0.01
                )
    return cost_matrix

# Function to validate and normalize a distribution
def get_distribution(input_str, num_assets):
    try:
        distribution = np.array([float(x.strip()) for x in input_str.split(",")])
        if len(distribution) != num_assets:
            st.error(f"The distribution must have exactly {num_assets} values.")
            return None
        distribution /= np.sum(distribution)  # Normalize to sum to 1
        return distribution
    except ValueError:
        st.error("Invalid input. Please enter numeric values separated by commas.")
        return None
    
# Streamlit App
st.title("Optimal Transport in Quantitative Finance")

# Step 1: Get the number of assets
num_assets = st.number_input("Enter the number of assets in the portfolio:", min_value=2, value=4)

# Step 2: Get the current portfolio distribution
current_portfolio_input = st.text_input(
    f"Enter the current portfolio weights (comma-separated, {num_assets} values):", 
    value="0.4, 0.3, 0.2, 0.1"
)
current_portfolio = get_distribution(current_portfolio_input, num_assets)

# Step 3: Get the target portfolio distribution
target_portfolio_input = st.text_input(
    f"Enter the target portfolio weights (comma-separated, {num_assets} values):", 
    value="0.2, 0.3, 0.3, 0.2"
)
target_portfolio = get_distribution(target_portfolio_input, num_assets)

# Step 4: Get the cost matrix using the new user-friendly function
cost_matrix = create_cost_matrix(num_assets)

# Step 5: Compute and display the results
if st.button("Compute Optimal Transport Plan"):
    if current_portfolio is not None and target_portfolio is not None and cost_matrix is not None:
        # Compute the Optimal Transport Plan
        optimal_transport_plan = ot.emd(current_portfolio, target_portfolio, cost_matrix)
        total_cost = np.sum(optimal_transport_plan * cost_matrix)

        # Display results
        st.subheader("Results")
        st.write("Current Portfolio Weights:", current_portfolio)
        st.write("Target Portfolio Weights:", target_portfolio)
        st.write("Optimal Transport Plan:")
        st.write(optimal_transport_plan)
        st.write(f"Total Transportation Cost: {total_cost:.4f}")

        # Visualize the transport plan
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(optimal_transport_plan, cmap='Blues', interpolation='nearest')
        fig.colorbar(cax, label="Transported Mass")
        ax.set_title("Optimal Transport Plan")
        ax.set_xlabel("Target Assets")
        ax.set_ylabel("Current Assets")
        ax.set_xticks(range(num_assets))
        ax.set_yticks(range(num_assets))
        ax.set_xticklabels([f"Asset {i+1}" for i in range(num_assets)])
        ax.set_yticklabels([f"Asset {i+1}" for i in range(num_assets)])
        
        st.pyplot(fig)

        analysis = analyze_transport_plan(optimal_transport_plan, cost_matrix)
        st.subheader("Plan Analysis")
        for insight in analysis:
            st.write(f"- {insight}")

        # Add detailed transfer explanations
        st.write("\n**Key Transitions:**")
        threshold = 0.05  # Consider transfers above 5% as significant
        significant_transfers = np.argwhere(optimal_transport_plan >= threshold)
        if len(significant_transfers) > 0:
            for i, j in significant_transfers:
                if i != j:  # Exclude self-transitions
                    st.write(
                        f"- Current Asset {i+1} → Target Asset {j+1}: "
                        f"{optimal_transport_plan[i,j]:.2%} (Cost: ${cost_matrix[i,j]:.2f} per unit)"
                    )
        else:
            st.write("No significant individual transfers above 5% threshold")



