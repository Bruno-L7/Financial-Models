import numpy as np
import ot  # POT library for Optimal Transport
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

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

# Function to get the cost matrix from user input
def get_cost_matrix(input_rows, num_assets):
    try:
        cost_matrix = np.zeros((num_assets, num_assets))
        for i, row_input in enumerate(input_rows):
            row = np.array([float(x.strip()) for x in row_input.split(",")])
            if len(row) != num_assets:
                st.error(f"Row {i+1} must have exactly {num_assets} values.")
                return None
            cost_matrix[i] = row
        return cost_matrix
    except ValueError:
        st.error("Invalid input in cost matrix. Please enter numeric values separated by commas.")
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

# Step 4: Get the cost matrix
st.subheader("Cost Matrix")
cost_matrix_inputs = []
for i in range(num_assets):
    default_row = ", ".join(["0.1"] * num_assets)  # Default row values
    row_input = st.text_input(f"Row {i+1} of the cost matrix (comma-separated, {num_assets} values):", value=default_row)
    cost_matrix_inputs.append(row_input)
cost_matrix = get_cost_matrix(cost_matrix_inputs, num_assets)

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