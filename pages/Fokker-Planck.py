import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Fokker-Planck Equation for Asset Price Distribution")

# Parameters in sidebar
st.sidebar.header("Parameters")
x_min = st.sidebar.slider(
    "Minimum asset price (x_min)",
    0.0, 200.0, 90.0, 1.0
)
x_max = st.sidebar.slider(
    "Maximum asset price (x_max)",
    min_value=x_min + 1.0,
    max_value=200.0,
    value=100.0,
    step=1.0
)
N = st.sidebar.slider(
    "Number of spatial points (N)",
    2, 200, 100, 2
)
T = st.sidebar.slider(
    "Total time (T) in years",
    0.1, 5.0, 1.0, 0.1
)
M = st.sidebar.slider(
    "Number of time steps (M)",
    100, 20000, 10000, 100
)
mu = st.sidebar.slider(
    "Drift coefficient (μ)",
    -0.1, 0.5, 0.05, 0.01
)
sigma = st.sidebar.slider(
    "Volatility (σ)",
    0.01, 1.0, 0.06, 0.01
)
x0 = st.sidebar.slider(
    "Initial asset price (x0)",
    x_min, x_max, 95.0, 1.0
)

# Calculate spatial and time steps
if N < 2:
    st.error("Number of spatial points (N) must be at least 2.")
else:
    dx = (x_max - x_min) / (N - 1)
    dt = T / M

    # Check stability condition for diffusion term
    if sigma != 0:
      dt_max = 0.5 * dx**2 / (sigma**2 * x_max**2)
      if dt > dt_max:
          st.warning(
                f"Stability condition violation: dt={dt:.5f} exceeds dt_max={dt_max:.5f}.\n"
            "Simulation may be unstable. Adjust M or σ."
            )
    # Define spatial grid
    x = np.linspace(x_min, x_max, N)

    # Initial condition (normalized delta function)
    pdf = np.zeros(N)
    idx = np.abs(x - x0).argmin()
    pdf[idx] = 1 / dx  # Delta function with area 1

    # Time-stepping loop
    try:
        for _ in range(M):
            # Flux (drift term): upwind at boundaries
            flux = np.zeros(N)
            flux[1:-1] = (mu * x[2:] * pdf[2:] - mu * x[:-2] * pdf[:-2]) / (2 * dx)
            flux[0] = (mu * x[1] * pdf[1]) / dx
            flux[-1] = -(mu * x[-2] * pdf[-2]) / dx

            # Diffusion term: central differences
            diffusion = np.zeros(N)
            diffusion[1:-1] = (sigma**2 * x[2:]**2 * pdf[2:] - 2 * sigma**2 * x[1:-1]**2 * pdf[1:-1] + sigma**2 * x[:-2]**2 * pdf[:-2]) / (2 * dx**2)
            diffusion[0] = (sigma**2 * x[1]**2 * pdf[1] - 2 * sigma**2 * x[0]**2 * pdf[0]) / (2 * dx**2)
            diffusion[-1] = (sigma**2 * x[-2]**2 * pdf[-2] - 2 * sigma**2 * x[-1]**2 * pdf[-1]) / (2 * dx**2)

            # Update PDF using explicit Euler
            pdf += dt * (-flux + diffusion)

        # Plot results
        fig, ax = plt.subplots()
        ax.plot(x, pdf)
        ax.set_xlabel('Asset Price')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'T={T} years | μ={mu} | σ={sigma}')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Simulation error: {str(e)}")

# Add explanation section
st.subheader("Graph Explanation")

# Construct user-friendly explanation
explanation = f"""
- **Starting Price**: ${x0:.2f} (concentrated at t=0)
- **Drift ({mu*100:.1f}% Annual)**: The distribution shifts {'' if mu > 0 else 'down'} to the {'' if mu < 0 else 'right'}, indicating an {'' if mu == 0 else 'upward'} trend in average price.
- **Volatility ({sigma*100:.1f}% Annual)**: Higher volatility broadens the distribution, representing increased uncertainty and price fluctuations.
- **Time Frame ({T:.1f} Years)**: Over time, the distribution widens, showing how uncertainty accumulates as {'' if T <= 0.5 else 'years pass'}.
- **Reflecting Boundaries**: Prices are kept between ${x_min:.0f} and ${x_max:.0f}, causing the distribution to {'' if (np.max(pdf) - pdf[0]) < 0.5 else 'accumulate'} at the edges if prices approach these limits.
"""

st.markdown(explanation)

# Add information box
st.info(
    """
    This app models the probability density of asset prices using the Fokker-Planck equation.
    Reflecting boundary conditions are applied at spatial limits.
    """
)