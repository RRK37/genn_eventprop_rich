

import numpy as np
import matplotlib.pyplot as plt

def plot_stdp_curve():
    """
    Plots a Spike-Timing-Dependent Plasticity (STDP) curve.
    The STDP rule describes how the strength of a synapse changes based on the
    relative timing of pre-synaptic and post-synaptic spikes.
    LTP (Long-Term Potentiation) occurs if the pre-synaptic spike precedes
    the post-synaptic spike.
    LTD (Long-Term Depression) occurs if the post-synaptic spike precedes
    the pre-synaptic spike.
    """

    # Parameters for the STDP model
    # A_p: amplitude for LTP (potentiation)
    # tau_p: time constant for LTP
    # A_d: amplitude for LTD (depression)
    # tau_d: time constant for LTD
    A_p = 0.8
    tau_p = 20.0  # ms
    A_d = 0.45 # Adjusted to better match the provided graph's LTD depth
    tau_d = 20.0  # ms

    # Time differences (delta_t = t_pre - t_post)
    # For LTP (pre-synaptic spike occurs before post-synaptic spike: delta_t < 0)
    delta_t_ltp = np.linspace(-80, 0, 200)  # Time window for LTP
    # For LTD (post-synaptic spike occurs before pre-synaptic spike: delta_t > 0)
    delta_t_ltd = np.linspace(0, 80, 200)  # Time window for LTD

    # Calculate synaptic strength change (dw)
    # dw_ltp: change in weight for LTP
    dw_ltp = A_p * np.exp(delta_t_ltp / tau_p)
    # dw_ltd: change in weight for LTD
    dw_ltd = -A_d * np.exp(-delta_t_ltd / tau_d)

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for better aesthetics
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot LTP curve
    ax.plot(delta_t_ltp, dw_ltp, color='red', linewidth=2.5, label='LTP (Pre before Post)')
    # Plot LTD curve
    ax.plot(delta_t_ltd, dw_ltd, color='blue', linewidth=2.5, label='LTD (Post before Pre)')

    # Add labels and title
    ax.set_xlabel("Time(Pre)-Time(Post) [msec]", fontsize=12)
    ax.set_ylabel("Relative Change Synaptic Strength", fontsize=12)
    ax.set_title("Spike-Timing-Dependent Plasticity (STDP)", fontsize=14, fontweight='bold')

    # Add horizontal and vertical lines at zero
    ax.axhline(0, color='black', linestyle=':', linewidth=1) # Dashed line for y=0
    ax.axvline(0, color='black', linestyle=':', linewidth=1) # Dashed line for x=0

    # Set axis limits and ticks
    ax.set_xlim([-80, 80])
    ax.set_ylim([-0.6, 1.0]) # Adjusted y-limits to match the provided graph
    ax.set_xticks([-80, -40, 0, 40, 80])
    ax.set_yticks([-0.5, 0, 0.5, 1.0]) # Adjusted y-ticks to match

    # Add text annotations
    # LTP annotations
    ax.text(-55, 0.65, "LTP", color='red', fontsize=14, fontweight='bold')
    ax.text(-65, 0.50, '"pre before post"', color='red', fontsize=11, style='italic')

    # LTD annotations
    ax.text(35, -0.3, "LTD", color='blue', fontsize=14, fontweight='bold')
    ax.text(25, -0.45, '"post before pre"', color='blue', fontsize=11, style='italic')

    # Improve layout and display grid
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

if __name__ == '__main__':
    plot_stdp_curve()
