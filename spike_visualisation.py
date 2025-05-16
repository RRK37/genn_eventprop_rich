# import numpy as np


# file_to_load = "z_bell_loss_0_-02_44_04/test_hidden0_spike_ID.npy" 

# data = np.load(file_to_load)

# print((data))


# for i in range(len(data)):
#     print(data[i])




import numpy as np
import matplotlib.pyplot as plt

def create_raster_plot(spike_times_file, spike_ids_file, title="Spike Raster Plot", output_filename="raster_plot.png"):
    """
    Creates and saves a spike raster plot from .npy files.

    Args:
        spike_times_file (str): Path to the .npy file containing spike timestamps.
        spike_ids_file (str): Path to the .npy file containing neuron IDs for spikes.
        title (str): The title for the plot.
        output_filename (str): The filename to save the plot (e.g., 'raster_plot.png').
    """
    try:
        # Load the spike data
        # test_hidden0_spike_t.npy contains the timestamps for each spike
        spike_times = np.load(spike_times_file)
        # test_hidden0_spike_ID.npy contains the ID of the neuron that spiked
        spike_ids = np.load(spike_ids_file)

        # Ensure both arrays have the same length, as they correspond to each other
        if len(spike_times) != len(spike_ids):
            print("Error: Spike times and spike IDs arrays must have the same length.")
            print(f"Length of spike_times: {len(spike_times)}")
            print(f"Length of spike_ids: {len(spike_ids)}")
            return

        if len(spike_times) == 0:
            print("Warning: No spike data found in the files. Plot will be empty.")
            # Still create an empty plot for consistency if desired
            # Or you could return here if an empty plot is not useful

        # Create the plot
        plt.figure(figsize=(12, 8)) # Adjust figure size as needed

        # Use a scatter plot for the raster plot.
        # 's' controls the marker size.
        # 'marker' can be changed (e.g., '|' for vertical lines, but '.' is common for dots)
        plt.scatter(spike_times, spike_ids, s=5, marker='.', c='black', alpha=0.7)

        # Set plot labels and title
        plt.xlabel("Time (simulation steps or ms)") # Adjust unit if known
        plt.ylabel("Neuron ID")
        plt.title(title)

        # Optional: Set limits for axes if you know the range or want to zoom
        # if len(spike_times) > 0:
        #     plt.xlim(spike_times.min() - 100, spike_times.max() + 100) # Add some padding
        # if len(spike_ids) > 0:
        #     plt.ylim(spike_ids.min() - 1, spike_ids.max() + 1)     # Add some padding

        # Improve layout
        plt.tight_layout()

        # Show grid (optional)
        plt.grid(True, linestyle=':', alpha=0.5)

        # Save the plot to a file
        plt.savefig(output_filename)
        print(f"Raster plot saved as '{output_filename}'")

        # Display the plot (optional, comment out if running in a non-GUI environment)
        plt.show()

    except FileNotFoundError:
        print(f"Error: One or both data files not found.")
        print(f"Tried to load: '{spike_times_file}' and '{spike_ids_file}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the paths to your .npy files
    # Ensure these files are in the same directory as the script, or provide the full path.
    times_file = "z_experimental_rec/test_input_spike_t.npy"
    ids_file = "z_experimental_rec/test_input_spike_ID.npy"

    create_raster_plot(times_file, ids_file, title="Raster Plot for Hidden Layer 0")