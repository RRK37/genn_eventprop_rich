import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Directory where your .npy files are stored.
# Example: "/path/to/your/weights/" or "." for the current directory.
DATA_DIRECTORY = "."

# File naming pattern. {} will be replaced by the epoch number.
FILE_PATTERN = "z_bell_loss_0_-02_44_04/test_w_hidden_output_{}.npy"

EPOCH_START = 0
EPOCH_END = 298  # Inclusive
EPOCH_STEP = 2

# Colormap for the plot (e.g., 'viridis', 'coolwarm', 'RdBu_r', 'inferno')
COLORMAP = 'coolwarm'
# ---------------------

def plot_weight_evolution():
    """
    Loads network weights from .npy files over epochs and plots their
    evolution as a heatmap.
    """
    epochs_to_try = list(range(EPOCH_START, EPOCH_END + 1, EPOCH_STEP))
    
    loaded_weights_list = []
    loaded_epochs = []
    num_weights = None

    print(f"Attempting to load weights from epoch {EPOCH_START} to {EPOCH_END} with step {EPOCH_STEP}...")

    for epoch in epochs_to_try:
        filename = FILE_PATTERN.format(epoch)
        filepath = os.path.join(DATA_DIRECTORY, filename)
        print(filepath)
        if not os.path.exists(filepath):
            # print(f"Info: File not found {filepath}. Skipping this epoch.")
            continue

        try:
            weights_at_epoch = np.load(filepath)
            
            # Ensure weights are float32, as specified
            if weights_at_epoch.dtype != np.float32:
                print(f"Warning: Weights in {filename} are {weights_at_epoch.dtype}, converting to float32.")
                weights_at_epoch = weights_at_epoch.astype(np.float32)

            # Flatten the weights array if it's not 1D.
            # The "neuron index" will then be the index in this flattened array.
            current_weights_flat = weights_at_epoch.flatten()

            if num_weights is None:
                num_weights = current_weights_flat.shape[0]
                if num_weights == 0:
                    print(f"Error: No weights found in the first successfully loaded file: {filename}.")
                    return
                print(f"Determined number of weights per epoch: {num_weights} (from {filename})")

            if current_weights_flat.shape[0] == num_weights:
                loaded_weights_list.append(current_weights_flat)
                loaded_epochs.append(epoch)
            else:
                print(f"Warning: Shape mismatch for {filename}. "
                      f"Expected {num_weights} weights, got {current_weights_flat.shape[0]}. Skipping this epoch.")
        
        except Exception as e:
            print(f"Warning: Could not load or process {filepath}: {e}. Skipping this epoch.")

    if not loaded_weights_list:
        print("Error: No weight files were successfully loaded. Cannot generate plot.")
        print(f"Please check DATA_DIRECTORY ('{DATA_DIRECTORY}') and FILE_PATTERN ('{FILE_PATTERN}').")
        return

    # Convert list of 1D arrays to a 2D array (num_weights x num_loaded_epochs)
    # Each column is the flattened weight vector for a given epoch.
    all_weights_evolution = np.array(loaded_weights_list).T  # Transpose to have weights as rows, epochs as columns

    print(f"Successfully loaded data for {len(loaded_epochs)} epochs.")
    print(f"Shape of the data matrix for plotting (num_weights, num_epochs): {all_weights_evolution.shape}")

    # --- Plotting ---
    plt.figure(figsize=(15, 10)) # Adjust size as needed

    # Using imshow. The matrix all_weights_evolution has shape (num_weights, num_loaded_epochs)
    # imshow(X) plots the data in X.
    # x-axis will correspond to columns (epochs), y-axis to rows (weight index).
    plt.imshow(all_weights_evolution, aspect='auto', interpolation='nearest', cmap=COLORMAP)
    
    plt.colorbar(label='Weight Value')

    # Set x-axis: epoch
    # The columns 0, 1, ..., N-1 of all_weights_evolution correspond to loaded_epochs[0], loaded_epochs[1], ...
    tick_positions_x = np.arange(len(loaded_epochs))
    
    # Show a reasonable number of ticks on x-axis
    num_x_ticks = 10 # Adjust as needed
    if len(loaded_epochs) > num_x_ticks:
        step_x = max(1, len(loaded_epochs) // num_x_ticks) # Ensure step is at least 1
        selected_tick_indices_x = tick_positions_x[::step_x]
        selected_tick_labels_x = [loaded_epochs[i] for i in selected_tick_indices_x]
    else:
        selected_tick_indices_x = tick_positions_x
        selected_tick_labels_x = loaded_epochs
    
    plt.xticks(ticks=selected_tick_indices_x, labels=selected_tick_labels_x)
    plt.xlabel('Epoch')

    # Set y-axis: "Neuron Index" (more accurately, Weight Index if flattened)
    # The rows 0, 1, ..., M-1 of all_weights_evolution correspond to weight indices.
    plt.ylabel('Weight Index (in flattened array)')
    
    # Optionally, adjust y-axis ticks if num_weights is very large
    num_y_ticks = 15 # Adjust as needed
    if num_weights > num_y_ticks:
        step_y = max(1, num_weights // num_y_ticks) # Ensure step is at least 1
        selected_tick_indices_y = np.arange(0, num_weights, step_y)
        plt.yticks(ticks=selected_tick_indices_y)
    # If num_weights is small, default y-ticks are usually fine.

    plt.title(f'Evolution of Network Weights Over Epochs\n(Layer: {FILE_PATTERN.replace("_{}.npy","")})')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    # Save the plot (optional)
    plot_filename = "weights_evolution.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as {plot_filename}")
    
    plt.show()

if __name__ == '__main__':
    # **IMPORTANT:**
    # 1. Modify `DATA_DIRECTORY` above if your files are not in the same
    #    directory as this script.
    # 2. Ensure `FILE_PATTERN` correctly matches your filenames.
    # 3. Ensure `EPOCH_START`, `EPOCH_END`, and `EPOCH_STEP` are correct.
    
    plot_weight_evolution()