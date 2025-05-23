import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Import numpy for arange
import matplotlib.style as style
import math # For ceiling and floor functions

def plot_training_results(file_path):
    """
    Reads neural network training results from a file and plots
    epoch vs. evaluation correct with enhanced styling and custom y-axis ticks.

    Args:
        file_path (str): The path to the data file.
                         Each row should contain space-separated values,
                         with epoch as the first number and evaluation
                         correct as the fourth number.
    """
    try:
        # --- Apply a style ---
        style.use('seaborn-v0_8-whitegrid')

        # Load the data from the text file
        data = pd.read_csv(
            file_path,
            sep="\\s+",
            header=None,
            usecols=[0, 3],
            names=["epoch", "evaluation_correct"],
            comment='#',
            engine='python'
        )

        if data.empty:
            print(f"No data found in the file: {file_path}")
            print("Please ensure the file is not empty and is formatted correctly.")
            return

        data['epoch'] = pd.to_numeric(data['epoch'], errors='coerce')
        data['evaluation_correct'] = pd.to_numeric(data['evaluation_correct'], errors='coerce')

        original_rows = len(data)
        data.dropna(subset=['epoch', 'evaluation_correct'], inplace=True)
        if len(data) < original_rows:
            print(f"Warning: {original_rows - len(data)} rows with non-numeric data were removed.")

        if data.empty:
            print("No valid numeric data to plot after cleaning.")
            return

        # --- Create the plot ---
        plt.figure(figsize=(12, 7))

        plt.plot(
            data["epoch"],
            data["evaluation_correct"],
            marker="o",
            markersize=4,
            linestyle='-',
            linewidth=1.5,
            color='#4A90E2',
            label="Evaluation Correctness"
        )

        # --- Customize Title and Labels ---
        plt.title("Training Progress - Gaussian Based Weighting", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Epoch Number", fontsize=14, labelpad=15)
        plt.ylabel("Evaluation Correctness", fontsize=14, labelpad=15)

        # --- Y-axis Tick Customization ---
        # Determine y-axis limits dynamically for better presentation of 0.1 ticks
        min_y = data["evaluation_correct"].min()
        max_y = data["evaluation_correct"].max()

        # Adjust min and max to be multiples of 0.1 for cleaner tick ranges
        # Add a small buffer as well
        y_tick_min = math.floor(min_y * 10) / 10
        y_tick_max = math.ceil(max_y * 10) / 10

        # Ensure there's some space above the max point
        if y_tick_max <= max_y:
            y_tick_max += 0.1
        # Ensure there's some space below the min point if it's not 0
        if y_tick_min >= min_y and y_tick_min > 0:
             y_tick_min -=0.1
             if y_tick_min < 0: y_tick_min = 0 # don't go below 0

        # Set y-axis limits and ticks
        plt.ylim(y_tick_min, y_tick_max)
        plt.yticks(np.arange(y_tick_min, y_tick_max + 0.01, 0.1)) # +0.01 to include the upper limit if it's a multiple of 0.1

        # --- Grid Customization ---
        plt.grid(True, axis='y', linestyle=':', linewidth=0.7, color='gray')
        plt.grid(False, axis='x')

        # --- Tick Parameters ---
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12) # Y-tick labels will now reflect the 0.1 interval

        # --- Legend ---
        plt.legend(fontsize=12, loc='lower right')

        # --- Spine Customization ---
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')

        plt.tight_layout(pad=1.5)

        output_filename = "epoch_vs_evaluation_correct_styled_ticks.png"
        plt.savefig(output_filename, dpi=300)
        print(f"\nPlot successfully generated and saved as '{output_filename}'")
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure the file path is correct and the file exists in that location.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty or does not contain parsable data.")
    except ValueError as ve:
        print(f"ValueError: There might be an issue with data conversion. {ve}")
        print("Please check if all relevant columns contain numeric data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(
            "Please ensure the file is formatted as expected (space-separated values,"
            " with at least 4 columns, where the 1st and 4th are numeric)."
        )

if __name__ == "__main__":
    # IMPORTANT: Replace this with the actual path to your file
    file_to_plot = "z_gaussian_x5/test_results2.txt"
    
    # Example for testing if the file is in the same directory as the script:
    # import os
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # file_to_plot = os.path.join(script_dir, "table_5_values", "test_results2.txt")

    plot_training_results(file_to_plot)
