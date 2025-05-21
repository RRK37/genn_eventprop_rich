import matplotlib.pyplot as plt
import pandas as pd
import os

def load_and_process_data(file_path, method_name):
    """
    Loads data from a single file, extracts epoch and evaluation correct,
    and returns a DataFrame.
    Each row is expected to have epoch as the first number and 
    evaluation correct as the fourth number.
    """
    epochs = []
    eval_corrects = []
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame() # Return an empty DataFrame

    try:
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                parts = line.strip().split()
                # Ensure there are enough parts in the line
                if len(parts) >= 4:
                    try:
                        epoch = int(parts[0])       # First number is epoch
                        eval_correct = float(parts[3]) # Fourth number is evaluation correct
                        epochs.append(epoch)
                        eval_corrects.append(eval_correct)
                    except ValueError:
                        # Print a warning if a line has parts but they can't be converted to int/float
                        if line.strip(): # Avoid warning for completely empty lines
                            print(f"Warning: Could not parse numeric values in line {line_number} of {file_path}: '{line.strip()}'")
                        continue # Skip to the next line
                else:
                    # Print a warning if a line doesn't have enough columns but is not empty
                    if line.strip(): 
                        print(f"Warning: Line {line_number} in {file_path} has insufficient columns: '{line.strip()}'")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame() # Return an empty DataFrame on read error

    # If no data was successfully read, print a warning and return empty DataFrame
    if not epochs:
        print(f"Warning: No valid data processed from {file_path}. File might be empty or all lines unparseable.")
        return pd.DataFrame()

    # Create a DataFrame
    df = pd.DataFrame({
        'epoch': epochs,
        'eval_correct': eval_corrects
    })
    df['method'] = method_name # Add a column for the method name
    return df

# Define file paths for the two training methods
file_path_exponential = "z_exponential_x5/test_results.txt"
file_path_gaussian = "z_gaussian_x5/test_results.txt"

# Load and process data for both methods
df_exponential = load_and_process_data(file_path_exponential, 'Exponential')
df_gaussian = load_and_process_data(file_path_gaussian, 'Gaussian')

plot_generated = False # Flag to track if a plot is generated

# Define the epoch limit
epoch_limit = 100

# Case 1: Both data files are loaded successfully
if not df_exponential.empty and not df_gaussian.empty:
    print(f"Successfully loaded {len(df_exponential)} records for Exponential method.")
    print(f"Successfully loaded {len(df_gaussian)} records for Gaussian method.")
    
    # Combine the data from both methods
    combined_df = pd.concat([df_exponential, df_gaussian], ignore_index=True)
    # Ensure 'epoch' is numeric for correct grouping and plotting
    combined_df['epoch'] = pd.to_numeric(combined_df['epoch'])
    
    # Filter for the first 100 epochs (0-99)
    combined_df = combined_df[combined_df['epoch'] < epoch_limit]
    print(f"Filtered data to include epochs 0-{epoch_limit-1}. Records remaining: {len(combined_df)}")

    if combined_df.empty:
        print(f"No data available for epochs 0-{epoch_limit-1} after filtering.")
    else:
        # Calculate the mean evaluation correct for each epoch and method using filtered data
        df_means = combined_df.groupby(['method', 'epoch'])['eval_correct'].mean().reset_index()

        plt.figure(figsize=(15, 9)) # Create a new figure for plotting

        # Plot data for the Exponential method
        exp_data_points = combined_df[combined_df['method'] == 'Exponential']
        exp_mean_line = df_means[df_means['method'] == 'Exponential']
        
        if not exp_data_points.empty:
            # Scatter plot for individual runs
            plt.scatter(exp_data_points['epoch'], exp_data_points['eval_correct'], 
                        alpha=0.3, s=15, label='Exponential - Individual Runs', color='cornflowerblue')
        if not exp_mean_line.empty:
            # Line plot for the mean
            plt.plot(exp_mean_line['epoch'], exp_mean_line['eval_correct'], 
                     label='Exponential - Mean', color='blue', linewidth=2.5)
        elif not exp_data_points.empty: 
            print("Note: Could not generate mean line for Exponential method (e.g., if epochs are unique per run), but individual points plotted.")

        # Plot data for the Gaussian method
        gauss_data_points = combined_df[combined_df['method'] == 'Gaussian']
        gauss_mean_line = df_means[df_means['method'] == 'Gaussian']
        
        if not gauss_data_points.empty:
            # Scatter plot for individual runs
            plt.scatter(gauss_data_points['epoch'], gauss_data_points['eval_correct'], 
                        alpha=0.3, s=15, label='Gaussian - Individual Runs', color='lightcoral')
        if not gauss_mean_line.empty:
            # Line plot for the mean
            plt.plot(gauss_mean_line['epoch'], gauss_mean_line['eval_correct'], 
                     label='Gaussian - Mean', color='red', linewidth=2.5)
        elif not gauss_data_points.empty:
            print("Note: Could not generate mean line for Gaussian method (e.g., if epochs are unique per run), but individual points plotted.")
            
        plt.title(f'Neural Network Training Performance (Epochs 0-{epoch_limit-1})', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Evaluation Correct', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        if not exp_data_points.empty or not gauss_data_points.empty: 
            plt.legend(loc='best', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.tight_layout() 
        
        plot_filename = f"training_performance_comparison_first_{epoch_limit}_epochs.png"
        plt.savefig(plot_filename) 
        print(f"Plot saved as {plot_filename}")
        plot_generated = True

# Case 2: Only one of the data files is loaded successfully
elif not df_exponential.empty or not df_gaussian.empty:
    single_df = df_exponential if not df_exponential.empty else df_gaussian
    method_name_single = single_df['method'].unique()[0] 
    
    print(f"Successfully loaded {len(single_df)} records for {method_name_single} method.")
    if df_exponential.empty:
        print(f"Note: Data for Exponential method was not found or was empty/unparseable.")
    if df_gaussian.empty:
        print(f"Note: Data for Gaussian method was not found or was empty/unparseable.")
    
    single_df['epoch'] = pd.to_numeric(single_df['epoch'])
    # Filter for the first 100 epochs (0-99)
    single_df = single_df[single_df['epoch'] < epoch_limit]
    print(f"Filtered data for {method_name_single} to include epochs 0-{epoch_limit-1}. Records remaining: {len(single_df)}")
    
    if single_df.empty:
        print(f"No data available for {method_name_single} for epochs 0-{epoch_limit-1} after filtering.")
    else:
        print(f"Plotting only for {method_name_single} method (Epochs 0-{epoch_limit-1}).")
        df_means_single = single_df.groupby(['epoch'])['eval_correct'].mean().reset_index()
        
        plt.figure(figsize=(15, 9)) 
        color_single_scatter = 'cornflowerblue' if method_name_single == 'Exponential' else 'lightcoral'
        color_single_line = 'blue' if method_name_single == 'Exponential' else 'red'

        if not single_df.empty:
            plt.scatter(single_df['epoch'], single_df['eval_correct'], 
                        alpha=0.3, s=15, label=f'{method_name_single} - Individual Runs', color=color_single_scatter)
        if not df_means_single.empty:
            plt.plot(df_means_single['epoch'], df_means_single['eval_correct'], 
                    label=f'{method_name_single} - Mean', linewidth=2.5, color=color_single_line)
        elif not single_df.empty:
            print(f"Note: Could not generate mean line for {method_name_single} method (e.g., if epochs are unique per run), but individual points plotted.")
                 
        plt.title(f'Neural Network Training Performance: {method_name_single} (Epochs 0-{epoch_limit-1})', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Evaluation Correct', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if not single_df.empty: 
            plt.legend(loc='best', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        
        plot_filename = f"training_performance_{method_name_single.lower().replace(' ', '_')}_first_{epoch_limit}_epochs.png"
        plt.savefig(plot_filename) 
        print(f"Plot saved as {plot_filename}")
        plot_generated = True

# Case 3: Neither data file is loaded successfully
else:
    print("Error: Both data files were not found or contained no processable data. No plot will be generated.")

if not plot_generated:
    print(f"No plot was generated. This could be due to missing files, unparseable data, or no data within the first {epoch_limit} epochs.")
