import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import time
from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from os.path import exists

def find_best_loss(path):
    
    filepath = path + "/test_results.txt"

    max_eval_loss = -float('inf')  # Initialize with negative infinity to ensure any number is larger
    found_data = False  # Flag to check if any valid data was processed

    with open(filepath, 'r') as file:
        for line_number, line in enumerate(file, 1):
            stripped_line = line.strip()  # Remove leading/trailing whitespace

            if not stripped_line:  # Skip empty lines
                continue

            parts = stripped_line.split()

            # The eval_loss is the fourth number on the line (index 3)
            if len(parts) > 3:
                try:
                    eval_loss_str = parts[3]
                    eval_loss = float(eval_loss_str)

                    if eval_loss > max_eval_loss:
                        max_eval_loss = eval_loss
                    found_data = True
                except ValueError:
                    print(f"Warning: Could not convert '{eval_loss_str}' to float on line {line_number}. Skipping this line.")
                except IndexError:
                    # This should ideally be caught by len(parts) > 3,
                    # but it's a good safeguard.
                    print(f"Warning: Line {line_number} does not have enough columns: '{stripped_line}'. Skipping this line.")
            else:
                print(f"Warning: Line {line_number} is too short to contain eval_loss: '{stripped_line}'. Skipping this line.")

    if not found_data:
        print(f"No valid data found in '{filepath}' to determine the maximum eval_loss.")
    elif max_eval_loss == -float('inf'):
        # This case implies found_data might be true but no valid eval_loss was parsed,
        # though the current logic should set found_data to True only if an eval_loss is processed.
        print(f"No valid eval_loss values were found in '{filepath}'.")
    else:
        print(f"The biggest eval_loss in '{filepath}' is: {max_eval_loss}")
        return max_eval_loss



# Define the hyperparameter space (6 dimensions in this case)
# Each Real space defines the lower and upper bounds for a hyperparameter.
param_space = [
    Real(16.0, 24.0, name='t_mem')
]

# Define your black-box function (e.g., training a neural network and returning validation loss)
# This is a placeholder. In a real scenario, this function would
# take hyperparameters, build/train a model, and return a score to minimize.
# For this example, let's create a synthetic function.
@use_named_args(param_space)
def black_box_function(t_mem, t_syn, eta, f_shift, n_delay, t_delay, glb_upper):
    """
    This is your black-box function.
    It takes hyperparameters as input and returns the value to be minimized (e.g., loss).
    """

    # Training Setup
    p["EVALUATION"]             = "speaker"
    p["N_EPOCH"]                = 100
    p["BALANCE_TRAIN_CLASSES"]  = True
    p["BALANCE_EVAL_CLASSES"]   = True
    p["TRAIN_DATA_SEED"]        = 321
    p["TEST_DATA_SEED"]         = 123
    p["TRIAL_MS"]               = 1000.0
    p["AUGMENTATION"]= {
        "NORMALISE_SPIKE_NUMBER": True,
        "random_shift": 40.0,
        "blend": [0.5,0.5]
    }
    p["N_INPUT_DELAY"]          = 10
    p["INPUT_DELAY"]            = 30


    # Learning parameters
    p["ETA"]                    = 0.001
    p["MIN_EPOCH_ETA_FIXED"]    = 300
    p["LOSS_TYPE"]              = "sum_weigh_exp"
    p["TAU_0"]                  = 1
    p["TAU_1"]                  = 100
    p["ALPHA"]                  = 5*10^(-5)
    p["GLB_UPPER"]              = 1e-9


    # Network parameters
    p["REG_TYPE"]               = "simple"
    p["TAU_MEM"]                = 20
    p["TAU_SYN"]                = 5
    p["N_HID_LAYER"]            = 1
    p["NUM_HIDDEN"]             = 1024
    p["RECURRENT"]              = True
    p["INPUT_HIDDEN_MEAN"]      = 0.03
    p["INPUT_HIDDEN_STD"]       = 0.01
    p["HIDDEN_HIDDEN_MEAN"]     = 0
    p["HIDDEN_HIDDEN_STD"]      = 0.02 
    p["HIDDEN_OUTPUT_MEAN"]     = 0 
    p["HIDDEN_OUTPUT_STD"]      = 0.03
    p["PDROP_INPUT"]            = 0
    p["NU_UPPER"]               = 14

    p["OUT_DIR"]                = "hyperparameter_optimisation"

    p["BUILD"] = True

    # Make a new directory
    os.mkdir(p["OUT_DIR"])

    mn= SHD_model(p)
    spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)

    # Read the created test_results.txt file and return the max eval accuracy
    val = find_best_loss(p["OUT_DIR"])

    #  Delete the new repository
    os.rmdir(p["OUT_DIR"])

    print(f"Returned loss: {val:.4f}")
    return val

if __name__ == '__main__':
    print("Starting Bayesian Optimization...")

    # Number of evaluations
    n_calls = 30 # Total number of times to call the black_box_function

    # Perform Bayesian optimization
    # gp_minimize uses Gaussian Processes as the surrogate model.
    # 'n_initial_points' specifies how many random points to evaluate before fitting the GP.
    result = gp_minimize(
        func=black_box_function,      # the function to minimize
        dimensions=param_space,       # the bounds on each dimension of x
        acq_func="EI",                # the acquisition function (Expected Improvement)
        n_calls=n_calls,              # the number of evaluations of f
        n_initial_points=10,          # the number of random initialization points
        random_state=123,             # for reproducibility
        verbose=True
    )

    print("\nOptimization Finished.")
    print(f"Best score achieved: {result.fun:.4f}")
    print("Best parameters found:")
    param_names = [dim.name for dim in param_space]
    best_params = dict(zip(param_names, result.x))
    for name, value in best_params.items():
        # Handle potentially float values for integer-like parameters
        if name in ['batch_size', 'num_layers', 'hidden_units_layer1']:
            print(f"  {name}: {int(round(value))}")
        else:
            print(f"  {name}: {value:.5f}")

    # You can access all evaluated points and their scores:
    # print("\nAll evaluations:")
    # for x_eval, y_eval in zip(result.x_iters, result.func_vals):
    #     print(f"  Params: {dict(zip(param_names, x_eval))}, Score: {y_eval:.4f}")

    # For more insights, you can plot the convergence:
    try:
        from skopt.plots import plot_convergence
        import matplotlib.pyplot as plt
        plot_convergence(result)
        plt.title("Convergence Plot")
        plt.xlabel("Number of Calls")
        plt.ylabel("Minimum Objective Value Found")
        plt.show()
    except ImportError:
        print("\nInstall matplotlib to see the convergence plot: pip install matplotlib")