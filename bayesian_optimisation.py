import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import time

# Define the hyperparameter space (6 dimensions in this case)
# Each Real space defines the lower and upper bounds for a hyperparameter.
param_space = [
    Real(0.1, 1.0, name='learning_rate'),
    Real(16, 128, name='batch_size', prior='log-uniform'), # Integer-like, but skopt handles it
    Real(0.8, 0.99, name='momentum'),
    Real(0.0001, 0.01, name='weight_decay', prior='log-uniform'),
    Real(1, 5, name='num_layers', prior='uniform'), # Integer-like
    Real(32, 512, name='hidden_units_layer1', prior='log-uniform') # Integer-like
]

# Define your black-box function (e.g., training a neural network and returning validation loss)
# This is a placeholder. In a real scenario, this function would
# take hyperparameters, build/train a model, and return a score to minimize.
# For this example, let's create a synthetic function.
@use_named_args(param_space)
def black_box_function(learning_rate, batch_size, momentum, weight_decay, num_layers, hidden_units_layer1):
    """
    This is your black-box function.
    It takes hyperparameters as input and returns the value to be minimized (e.g., loss).
    """
    print(f"Evaluating with: LR={learning_rate:.4f}, Batch={int(batch_size)}, Momentum={momentum:.3f}, "
          f"WD={weight_decay:.5f}, Layers={int(num_layers)}, Hidden1={int(hidden_units_layer1)}")

    # Simulate some computation time
    time.sleep(0.5)

    # Example synthetic objective function (replace with your actual model training)
    # We want to minimize this value.
    # A simple function for demonstration:
    val = (learning_rate - 0.2)**2 + \
          (np.log(batch_size) - np.log(32))**2 + \
          (momentum - 0.9)**2 + \
          (np.log(weight_decay) - np.log(0.001))**2 + \
          (int(num_layers) - 3)**2 + \
          (np.log(int(hidden_units_layer1)) - np.log(128))**2

    # Add some noise to make it more realistic for optimization
    val += np.random.randn() * 0.01

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