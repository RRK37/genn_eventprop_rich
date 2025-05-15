filepath = "z_bell_loss_0_-02_44_04/test_results.txt"

max_eval_loss = -float('inf')  # Initialize with negative infinity to ensure any number is larger
found_data = False  # Flag to check if any valid data was processed

try:
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

except FileNotFoundError:
    print(f"Error: The file '{filepath}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


