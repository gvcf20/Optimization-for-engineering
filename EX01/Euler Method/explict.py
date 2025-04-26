import numpy as np
import matplotlib.pyplot as plt

# ODE to be solved:
# dy/dt = -4*(y-2), with initial condition y(t=0) = 1

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

def main(h, iter_limit=50):
    """
    Solves the ODE using the Explicit Euler Method.

    Args:
        h (float): Step size.
        iter_limit (int): Number of iterations.

    Returns:
        np.array: Array containing the approximated y values.
    """
    y_values = np.zeros(iter_limit)
    y_values[0] = 1

    for i in range(1, iter_limit):  # Loop for the explicit method
        y_values[i] = y_values[i-1] + h * (-4 * (y_values[i-1] - 2))

    return y_values

if __name__ == '__main__':
    step_sizes = [0.4, 0.2, 0.05]  # Different step sizes to test

    y_values_dict = {}

    for h in step_sizes:  # Loop over different step sizes
        y_values_dict[h] = main(h)

    for h, y_values in y_values_dict.items():  # Loop for plotting the curves for different step sizes
        t_values = [i for i in range(len(y_values))]
        plt.plot(t_values, y_values, label=f'h = {h}')

    plt.xlabel('Steps')
    plt.ylabel('y')
    plt.title('Euler Method (Explicit)')
    plt.legend()
    plt.grid(True)
    plt.show()