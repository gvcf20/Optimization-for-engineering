import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ODE to be solved:
# dy/dt = -4*(y-2), with initial condition y(t=0) = 1

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

def func(yn1, yn, h):
    """
    Defines the function for the implicit Euler method.

    Args:
        yn1 (float): Current value to be solved.
        yn (float): Previous known value.
        h (float): Step size.

    Returns:
        float: The value of the function based on the implicit formula.
    """
    return yn1 - yn - h * (-4 * (yn1 - 2))


def main(h, iter_limit=50):
    """
    Solves the ODE using the Implicit Euler Method.

    Args:
        h (float): Step size.
        iter_limit (int): Number of iterations.

    Returns:
        np.array: Array containing the approximated y values.
    """
    y_values = np.zeros(iter_limit)
    y_values[0] = 1

    for i in range(1, iter_limit):
        yn = y_values[i-1]
        y_values[i] = fsolve(func, yn, args=(yn, h))[0]  # Solve for the next value using fsolve

    return y_values


if __name__ == '__main__':
    step_sizes = [0.4, 0.2, 0.05]  # Different step sizes to test

    y_values_dict = {}

    for h in step_sizes:  # Loop over different step sizes
        y_values_dict[h] = main(h)

    for h, y_values in y_values_dict.items():  # Plotting the curves for different step sizes
        t_values = [i for i in range(len(y_values))]
        plt.plot(t_values, y_values, label=f'h = {h}')

    plt.xlabel('Steps')
    plt.ylabel('y')
    plt.title('Euler Method (Implicit)')
    plt.legend()
    plt.grid(True)
    plt.show()
