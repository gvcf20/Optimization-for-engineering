import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

def func(t, y):
    """
    Defines the ODE function.

    Args:
        t (float): Independent variable (time).
        y (float): Dependent variable.

    Returns:
        float: Value of the derivative dy/dt.
    """
    return -4 * (y - 2)


def main(method, iter_limit=50):
    """
    Solves the ODE using the specified method with solve_ivp.

    Args:
        method (str): Integration method to use ('RK45', 'BDF', etc.).
        iter_limit (int): Time span limit for the solver.

    Returns:
        OdeResult: Object containing the solution.
    """
    solution = solve_ivp(func, method=method, y0=[1], t_span=[0, iter_limit])

    return solution


if __name__ == '__main__':
    methods = ['RK45', 'BDF']  # Methods to compare

    y_values_dict = {}

    for method in methods:  # Loop over different methods
        y_values_dict[method] = main(method=method)

    for method, solution in y_values_dict.items():  # Plotting the solutions for different methods
        t_values = [i for i in range(len(solution.y[0]))]
        plt.plot(t_values, solution.y[0], label=f'Method = {method}')

    plt.xlabel('Steps')
    plt.ylabel('y')
    plt.title('solve_ivp Methods Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
