import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def func(t,y):
    return -4*(y-2)


def main(method,iter_limit = 50):

    solution = solve_ivp(func, method=method,y0=[1],t_span=[0,iter_limit], )

    print(solution.y[0])

    return solution




if __name__ == '__main__':

    methods = ['RK45', 'BDF']

    y_values_dict = {}

    for method in methods:
        y_values_dict[method] = main(method=method)

    for h, y_values in y_values_dict.items():
        t_values = [i for i in range(len(y_values.y[0]))]
        plt.plot(t_values, y_values.y[0], label=f'method = {h}')

    plt.xlabel('steps')
    plt.ylabel('y')
    plt.title(f'Solve_ivp')
    plt.legend()
    plt.grid(True)
    plt.show()