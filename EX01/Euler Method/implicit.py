import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
# ODE to be solved:
# (dy/dt) = -4*(y-2) ; y(t = 0) = 1 
# t0 = 0, y0 = 1

def func(yn1,yn, h):
    return yn1 - yn - h*(-4*(yn1 - 2))
 
def main(h, iter_limit = 50):

    y_values = np.zeros(iter_limit)

    y_values[0] = 1

    for i in range(1,iter_limit):

        yn = y_values[i-1]
        
        y_values[i] = fsolve(func,yn,args = (yn,h))[0]

    return y_values


if __name__ == '__main__':

    step_size = [0.4,0.2,0.05]

    y_values_dict = {}

    for h in step_size:

        y_values_dict[h] = main(h)

    for h, y_values in y_values_dict.items():
        t_values = [i for i in range(len(y_values))]
        plt.plot(t_values, y_values, label=f'h = {h}')

    plt.xlabel('steps')
    plt.ylabel('y')
    plt.title('Euler Method (Implicit)')
    plt.legend()
    plt.grid(True)
    plt.show()

