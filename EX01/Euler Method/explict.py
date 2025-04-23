import numpy as np
import matplotlib.pyplot as plt

# ODE to be solved:
# (dy/dt) = -4*(y-2) ; y(t = 0) = 1 
# t0 = 0, y0 = 1
def main(h, iter_limit = 50):

    y_values = np.zeros(iter_limit)

    y_values[0] = 1
    
    for i in range(1,iter_limit):

        y_values[i] = y_values[i-1] + h*(-4*(y_values[i-1]))
    

    print(y_values)

    return y_values


if __name__ == '__main__':

    step_sizes = [0.4,0.2,0.05]

    y_values_dict = {}

    for h in step_sizes:

        y_values_dict[h] = main(h)

    for h, y_values in y_values_dict.items():
        t_values = [i for i in range(len(y_values))]
        plt.plot(t_values, y_values, label=f'h = {h}')

    plt.xlabel('steps')
    plt.ylabel('y')
    plt.title('Euler Method (Explicit)')
    plt.legend()
    plt.grid(True)
    plt.show()