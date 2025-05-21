import matplotlib.pyplot as plt
import numpy as np

from b import F

def equidistance_search(Ti, Tf, num_intervals = 25, plot = True):

    T_values = np.linspace(Ti, Tf, num_intervals + 1)
    F_values = [-F(T) for T in T_values]

    if plot == True:

        plt.figure(figsize=(10, 5))
        plt.plot(T_values, F_values, marker='o')
        plt.title("Objective Function F(T) over Temperature")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Objective Function F(T) [€/s]")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    return F_values 


if __name__ == '__main__':

    a = equidistance_search(Ti=300, Tf=380)
    