import matplotlib.pyplot as plt
import numpy as np

from EX07.b import F

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

'''
c) Perform an equidistance search in the interval of T = [300 K ... 380 K]. Divide the 
interval into 25 equal sections and plot the course of the objective function. 
'''

def equidistance_search(Ti, Tf, num_intervals=25, plot=True, print_solution=True):
    # Create temperature values equally spaced in the interval [Ti, Tf]
    T_values = np.linspace(Ti, Tf, num_intervals + 1)

    # Evaluate the objective function F at each temperature value
    F_values = [F(T) for T in T_values]

    # Plot the values if enabled
    if plot == True:
        plt.figure(figsize=(10, 5))
        plt.plot(T_values, F_values, marker='o')
        plt.title("Objective Function F(T) over Temperature")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Objective Function F(T) [€/s]")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Find the maximum revenue and its corresponding temperature
    max_revenue = max(F_values)
    max_index = F_values.index(max_revenue)
    max_temperature = T_values[max_index]

    # Estimate the uncertainty interval (2 × step size)
    uncertainty_interval = 2 * (Tf - Ti) / (num_intervals + 1)

    # Print solution details if enabled
    if print_solution == True:
        print(f'Max Revenue = {max_revenue} at temperature = {max_temperature}')
        print(f'Uncertainty interval = {uncertainty_interval} K')

    # Return values for further use
    return T_values, F_values, max_revenue, max_index, max_temperature, uncertainty_interval 


if __name__ == '__main__':
    # Run the equidistant search for the range [300 K, 380 K]
    _, _, _, _, _, _ = equidistance_search(Ti=300, Tf=380)
