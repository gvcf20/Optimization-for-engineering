import numpy as np
import matplotlib.pyplot as plt

from c import equidistance_search

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

'''
d) Carry out another equidistant search in the remaining uncertainty interval. Also 
divide this interval into 25 equal sections. What accuracy can be expected? How 
many function calls would be required to achieve the same accuracy if only a 
single equidistant search is performed? 
'''

if __name__ == '__main__':

    # Perform the first equidistant search over the full interval [300 K, 380 K]
    T_values, F_values, max_revenue, max_index, max_temperature, uncertainty_interval = equidistance_search(
        Ti=300, Tf=380, num_intervals=25, plot=False, print_solution=False
    )

    # Define new interval around the previously found maximum
    Ti = T_values[max_index - 1]  # temperature just before the max
    Tf = T_values[max_index + 1]  # temperature just after the max

    # Refine the search in the narrower interval
    T_values, F_values, max_revenue, max_index, max_temperature, uncertainty_interval = equidistance_search(
        Ti=Ti, Tf=Tf, num_intervals=25, plot=True, print_solution=True
    )

    # Estimate how many function evaluations would be required to achieve the same accuracy in one search
    print(f'To achieve this uncertainty interval with only one search it would be needed {round(((80) / uncertainty_interval) - 1)} functions evaluations')
