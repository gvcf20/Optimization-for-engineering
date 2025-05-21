import numpy as np
import matplotlib.pyplot as plt

from b import F

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

'''
e) Write a function in which the simultaneous search based on the golden section 
method is implemented and solve the given optimization problem for the 
interval T = [300 K ... 380 K]. Determine the optimal reactor temperature so that 
the remaining uncertainty interval is smaller than the one of task d)! Compare 
the number of function calls to task d). 
'''

def golden_ratio_search(f, a, b, tol=1e-6, max_iter=1000, print_solution=True):
    # Golden ratio constant (~0.618)
    gr = (5**0.5 - 1) / 2

    # Initialize counters
    n_iter = 0
    function_calls = 0

    # Initial internal points
    c = b - (b - a) * gr
    d = a + (b - a) * gr

    # Optimization loop
    while b - a > tol and n_iter <= max_iter:
        c = b - (b - a) * gr
        d = a + (b - a) * gr

        if f(c) < f(d):
            a = c
        else:
            b = d

        function_calls += 2
        n_iter += 1

    # Final results
    max_temperature = (b + a) / 2
    max_revenue = f(max_temperature)
    uncertainty_interval = b - a  # <- new line

    print(f'Solution was found in {n_iter} iterations')
    print(f'Function calls = {function_calls}')

    if print_solution:
        print(f'Max Revenue = {max_revenue} at temperature = {max_temperature}')
        print(f'Uncertainty interval = {uncertainty_interval:.6f} K')

    return max_temperature, max_revenue, function_calls, uncertainty_interval


if __name__ == '__main__':
    # Run golden ratio search
    max_temperature, max_revenue, function_calls, uncertainty_interval = golden_ratio_search(F, 300, 380, max_iter=10000)

    # Summary comparison
    print(f'\nGolden Search Function calls = {function_calls}')
    print(f'Golden Search Uncertainty interval = {uncertainty_interval:.6f} K')
    print('Equidistant Search Function calls = 50') # Output from d.py
    print('Equidistant Search Uncertainty interval = 0.49') # Output from d.py


    ### Golden Search uses more function calls but has a much smaller uncertainty interval, meaning it can be 
    ### much more accurate with much less function calls.


