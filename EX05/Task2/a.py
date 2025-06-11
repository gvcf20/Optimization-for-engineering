import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp

'''
a) Set up a model to calculate the time-dependent concentration of component A 
and solve it using an ODE solver. Let the ODE solver calculate the concentrations 
at the time points texp. Assume values for k and n! Plot the concentration course 
of component A.
'''
### Chatgpt was used with the following prompts: 
## Correct grammar in my comments and make them better

### To solve this problem I used the same aproach we did on exercise 1, with
### solve_ivp function and choose the initial value of 1 for 

# Define the differential equation: rate of change of A over time
def func(t, y, k, n):
    return -1 * k * (y ** n)

if __name__ == '__main__':

    ca_values = []

    k = 1  # Reaction rate constant
    n = 1  # Reaction order

    # Solve the ODE using the BDF method from t=0 to t=822 with initial concentration y0=1
    solution = solve_ivp(func, method='BDF', y0=[1], t_span=[0, 822], args=(k, n))

    # Use the index of each time step as x-axis (step number)
    t_values = [i for i in range(len(solution.y[0]))]

    # Plot concentration of A over the computed steps
    plt.plot(t_values, solution.y[0], label=f'k = {k}; \nn = {n}')

    plt.xlabel('Steps')         
    plt.ylabel('Ca')             
    plt.title('solve_ivp')       
    plt.legend()                 
    plt.grid(True)               
    plt.show()
