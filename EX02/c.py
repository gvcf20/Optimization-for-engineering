import numpy as np                    
from scipy.integrate import solve_ivp   
import matplotlib.pyplot as plt        

from a import p as parameters          
from b import f               

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

# Function to solve the system of ODEs
def solve_ODEs(axial_lenght=50, plot=True):

    p = parameters()  # Instantiate the problem parameters

    # Initial concentrations of components A, B, C, D
    y0 = [p.CaIn, p.CbIn, p.CcIn, p.CdIn]

    # Define the spatial domain 
    x_span = (0, axial_lenght)

    # Points at which to evaluate the solution
    x_eval = np.linspace(*x_span, 3000)

    # Solve the ODE system using Runge-Kutta 4(5) method
    sol = solve_ivp(f, x_span, y0, method='RK45', t_eval=x_eval)

    # Extract the position points and concentrations
    t = sol.t
    y1, y2, y3, y4 = sol.y

    # Store the results in a list for convenience
    gradients = [y1, y2, y3, y4]

    # Plot the solution if requested
    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.plot(t, y1, label='Ca')  # Concentration of A
        plt.plot(t, y2, label='Cb')  # Concentration of B
        plt.plot(t, y3, label='Cc')  # Concentration of C
        plt.plot(t, y4, label='Cd')  # Concentration of D
        plt.xlabel('Length x')       # x-axis label (axial length of the reactor)
        plt.ylabel('Concentration')  # y-axis label
        plt.title('Solution of ODE System using RK45')  # Plot title
        plt.legend()                 # Show legend
        plt.grid(True)              # Show grid
        plt.show()                  # Display the plot

    return gradients, t  # Return concentrations and the corresponding x values

# Execute the script if it’s run directly
if __name__ == '__main__':

    grads, t = solve_ODEs()  # Solve and store the result
