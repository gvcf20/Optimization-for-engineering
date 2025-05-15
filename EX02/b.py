import numpy as np  
import math         

from a import p as parameters  

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

# Define the function that computes the derivatives (ODE system)
def f(t, guess):

    p = parameters()  # Create an instance of the parameters class

    # Extract concentrations from the input list 'guess'
    Ca = guess[0]
    Cb = guess[1]
    Cc = guess[2]
    Cd = guess[3]

    # Calculate reaction rate constants using the Arrhenius equation
    rate1 = p.k10 * np.exp(-p.Ea1 / (p.R * p.T))
    rate2 = p.k20 * np.exp(-p.Ea2 / (p.R * p.T))
    rate3 = p.k30 * np.exp(-p.Ea3 / (p.R * p.T))

    # Define a scaling factor 
    factor = p.A0 / p.q

    # Compute the derivatives 
    dCa = factor * (-rate1 * Ca**p.n1)  # A is consumed in reaction 1
    dCb = factor * (rate1 * Ca**p.n1 - rate2 * Cb**p.n2 - rate3 * Cb**p.n3)  # B is formed and consumed
    dCc = factor * (rate2 * Cb**p.n2)   # C is produced in reaction 2
    dCd = factor * (rate3 * Cb**p.n3)   # D is produced in reaction 3

    # Group all derivatives into a list
    gradients = [dCa, dCb, dCc, dCd]

    return gradients  # Return the derivatives

# Entry point of the script
if __name__ == '__main__':

    P = parameters()  # Create an instance of the parameter class

    # Evaluate the derivatives at the initial condition
    gradients = f(P, [P.CaIn, P.CbIn, P.CcIn, P.CdIn])

    # Print the calculated derivatives
    print(gradients)
