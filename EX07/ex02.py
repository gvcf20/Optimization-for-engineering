import numpy as np                    
from scipy.integrate import solve_ivp   
import matplotlib.pyplot as plt

# Functions and class from EX02 adapted for use in EX03

class parameters:

    def __init__(self, _T, _L):
        
        self.CaIn = 12 # mol/m^3
        self.k10 = 5.4e10 #s^-1
        self.k20 = 4.6e17 #s^-1
        self.k30 = 5e7   #s^-1
        self.n1 = 1.1
        self.R = 8.3145 #J/mol/K
        self.A0 = 0.1 #m^2
        self.CbIn = 0 # mol/m^3
        self.CcIn = 0 # mol/m^3
        self.CdIn = 0 # mol/m^3
        self.Ea1 = 7.5e4 #J/mol
        self.Ea2 = 1.2e5 #J/mol
        self.Ea3 = 5.5e4 #J/mol
        self.n2 = 1
        self.n3 = 1
        self.T = _T # K
        self.q = 0.12 # m^3/s
        self.L = _L

        return None

def f(t, guess, T,L):

    p = parameters(T,L)  # Create an instance of the parameters class

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

def solve_ODEs(T,L,axial_lenght=50, plot=True):

    p = parameters(T,L)  # Instantiate the problem parameters

    # Initial concentrations of components A, B, C, D
    y0 = [p.CaIn, p.CbIn, p.CcIn, p.CdIn]

    # Define the spatial domain 
    x_span = (0, axial_lenght)

    # Points at which to evaluate the solution
    x_eval = np.linspace(*x_span, 100)

    # Solve the ODE system using Runge-Kutta 4(5) method
    sol = solve_ivp(f, x_span, y0, method='RK45', args=(T,L), t_eval=x_eval)

    # Extract the position points and concentrations
    t = sol.t
    y1, y2, y3, y4 = sol.y[:,-1]

    # Store the results in a list for convenience
    gradients = [y1, y2, y3, y4]

    return gradients, t  # Return concentrations and the corresponding x values
