import numpy as np                    
from scipy.integrate import solve_ivp   
import matplotlib.pyplot as plt       
from scipy.optimize import minimize

from ex02 import parameters, f, solve_ODEs

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

'''
b) Re-use the program from exercise 4 and extend it by penalty functions to 
consider the constraints. Solve the optimization problem using initial values of 
𝑥0 = [15m 300K] with weighting factors of 𝜎 = 0.1 and 𝜎 = 10. In MATLAB, 
apply fminsearch and in Python apply scipy.optimize.minimize(fun, 
x0, method='Nelder-Mead'). Prove that the constraints have been met 
sufficiently!  
'''

def F(T, L, pA = 2, pB = 7, pT = 0.06):

    # Retrieve reactor parameters at temperature T
    p = parameters(T, L)

    # Solve the system of ODEs along the reactor of length 30 meters
    sol, x = solve_ODEs(T,L, axial_lenght=L)

    # Extract outlet concentrations for species A and B
    cA_out = sol[0]
    cB_out = sol[1]

    # Calculate revenue: gain from B production minus cost of consumed A
    revenue = p.q * (-pA * (p.CaIn - cA_out) + pB * cB_out)

    # Compute temperature control cost based on deviation from 298 K
    temp_cost = -p.q * pT * abs(T - 298)

    # Total objective function: net revenue minus temperature cost
    return revenue + temp_cost

def penalized_F(x, sigma, pA = 2, pB = 7, pT = 0.06):

    T,L = x

    base_value = -F(T, L, pA, pB, pT)  # negative for maximization

    # Retrieve concentrations for penalty calculation
    p = parameters(T, L)
    sol, _ = solve_ODEs(T, L, axial_lenght=L)
    cB_out = sol[1]

    # Constraint 1: Product B concentration must not exceed its solubility limit
    cB_sat = np.exp(-3.75 + 1.14e-2 * T)
    penalty1 = sigma * max(0, cB_out - cB_sat)**2

    # Constraint 2: Reactor length must not exceed 20 m
    penalty2 = sigma * max(0, L - 20)**2

    # Constraint 3: Partial pressure of component C must be ≤ 120 mbar
    T_C = T - 273.15
    Pc = np.exp(8 - 1730 / (208 + T/T_C)) # I considered that (T/ºC) would be the temperature in Celcius

    penalty3 = sigma * max(0, Pc - 120)**2

    # Sum up all penalties with base objective function
    total = base_value + penalty1 + penalty2 + penalty3

    return total


if __name__ == '__main__':

    sigmas = [0.1,10,1000000]

    for sigma in sigmas:

        x0 = [300, 15]  # Initial guess [T, L]
        res = minimize(penalized_F, x0,args=(sigma,), method='Nelder-Mead')

        # Report the optimal values for each sigma
        print(f"\nOptimal solution for sigma = {sigma}:")
        print(f"T = {res.x[0]:.2f} K, L = {res.x[1]:.2f} m")
        print("Objective value:", -F(res.x[0], res.x[1]))  # remove minus to get actual max

        # ---- Constraint Check and Proof ----

        # Solve ODEs at optimal point to get concentrations
        sol, _ = solve_ODEs(res.x[0], res.x[1], axial_lenght=res.x[1])
        cA_out = sol[0]
        cB_out = sol[1]

        # Calculate constraint quantities
        cB_sat = np.exp(-3.75 + 1.14e-2 * res.x[0])
        T_C = res.x[0] - 273.15
        Pc = np.exp(8 - 1730 / (208 + 300/T_C))

        # Print constraint values
        print(f"Constraint 1: cB_out = {cB_out:.4f} mol/m³, cB_sat = {cB_sat:.4f} -> {'OK' if cB_out <= cB_sat else 'VIOLATED'}")
        print(f"Constraint 2: L = {res.x[1]:.2f} m ≤ 20.00 m -> {'OK' if res.x[1] <= 20 else 'VIOLATED'}")
        print(f"Constraint 3: Pc = {Pc:.2f} mbar ≤ 120 mbar -> {'OK' if Pc <= 120 else 'VIOLATED'}")
