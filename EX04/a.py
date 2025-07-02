import numpy as np                    
from scipy.integrate import solve_ivp   
import matplotlib.pyplot as plt       

from ex02 import parameters, f, solve_ODEs

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

'''
Write a function that returns the value of the objective function F for a given 
reactor temperature! Check your implementation using the given control solution. 
Control: F (T = 340 K) = 3.03 €/s, cA,out = 0.182 mol/m³, cB,out = 0.134 mol/m³ 
'''

def F(T, L, pA = 2, pB = 7, pT = 0.06):

    # Retrieve reactor parameters at temperature T
    p = parameters(T, L)

    # Solve the system of ODEs along the reactor of length 30 meters
    sol, x = solve_ODEs(T,L, axial_lenght=L)

    # Extract outlet concentrations for species A and B
    cA_out = sol[0]
    cB_out = sol[1]

    print(cA_out,cB_out)

    # Calculate revenue: gain from B production minus cost of consumed A
    revenue = p.q * (-pA * (p.CaIn - cA_out) + pB * cB_out)

    # Compute temperature control cost based on deviation from 298 K
    temp_cost = -p.q * pT * abs(T - 298)

    # Total objective function: net revenue minus temperature cost
    return revenue + temp_cost


if __name__ == '__main__':

    # Evaluate objective function at T = 305 K and print result
    print('F(T = 305 K, L = 36 m) = ', -F(305, 36))
