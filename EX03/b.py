import numpy as np                    
from scipy.integrate import solve_ivp   
import matplotlib.pyplot as plt       

from ex02 import parameters, f, solve_ODEs

'''
Write a function that returns the value of the objective function F for a given 
reactor temperature! Check your implementation using the given control solution. 
Control: F (T = 340 K) = 3.03 €/s, cA,out = 0.182 mol/m³, cB,out = 0.134 mol/m³ 
'''

def F(T, pA = 2, pB = 7, pT = 0.06):

    p = parameters(T)

    sol,x= solve_ODEs(T, axial_lenght=30)

    cA_out = sol[0]
    cB_out = sol[1]

    revenue = p.q * (-pA * (p.CaIn - cA_out) + pB * cB_out)
    temp_cost = p.q * pT * abs(T - 298)

    return revenue - temp_cost


if __name__ == '__main__':

    print('F(T = 340 K) = ', -F(340))