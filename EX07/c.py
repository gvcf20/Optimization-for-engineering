import numpy as np                    
from scipy.integrate import solve_ivp   
import matplotlib.pyplot as plt       
from scipy.optimize import minimize

from ex02 import parameters, f, solve_ODEs

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

'''
c) Furthermore, solve the optimization problem using the MATLAB solver fmincon 
/ the Python solver scipy.optimize.minimize for constrained optimization 
problems. Choose a suitable algorithm / method by yourself. Prove that the 
constraints have been met sufficiently! 
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


def constraint1(x):  # cB_sat - cB_out >= 0
    T, L = x
    try:
        sol, _ = solve_ODEs(T, L, axial_lenght=L)
        cB_out = sol[1]
        cB_sat = np.exp(-3.75 + 1.14e-2 * T)
        return cB_sat - cB_out
    except:
        return -1e6  # infeasible

def constraint2(x):  # 20 - L >= 0
    return 20 - x[1]

def constraint3(x):  # 120 - Pc >= 0
    T_C = x[0] - 273.15
    Pc = np.exp(8 - 1730 / (208 + T_C))
    return 120 - Pc

# Objective function to minimize (-F for maximization)
def neg_F(x):
    T, L = x
    return -F(T, L)

# Initial guess and bounds
x0 = [300, 15]
bounds = [(273, 600), (1, 30)]  # T between 273K and 600K, L between 1 and 20m

# Constraints in scipy format
constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2},
    {'type': 'ineq', 'fun': constraint3}
]

# Solve using SLSQP 
res = minimize(neg_F, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# Output results
T_opt, L_opt = res.x
print("\n=== Constrained Optimization (SLSQP) ===")
print(f"Optimal T = {T_opt:.2f} K, L = {L_opt:.2f} m")
print("Objective value (€/s):", -res.fun)

# Constraint verification
sol, _ = solve_ODEs(T_opt, L_opt, axial_lenght=L_opt)
cB_out = sol[1]
cB_sat = np.exp(-3.75 + 1.14e-2 * T_opt)
T_C = T_opt - 273.15
Pc = np.exp(8 - 1730 / (208 + T_C))

print(f"\nConstraint 1 (cB_out ≤ cB_sat): {cB_out:.4f} ≤ {cB_sat:.4f} → {'OK' if cB_out <= cB_sat else 'VIOLATED'}")
print(f"Constraint 2 (L ≤ 20): {L_opt:.2f} ≤ 20 → {'OK' if L_opt <= 20 else 'VIOLATED'}")
print(f"Constraint 3 (Pc ≤ 120): {Pc:.2f} ≤ 120 → {'OK' if Pc <= 120 else 'VIOLATED'}")
