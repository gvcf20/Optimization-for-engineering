import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

'''
Fit the parameter vector x = [k n]T, analogous to Task 1, to the measured values. 
Use the Matlab function lsqnonlin(), with the algorithm of Marquardt-Levenberg 
('Algorithm',{'levenberg-marquardt', 0.005}).  In  Python  you  can  use the  function  
least_squares() from scipy.optimize. Additionally, provide a graphical 
representation of the fit alongside the experimental data (in one figure)
'''
### Chatgpt was used with the following prompts: 
## Correct grammar in my comments and make them better

# Experimental time points
t_exp = np.array([2, 72, 322, 372, 422, 472, 522, 672, 822])

# Corresponding concentration values for component A
cA_exp = np.array([10, 8, 5, 4, 3, 2.5, 2, 1, 0.5])

# Differential equation: rate of change of concentration
def func(t, y, k, n):
    return -k * (y ** n)

# Solves the ODE for given parameters k and n, returning the concentrations at t_exp
def solve_concentration(k, n, t_eval):
    sol = solve_ivp(func, [t_eval[0], t_eval[-1]], [cA_exp[0]], args=(k, n), t_eval=t_eval, method='RK45')
    return sol.y[0]

# Residual function for optimization: difference between model and experimental values
def residuals(params):
    k, n = params
    cA_model = solve_concentration(k, n, t_exp)
    return cA_model - cA_exp

if __name__ == '__main__':

    # Initial guesses for k and n
    initial_guess = [0.001, 1]

    # Optimization using Levenberg-Marquardt method
    result = least_squares(residuals, initial_guess, method='lm')

    # Extract optimized parameters
    k_opt, n_opt = result.x

    # Compute model prediction with fitted parameters
    cA_fit = solve_concentration(k_opt, n_opt, t_exp)

    # Plot experimental vs. fitted concentration over time
    plt.figure(figsize=(8, 5))
    plt.plot(t_exp, cA_exp, 'o', label='Experimental data')  # Experimental points
    plt.plot(t_exp, cA_fit, '-', label=f'Fit: k={k_opt:.4f}, n={n_opt:.4f}')  # Fitted curve
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration cA [mol/l]')
    plt.title('Fit of concentration over time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
