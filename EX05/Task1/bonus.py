from scipy.optimize import least_squares
import numpy as np

from EX05.Task1.task1 import residuals, Data

### Chatgpt was used with the following prompts: 
# Correct grammar in my comments and make them better

def residuals(params, data):

    k0, Ea, n = params

    c = np.log(k0)
    a = (-1)*(Ea/(data.R))

    x1 = 1/data.T_flat
    x2 = np.log(data.sigma_flat)
    yk = c + a*x1 +n*x2

    y = np.log(data.G_flat)

    return y - yk

def Jacobian_matrix(params, data):
    k0, Ea, n = params
    R = data.R

    x1 = 1 / data.T_flat
    x2 = np.log(data.sigma_flat)

    # Partial derivatives
    dres_dk0 = -1 / k0 * np.ones_like(x1)
    dres_dEa = x1 / R
    dres_dn  = -x2

    J = np.column_stack([dres_dk0, dres_dEa, dres_dn])
    return J


if __name__ == '__main__':

    data = Data()

    initial_guess = [1, 300, 1]  # Initial guesses for k0, EA (J/mol), and n

    result = least_squares(residuals, initial_guess, jac=Jacobian_matrix, args=(data,))

    print(result)

    print("Fitted parameters:")
    print(f"k0 = {result.x[0]:.4e}")
    print(f"Ea = {result.x[1]:.2f} J/mol")
    print(f"n = {result.x[2]:.4f}")
    

