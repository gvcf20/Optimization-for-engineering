from scipy.optimize import least_squares
import numpy as np

from EX05.Task1.task1 import residuals, Data

'''
Bonus Task 
Transform the problem of Task 1 into a linear optimization problem by linearizing the 
growth rate G. Formulate the error function and the corresponding Jacobian matrix! 
Apply the discussed Gauss-Newton method for sum of squares. What advantages does 
this approach offer compared to the approach of Task 1? How can you explain the 
deviations compared to the results of Task 1?
'''

### Chatgpt was used with the following prompts: 
# Correct grammar in my comments and make them better

# Residual function after linearizing the model using logarithms
def residuals(params, data):
    k0, Ea, n = params

    # Linearization constants
    c = np.log(k0)
    a = (-1) * (Ea / data.R)

    # Predictor variables for the linearized model
    x1 = 1 / data.T_flat
    x2 = np.log(data.sigma_flat)

    # Linear model prediction
    yk = c + a * x1 + n * x2

    # Log of experimental data
    y = np.log(data.G_flat)

    # Return residuals (difference between model and data)
    return y - yk

# Jacobian matrix of partial derivatives for Gauss-Newton method
def Jacobian_matrix(params, data):
    k0, Ea, n = params
    R = data.R

    x1 = 1 / data.T_flat
    x2 = np.log(data.sigma_flat)

    # Partial derivatives of the residuals w.r.t each parameter
    dres_dk0 = -1 / k0 * np.ones_like(x1)
    dres_dEa = x1 / R
    dres_dn  = -x2

    # Combine into Jacobian matrix
    J = np.column_stack([dres_dk0, dres_dEa, dres_dn])
    return J

if __name__ == '__main__':

    data = Data()

    # Initial guesses for k0, EA (J/mol), and n
    initial_guess = [1, 300, 1]

    # Least squares optimization using the analytical Jacobian (Gauss-Newton method)
    result = least_squares(residuals, initial_guess, jac=Jacobian_matrix, args=(data,))

    # Output the optimization result
    print(result)

    print("Fitted parameters:")
    print(f"k0 = {result.x[0]:.4e}")
    print(f"Ea = {result.x[1]:.2f} J/mol")
    print(f"n = {result.x[2]:.4f}")
