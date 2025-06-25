import numpy as np
from scipy.optimize import minimize


'''
d)  Solve the optimization problem of task a) by using a built-in solver for 
constrained optimization problems. In MATLAB you may use fmincon. Python 
users may apply scipy.minimize. Formulate your objective function in such 
a way that it returns the packaging surface A for any vector x = [x1, x2 ,x3 ]T. Use 
the SQP algorithm to solve the problem.
'''

def A(x): #Area function as described in the pdf solution for letter a
    x1, x2, x3 = x
    return 2 * x1 * x2 + 2 * x3 * x2 + x1 * x3

def volume_constraint(x): #Volume Constraints
    x1, x2, x3 = x
    return x1 * x2 * x3 - 25000

if __name__ == '__main__':

    bounds = [(30, np.inf), (20, np.inf), (0, np.inf)] #dimension constraints

    constraints = ({
        'type': 'eq',
        'fun': volume_constraint
    })

    x0 = [35, 25, 10]

    result = minimize(A, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        print("Optimal solution:")
        print(f"x1 = {result.x[0]:.4f} cm")
        print(f"x2 = {result.x[1]:.4f} cm")
        print(f"x3 = {result.x[2]:.4f} cm")
        print(f"Minimum A = {result.fun:.4f} cm²")
    else:
        print("Optimization failed:", result.message)