import numpy as np
import math

## System of Algebraic Equations (AE) to be solved:
## 3x - y^3 = e^-z
## x^2 + (xz)^3 = e^-y
## yz - x^2 = -e^(xyz)

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

def get_residuals(guess, AE):
    """Compute the residuals of the system for a given guess."""
    residuals = np.array([f(*guess) for f in AE])
    return residuals

def main(iter_limit=100, tol=1e-6, initial_guess=[1, 1, 1], AE=np.array([
        lambda x, y, z: 3*x - y**3 - math.e ** (-z),
        lambda x, y, z: x**2 + (x*z)**3 - math.e**(-y),
        lambda x, y, z: y*z - x**2 + math.e**(x*y*z)
    ])):
    """
    Main function to solve a nonlinear system of equations using Newton's method.
    
    Parameters:
    - iter_limit: Maximum number of iterations allowed.
    - tol: Tolerance threshold to consider a solution acceptable.
    - initial_guess: Starting point for the iterative process.
    - AE: System of equations represented as an array of lambda functions.
    """
    num_iter = 1
    residuals = get_residuals(initial_guess, AE)
    bool_tol = [abs(residual) <= tol for residual in residuals]

    guess = initial_guess

    # Iteration loop: stops when either all residuals are below tolerance or max iterations are reached
    while num_iter <= iter_limit and np.False_ in bool_tol:
        J = get_jacobian(AE, guess)                  # Compute the Jacobian matrix
        guess = get_new_guess(J, guess, residuals)   # Compute the new guess
        residuals = get_residuals(guess, AE)         # Update residuals
        bool_tol = [abs(residual) <= tol for residual in residuals]  # Re-check tolerances
        num_iter += 1

    solution = guess

    if num_iter > iter_limit:
        print('Limit number of iterations reached')
    else:    
        print(f'Solution found in {num_iter} iterations\n')
        print(f'Solution = {solution}')

    return solution

def get_jacobian(AE, guess):
    """
    Estimate the Jacobian matrix using the central difference method.

    Parameters:
    - AE: Array of lambda functions representing the system.
    - guess: Current guess for the variables.

    Returns:
    - Jacobian matrix (numpy array)
    """
    h = 1e-7
    guess = np.array(guess, dtype=float).reshape(-1)
    n = len(guess)
    Jacobian = np.zeros((len(AE), n))

    for j in range(len(AE)):
        for i in range(n):
            guess_plus = guess.copy()
            guess_minus = guess.copy()

            guess_plus[i] += h
            guess_minus[i] -= h

            f_plus = AE[j](*guess_plus)
            f_minus = AE[j](*guess_minus)

            Jacobian[j, i] = (f_plus - f_minus) / (2 * h)

    return Jacobian

def get_new_guess(J, old_guess, residuals):
    """
    Compute the next guess using Newton's method:
    Solves the linear system J(x_k) * dx = -f(x_k)

    Parameters:
    - J: Jacobian matrix at the current guess.
    - old_guess: Current guess.
    - residuals: Residuals at the current guess.

    Returns:
    - Updated guess (numpy array)
    """
    dx = np.linalg.solve(J, -residuals)
    new_guess = old_guess + dx

    print(new_guess)
    return new_guess

if __name__ == '__main__':
    """
    Run the solver. If an error occurs (e.g., due to overflow), handle it gracefully.
    """
    try:
        main(initial_guess=[1, 1, 100000], tol=1e-6, iter_limit=100)
    except Exception as e:
        print(f'An error occurred: {e}')
        print('Try again. Suggested input: [1, 1, 1]')
