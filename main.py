import numpy as np
import math

## AE system to be solved:
## 3x - y^3 = e^-z
## x^2 + (xz)^3  = e^-y
## yz - x^2 = -e^(xyz)


def get_residuals(guess):

    residuals_func = np.array([lambda x,y,z: 3*x - y**3 - math.e ** (-z),
                        lambda x,y,z: x**2 + (x*z)**3 - math.e**(-y),
                        lambda x,y,z: y*z - x**2 + math.e**(x*y*z)])

    residuals = np.array([f(*guess) for f in residuals_func])
    
    return residuals

def main(iter_limit = 100, tol = 1e-6, initial_guess = [1,1,1],AE = np.array([lambda x,y,z: 3*x - y**3 - math.e ** (-z),
                        lambda x,y,z: x**2 + (x*z)**3 - math.e**(-y),
                        lambda x,y,z: y*z - x**2 + math.e**(x*y*z)])):

    num_iter = 1
    residuals = get_residuals(initial_guess)
    bool_tol = [abs(residual) <= tol for residual in residuals]

    guess = initial_guess

    while num_iter <= iter_limit and np.False_ in bool_tol:

        J = get_jacobian(AE,guess)
        
        guess = get_new_guess(J,guess,residuals)
        residuals = get_residuals(guess)
        bool_tol = [abs(residual) <= tol for residual in residuals]

        num_iter += 1

    print(f'Solution found in {num_iter} iterations \n')
    print(f'Solution = {guess}')
    return None

def get_jacobian(AE, guess):
    h = 0.0000001

    guess = np.array(guess, dtype=float).reshape(-1)

    n = len(guess)

    x1 = guess[0]
    x2 = guess[1]
    x3 = guess[2]
    guess_plus = np.array([h,h,h]) + np.array([guess])

    guess_minus = np.array([h,h,h]) - np.array([guess])

    # Jacobian = np.array([[3,-3*x2**2, math.e**(-x3)],[2*x1 + 3*(x1**2)*(x3**3),math.e**(-x2), 3*(x1**3)*(x3**2)],[-2*x1 + (x2*x3)*math.e**(x1*x2*x3), x3+(x1*x3)*math.e**(x1*x2*x3),x2+x1*x2*math.e**(x1*x2*x3)]])
    # print(Jacobian)
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

    # print(Jacobian)
    return Jacobian


def get_new_guess(J,old_guess,residuals):

    dx = np.linalg.solve(J, -residuals)
    new_guess = old_guess + dx

    print(new_guess)
    return new_guess



if __name__ == '__main__':

    try:

        main(initial_guess=[1,1,100000], tol=1e-6, iter_limit=100,)
    
    except Exception as e:

        print(f' An error ocurred: {e}')
        print('Try again. Suggested input: [1,1,1]')