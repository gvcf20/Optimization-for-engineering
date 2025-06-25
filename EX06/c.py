import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

'''
c) Transform the constrained optimization problem of task b) into an 
unconstrained  optimization  problem  by  using  quadratic  penalty  functions!  
Solve the reformulated problem with weighting factors of σ = 1, 10, and 100, 
and plot the curves of the reformulated objective functions. How do you explain 
the differences? 
'''

def A(x): #Area function as described in the pdf solution for letter b
    x1, x2 = x
    if x1 == 0 or x2 == 0:
        return np.inf
    return 2 * x1 * x2 + 25000 * ((2 * x2 + x1) / (x2 * x1))

def f(x, sigma1, sigma2): #Objective function with penalties as described in the pdf solution for letter c
    x1, x2 = x
    a = A(x)
    penalty1 = (sigma1 / 2) * (max(30 - x1, 0))**2
    penalty2 = (sigma2 / 2) * (max(20 - x2, 0))**2
    return a + penalty1 + penalty2

if __name__ == '__main__':

    x0 = [1, 1]

    sigmas = [(1, 1), (10, 10), (100, 100)]

    for sigma1, sigma2 in sigmas:
        result = minimize(f, x0, args=(sigma1, sigma2), method='L-BFGS-B')
        print(f"Sigma1: {sigma1}, Sigma2: {sigma2}")
        print(f"Optimal x: {result.x}")
        print(f"Minimum value: {result.fun}")
        print("-" * 40)


    ## Chatgpt was used for ploting with the following prompt:
    # Given functions A and f and 3 pais of sigma values, plot
    # the four curves in a single plot


    x1_vals = np.linspace(30, 50, 100)
    x2_vals = np.linspace(20, 40, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Prepare figure
    fig = plt.figure(figsize=(16, 15))
    ax_base = fig.add_subplot(2, 2, 1, projection='3d')

    # --- Plot A(x) ---
    Z_base = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z_base[i, j] = A([X1[i, j], X2[i, j]])

    ax_base.plot_surface(X1, X2, Z_base, cmap='viridis', alpha=0.8)
    ax_base.set_title('Base Function $A(x)$')
    ax_base.set_xlabel('$x_1$')
    ax_base.set_ylabel('$x_2$')
    ax_base.set_zlabel('$A(x)$')

    # --- Penalized Objective Functions ---
    sigmas = [(1, 1), (10, 10), (100, 100)]

    for idx, (sigma1, sigma2) in enumerate(sigmas, start=2):
        ax = fig.add_subplot(2, 2, idx, projection='3d')

        Z_pen = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                Z_pen[i, j] = f([X1[i, j], X2[i, j]], sigma1, sigma2)

        result = minimize(f, [10, 10], args=(sigma1, sigma2), method='L-BFGS-B')
        opt_x = result.x
        opt_val = result.fun

        ax.plot_surface(X1, X2, Z_pen, cmap='viridis', alpha=0.8)
        ax.scatter(opt_x[0], opt_x[1], opt_val, color='r', s=50, label='Optimum')
        ax.set_title(f'$\\tilde{{F}}(x)$ with $\\sigma_1={sigma1}, \\sigma_2={sigma2}$')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$\\tilde{F}(x)$')
        ax.legend()

    plt.tight_layout()
    plt.show()

