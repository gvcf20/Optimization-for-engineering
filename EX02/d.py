from a import p as parameters     
from b import f                  
from EX06.c import solve_ODEs           

## Chatgpt was used for generating commentary across the code with the following prompt:
## Correct grammar in my comments and make them better

# Lagrange interpolation with three points
def lagrange_interpolation(x0, x1, x2, y0, y1, y2, x):
    # Compute the Lagrange basis polynomials
    L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
    L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
    L2 = ((x - x1) * (x - x0)) / ((x2 - x1) * (x2 - x0))

    # Compute the interpolated value
    Px = y0 * L0 + y1 * L1 + y2 * L2

    return Px

# Golden section search to find the maximum of a unimodal function
def golden_ratio_search(f, a, b, tol=1e-5, max_iter=100):
    gr = (5**0.5 - 1) / 2  # Golden ratio constant (~0.618)
    n_iter = 0             # Iteration counter

    # Initial internal points
    c = b - (b - a) * gr
    d = a + (b - a) * gr

    # Loop until the interval is smaller than the tolerance or max iterations is reached
    while b - a > tol and n_iter <= max_iter:
        c = b - (b - a) * gr
        d = a + (b - a) * gr

        # Since we want to maximize, we keep the side with the higher value
        if f(c) < f(d):
            a = c  # Move left bound up
        else:
            b = d  # Move right bound down

        n_iter += 1  # Count iterations

    print(f'Solution was found in {n_iter} iterations')
    return (b + a) / 2  # Return midpoint of final interval as the maximum

# Helper to evaluate the interpolated value at a point t using Lagrange over 3 nearest points
def evaluate_lagrange_interpolation(t_data, y_data, t):
    for i in range(1, len(t_data) - 1):
        if t_data[i - 1] <= t <= t_data[i + 1]:
            return lagrange_interpolation(
                t_data[i - 1], t_data[i], t_data[i + 1],  # x-values
                y_data[i - 1], y_data[i], y_data[i + 1],  # y-values
                t  # point to interpolate
            )
    # If t is outside valid bounds, raise an error
    raise ValueError("t out of interpolation bounds")

# Main execution block
if __name__ == '__main__':

    # Solve the ODE system and extract concentration gradients and the spatial domain
    gradients, t_data = solve_ODEs(plot=True)

    # Select concentration of species B (index 1)
    y_data = gradients[1]

    # Create an interpolation function for species B using Lagrange interpolation
    f_interpolation = lambda t: evaluate_lagrange_interpolation(t_data, y_data, t)

    # Use golden ratio search to find the t (position) where Cb is maximized
    t_max = golden_ratio_search(f_interpolation, t_data[1], t_data[-2])

    # Evaluate the interpolated function at the maximum point
    y_max = f_interpolation(t_max)

    # Print results
    print(f'The concentration of Cb is maximized at length = {t_max}')
    print(f'The maximum value of Cb is {y_max}')
    print(y_max, t_max)
