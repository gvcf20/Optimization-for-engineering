from scipy.optimize import least_squares
import numpy as np

'''
Task 1: 
Fit the parameter vector x = [k0 EA n] T to the measured values. Use the MATLAB function 
lsqnonlin(), with the algorithm of Marquardt-Levenberg ('Algorithm',{'levenberg-
marquardt',0.005}). In Python you can use the function least_squares() from scipy.optimize.
'''

### Chatgpt was used with the following prompts: 
## Correct grammar in my comments and make them better

class Data:

    def __init__(self):
        
        ### Chatgpt was used with the following prompt: 
        # Transform this table in numpy vector, with the table image also prompted

        # Supersaturation values (σ)
        sigma = np.array([0.025, 0.050, 0.075, 0.100])

        # Temperatures in Celsius
        T_C = np.array([40, 60, 80])

        # Temperatures converted to Kelvin (T_K = T_C + 273.15)
        T_K = T_C + 273.15

        # Growth rate G (in µm/s) measured for each T and σ combination
        G = np.array([
            [0.0245, 0.0508, 0.1051, 0.1345],  # T = 40°C
            [0.0441, 0.1172, 0.2163, 0.3215],  # T = 60°C
            [0.0842, 0.2386, 0.4202, 0.6332],  # T = 80°C
        ])

        # Flattened version of the measured growth rates
        self.G_flat = G.flatten()

        # Generate a meshgrid for T and σ combinations and flatten them
        T_mesh, sigma_mesh = np.meshgrid(T_K, sigma, indexing='ij')
        self.T_flat = T_mesh.flatten()
        self.sigma_flat = sigma_mesh.flatten()

        # Universal gas constant R (J/mol·K)
        self.R = 8.3145

        return None

# Residual function to minimize: difference between measured G and model prediction
def residuals(params, data):
    k0, Ea, n = params
    Gk = k0 * np.exp((-Ea)/(data.R*data.T_flat))*(data.sigma_flat**n)
    return  data.G_flat - Gk

if __name__ == '__main__':

    data = Data()

    # Initial guesses for k0, EA (J/mol), and n
    initial_guess = [1, 1, 1]  
    result = least_squares(residuals, initial_guess, args=(data,))

    # Print optimization result object
    print(result)

    # Display the optimized parameters
    print("Fitted parameters:")
    print(f"k0 = {result.x[0]:.4e}")
    print(f"EA = {result.x[1]:.2f} J/mol")
    print(f"n = {result.x[2]:.4f}")
