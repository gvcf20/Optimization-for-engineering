from scipy.optimize import least_squares
import numpy as np
import numpy as np


### Chatgpt was used with the following prompt: 
# Transform this table in numpy vector, with the table image also prompted

class Data:

    def __init__(self):
        
        ### Chatgpt was used with the following prompt: 
        # Transform this table in numpy vector, with the table image also prompted


        # Supersaturation values (σ)
        sigma = np.array([0.025, 0.050, 0.075, 0.100])

        # Temperatures in Celsius
        T_C = np.array([40, 60, 80])

        # Temperatures in Kelvin (T_K = T_C + 273.15)
        T_K = T_C + 273.15

        G = np.array([
            [0.0245, 0.0508, 0.1051, 0.1345],  # T = 40°C
            [0.0441, 0.1172, 0.2163, 0.3215],  # T = 60°C
            [0.0842, 0.2386, 0.4202, 0.6332],  # T = 80°C
        ])

        self.G_flat = G.flatten()

        T_mesh, sigma_mesh = np.meshgrid(T_K, sigma, indexing='ij')

        self.T_flat = T_mesh.flatten()
        self.sigma_flat = sigma_mesh.flatten()

        self.R = 8.3145

        return None

def residuals(params, data):
    k0, Ea, n = params
    Gk = k0 * np.exp((-Ea)/(data.R*data.T_flat))*(data.sigma_flat**n)
    return  data.G_flat - Gk

if __name__ == '__main__':

    data = Data()
    initial_guess = [1, 1, 1]  # Initial guesses for k0, EA (J/mol), and n
    result = least_squares(residuals, initial_guess, args=(data,))

    print("Fitted parameters:")
    print(f"k0 = {result.x[0]:.4e}")
    print(f"EA = {result.x[1]:.2f} J/mol")
    print(f"n = {result.x[2]:.4f}")
