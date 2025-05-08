import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from a import p
from b import f

import time
from tqdm import tqdm

if __name__ == '__main__':
    P = p()
    y0 = [P.CaIn, P.CbIn, P.CcIn, P.CdIn]
    t0, tf = 0, 50
    t_eval = np.linspace(t0, tf, 100)

    print("Solving ODE system")

    start = time.time()

    sol = solve_ivp(lambda t, y: f(t, y, P), (t0, tf), y0, method='RK45', dense_output=False)

    concentrations = []
    for t in tqdm(t_eval):
        concentrations.append(sol.sol(t))

    end = time.time()
    print(f"Solução completada em {end - start:.2f} segundos.")

    concentrations = np.array(concentrations).T

    print(concentrations)
    plt.plot(t_eval, concentrations[0], label='Ca')
    plt.plot(t_eval, concentrations[1], label='Cb')
    plt.plot(t_eval, concentrations[2], label='Cc')
    plt.plot(t_eval, concentrations[3], label='Cd')
    plt.xlabel('Position [m]')
    plt.ylabel('Gradient [mol/m³]')
    plt.title('Gradient vs Position')
    plt.legend()
    plt.grid()
    plt.show()