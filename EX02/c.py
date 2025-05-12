import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from a import p as parameters
from b import f

import time
from tqdm import tqdm

if __name__ == '__main__':

    p = parameters()

    y0 = [p.CaIn,p.CbIn,p.CcIn,p.CdIn]

    x_span = (0, 50)
    x_eval = np.linspace(*x_span, 50)

    sol = solve_ivp(f, x_span, y0, method='RK45', t_eval=x_eval, max_step = 0.001, rtol = 1e-6,atol = 1e-8) 

    t = sol.t
    y1, y2, y3, y4 = sol.y

    plt.figure(figsize=(10, 6))
    plt.plot(t, y1, label='y1')
    plt.plot(t, y2, label='y2')
    plt.plot(t, y3, label='y3')
    plt.plot(t, y4, label='y4')
    plt.xlabel('Time t')
    plt.ylabel('Solutions')
    plt.title('Solution of 4 ODEs using RK45')
    plt.legend()
    plt.grid(True)
    plt.show()