import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from a import p as parameters
from b import f

import time
from tqdm import tqdm

def solve_ODEs(axial_lenght = 50, plot = True):

    p = parameters()

    y0 = [p.CaIn,p.CbIn,p.CcIn,p.CdIn]

    x_span = (0, axial_lenght)

    x_eval = np.linspace(*x_span, 3000)

    sol = solve_ivp(f, x_span, y0, method='RK45', t_eval=x_eval) 

    t = sol.t
    y1, y2, y3, y4 = sol.y

    gradients = [y1, y2, y3, y4]

    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.plot(t, y1, label='Ca')
        plt.plot(t, y2, label='Cb')
        plt.plot(t, y3, label='Cc')
        plt.plot(t, y4, label='Cd')
        plt.xlabel('Lenght x')
        plt.ylabel('Gradients')
        plt.title('Solution of ODEs System using RK45')
        plt.legend()
        plt.grid(True)
        plt.show() 
       
    return gradients, t

if __name__ == '__main__':

    grads, t = solve_ODEs()