import numpy as np
import math

from a import p as parameters

def f(t,guess):

    p = parameters()

    Ca = guess[0]
    Cb = guess[1]
    Cc = guess[2]
    Cd = guess[3]

    rate1 = p.k10 * np.exp(-p.Ea1 / (p.R * p.T))
    rate2 = p.k20 * np.exp(-p.Ea2 / (p.R * p.T))
    rate3 = p.k30 * np.exp(-p.Ea3 / (p.R * p.T))
    factor = p.A0 / p.q


    dCa = factor * (-rate1 * Ca**p.n1)
    dCb = factor * (rate1 * Ca**p.n1 - rate2 * Cb**p.n2 - rate3 * Cb**p.n3)
    dCc = factor * (rate2 * Cb**p.n2)
    dCd = factor * (rate3 * Cb**p.n3)

    gradients = [dCa,dCb,dCc,dCd]

    return gradients


if __name__ == '__main__':

    P = parameters()
    gradients = f(P,[P.CaIn,P.CbIn,P.CcIn,P.CdIn])
    print(gradients)