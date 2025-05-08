import numpy as np
import math

from a import p

def f(t,guess,p):

    Ca = guess[0]
    Cb = guess[1]
    Cc = guess[2]
    Cd = guess[3]

    dCa = (p.A0/p.q)*(-p.k10*(math.e**(-p.Ea1/(p.R*p.T)))*(Ca**(p.n1)))

    dCb = (p.A0/p.q)*(p.k10*(math.e**(-p.Ea1/(p.R*p.T)))*(Ca**(p.n1)) -p.k20*(math.e**(-p.Ea2/(p.R*p.T)))*(Cb**p.n2) -p.k30*(math.e**(-p.Ea3/(p.R*p.T)))*(Cb**p.n3))
    
    dCc = (p.A0/p.q)*(p.k20*(math.e**(-p.Ea2/(p.R*p.T)))*(Cb**(p.n2)))

    dCd = (p.A0/p.q)*(p.k30*(math.e**(-p.Ea3/(p.R*p.T)))*(Cb**(p.n3)))

    gradients = np.array([dCa,dCb,dCc,dCd])

    return gradients


if __name__ == '__main__':

    P = p()
    print('oi')
    gradients = f(P,[P.CaIn,P.CbIn,P.CcIn,P.CdIn])