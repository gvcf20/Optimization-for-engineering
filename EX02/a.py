import math

'''
Define a structure (in MATLAB) / class (in Python) “p” for all constant 
parameters.
'''


class p:

    def __init__(self):
        
        self.CaIn = 12 # mol/m^3
        self.k10 = 5.4e10 #s^-1
        self.k20 = 4.6e17 #s^-1
        self.k30 = 5e7   #s^-1
        self.n1 = 1.1
        self.R = 8.3145 #J/mol/K
        self.A0 = 0.1 #m^2
        self.CbIn = 0 # mol/m^3
        self.CcIn = 0 # mol/m^3
        self.CdIn = 0 # mol/m^3
        self.Ea1 = 7.5e4 #J/mol
        self.Ea2 = 1.2e4 #J/mol
        self.Ea3 = 5.5e4 #J/mol
        self.n2 = 1
        self.n3 = 1
        self.T = 340 # K
        self.q = 0.12 # m^3/s

        return None
    

if __name__ == '__main__':
    P = p()




