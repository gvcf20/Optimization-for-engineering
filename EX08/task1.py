#Install, Import, and Test CasADi: 

from casadi import *

if __name__ == '__main__':

    x = SX.sym('x', 1)
    print(jacobian(sin(x), x))




