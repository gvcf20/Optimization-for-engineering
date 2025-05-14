# At which length the concentration on B is maximized?

from a import p as parameters
from b import f
from c import solve_ODEs


def lagrange_interpolation(x0,x1,x2,y0,y1,y2,x):

    L0 = ((x-x1)*(x-x2))/((x0-x1)*(x0-x2))
    L1 = ((x-x0)*(x-x2))/((x1-x0)*(x1-x2))
    L2 = ((x-x1)*(x-x0))/((x2-x1)*(x2-x0))

    Px = y0*L0 + y1*L1 + y2*L2

    return Px



def golden_ratio_search(f,a,b,tol = 1e-5, max_iter = 100):

    gr = (5**0.5 - 1)/2
    n_iter = 0
    c = b - (b - a) * gr
    d = a + (b - a) * gr

    while b - a > tol and n_iter <= max_iter:
        c = b - (b - a) * gr
        d = a + (b - a) * gr

        if f(c) < f(d):

            a = c
        
        else:

            b = d

        n_iter += 1


    print(f'Solution was found in {n_iter} iterations')
    return (b+a)/2

def evaluate_lagrange_interpolation(t_data, y_data, t):
    for i in range(1, len(t_data) - 1):
        if t_data[i-1] <= t <= t_data[i+1]:
            return lagrange_interpolation(
                t_data[i-1], t_data[i], t_data[i+1],
                y_data[i-1], y_data[i], y_data[i+1],
                t
            )
    raise ValueError("t out of interpolation bounds")


if __name__ == '__main__':

    gradients, t_data = solve_ODEs(plot=True)

    y_data = gradients[1]

    f_interpolation = lambda t: evaluate_lagrange_interpolation(t_data,y_data,t)

    t_max = golden_ratio_search(f_interpolation, t_data[1], t_data[-2])

    y_max = f_interpolation(t_max)

    print(f'The concentration of Cb is maximized at lenght = {t_max}')
    print(f'The maximum value of Cb is {y_max}')
    print(y_max,t_max)


