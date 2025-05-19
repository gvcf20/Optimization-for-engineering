import matplotlib.pyplot as plt
import numpy as np

from b import F


def equidistance_search(h = 3.2):

    interval = list(np.arange(300,380+h,h))
    revenues = []
    for T in interval:
        revenues.append(F(T))

    plt.figure(figsize=(12,10))
    plt.scatter(interval,revenues,label = 'Revenue')
    plt.xlabel('Temperature')       # x-axis label (axial length of the reactor)
    plt.ylabel('Revenue')  # y-axis label
    plt.title('Equidistant Search')  # Plot title
    plt.legend()                 # Show legend
    plt.grid(True)              # Show grid
    plt.show()
    return 


if __name__ == '__main__':

    a = equidistance_search()
    print(a)