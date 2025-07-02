
from a import F
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    T_values = np.linspace(300, 380, 20)
    L_values = np.linspace(1, 40, 20)
    T_grid, L_grid = np.meshgrid(T_values, L_values)

    Z = np.zeros_like(T_grid)

    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            T = T_grid[i, j]
            L = L_grid[i, j]
            try:
                Z[i, j] = F(T, L)
            except Exception as e:
                # print(f"Error at T={T}, L={L}: {e}")
                Z[i, j] = np.nan  

    max_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    max_revenue = Z[max_idx]
    best_T = T_grid[max_idx]
    best_L = L_grid[max_idx]

    print(f"Maximum revenue: {max_revenue:.2f}")
    print(f"Achieved at T = {best_T:.2f} K, L = {best_L:.2f} m")

    # Plot
    plt.figure(figsize=(10, 6))
    cp = plt.contourf(T_grid, L_grid, Z, levels=20, cmap='viridis')
    plt.colorbar(cp, label='Revenue')
    plt.xlabel('Temperature T (K)')
    plt.ylabel('Reactor Length L (m)')
    plt.title('2D Contour Plot of Revenue F(T, L)')
    plt.scatter(best_T, best_L, color='red', label='Max Revenue')
    plt.legend()
    plt.show()
