from casadi import *
import numpy as np
import matplotlib.pyplot as plt


# Parameters
t_end = 5

# 1. Define symbols
x = MX.sym('x')
T = MX.sym('T')

# 2. Define the ODE
xdot = -T * x
dae = {'x': x, 'p': T, 'ode': xdot}

# 3. Create the integrator (final time = 5)
F = integrator('F', 'idas', dae, {'tf': t_end})

# 4. Create the optimization problem
opti = Opti()
T_var = opti.variable()

# 5. Solve the ODE with symbolic T_var
x_final = F(x0=1, p=T_var)['xf'][-1]  # scalar result

# 6. Define the objective: (x(5) - 0.3)^2
opti.minimize((x_final - 0.3)**2)

# 7. Constraints
opti.subject_to(opti.bounded(0.1, T_var, 5))

# 8. Solve
opti.solver('ipopt')
sol = opti.solve()

# 9. Extract solution
T_opt = sol.value(T_var)
x_T_opt = F(x0=1, p=T_opt)['xf'].full().flatten()[0]

print(f"Optimal T: {T_opt}")
print(f"x(5) with optimal T: {x_T_opt}")

# Plot done by chatgpt after I prompted the code above
t_vals = np.linspace(0, t_end, 100)
x_vals = np.exp(-T_opt * t_vals)  # Analytical solution
plt.plot(t_vals, x_vals, label=f"T = {T_opt:.4f}")
plt.axhline(0.3, color='r', linestyle='--', label='Target = 0.3')
plt.xlabel("Time [t]")
plt.ylabel("x(t)")
plt.grid(True)
plt.legend()
plt.title("ODE Solution with Optimized T")
plt.tight_layout()
plt.show()
