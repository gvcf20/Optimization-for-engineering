from casadi import *
import matplotlib.pyplot as plt 

#1. Define / Create necessary symbolic variables: 
Ca = SX.sym('Ca',1)  # Concentration of A [mol/L]  
t = SX.sym('t',1)  # Time [min]
k = 0.25 #Constant [1/min]

#2. Define symbolic functions (ODE explicit)

ODE = -k * Ca
ODE_F = Function('ode', [t, Ca], [ODE])

#3. Define CasADi objects:
dae = {'x': Ca, 't': t, 'ode': ODE_F(t, Ca)}

#a) Define a suitable time grid: 
Start = 0  
Stop  = 10
Step  = 1
grid = np.arange(Start, Stop+Step, Step)     
#b) Pass the time grid to the integrator object: 
Int = integrator('Int','idas',dae, grid[0],grid)

#4. Solve symbolic function: 
Ca0 = 10 #Inicial concentration of A [mol/L] 
res   = Int(x0=Ca0)

#5. Read-out the solution:
x_sol = res['xf'].full().flatten() 
print(f"Concentration of A after 10 minutes reaction: {x_sol[-1]}")

#6. Plot done by chatgpt after I prompted the code above

plt.figure(figsize=(8, 5))
plt.plot(grid, x_sol, marker='o', linestyle='-', color='blue', label='C_A(t)')
plt.xlabel('Time [min]')
plt.ylabel('Concentration of A [mol/L]')
plt.title('Concentration of A over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


