from casadi import *

#1. Define / Create necessary symbolic variables: 
x = SX.sym('x',2)  # Differential Variables  
t = SX.sym('t',1)  # Time 

#2. Define symbolic functions (ODE explicit) 
ODE = np.array([-x[1], x[0]]) 
ODE_F = Function('ode_function', [t, x], [ODE]) 

#3. Define CasADi objects:  
dae = {'x': x, 't': t, 'ode': ODE_F(t, x)} 
Int = integrator('Int','idas',dae) 

#4. Solve symbolic function: 
Inits = np.array([1, 0])# Initial Condition for ODE Var.  
res   = Int(x0=Inits); 

#5. Read-out the solution: 
x_sol = res['xf'].full().flatten() 
print(f" Result: x1 = {x_sol[0]}, x2 = {x_sol[1]}")

#a) Define a suitable time grid: 
Start = 0  
Stop  = 10
Step  = 1
grid = np.arange(Start, Stop+Step, Step)     
#b) Pass the time grid to the integrator object: 
Int = integrator('Int','idas',dae, grid[0],grid)

#4´. Solve symbolic function: 
Inits = np.array([1, 0])# Initial Condition for ODE Var.  
res   = Int(x0=Inits); 

#5´. Read-out the solution: 
x_sol = res['xf'].full().flatten() 
print(f" Result: x1 = {x_sol[0]}, x2 = {x_sol[1]}")