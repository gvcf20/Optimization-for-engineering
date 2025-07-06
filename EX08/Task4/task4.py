from casadi import *

#1. Define the objective function as a CasADi function: 
alpha = MX.sym('x',1)   
g = 10 # [m/s^2]
v0 = 10 # [m/s]    
distance = (v0**2 / g) * sin(2 * alpha)
obj_F = Function('f', [alpha], [-distance]) 

#2. Start "Opti stack framework"  Define opti-object and optimization variables  
opti = Opti()  
# Opti-object    
alpha_var = opti.variable()  
#3. Pass objective function to opti-object 
opti.minimize(obj_F(alpha_var)) 
#4. Add constraints to opti-object Solve symbolic function: 
deg2rad = np.pi / 180
opti.subject_to(alpha_var <= 85 * deg2rad) 
opti.subject_to(alpha_var >= 10 * deg2rad)  
#5. Solve the NLP and print solution 
opti.solver('ipopt') 
sol = opti.solve() 
print('Max angle:', sol.value(alpha_var)*180/(np.pi))
max_distance = (v0**2 / g) * sin(2 * sol.value(alpha_var))
print('Max Distance:', max_distance)