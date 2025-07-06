from casadi import *

#1. Define the objective function as a CasADi function: 
x = MX.sym('x',1)   
y = MX.sym('x',1)   
objective = (y - x**2)**2 

obj_F = Function('f', [x, y], [objective]) # CasADi Func. 
#2. Start "Opti stack framework"  Define opti-object and optimization variables  
opti = Opti()  
# Opti-object  
x_var = opti.variable()   
y_var = opti.variable()  
#3. Pass objective function to opti-object 
opti.minimize(obj_F(x_var, y_var)) 
#4. Add constraints to opti-object Solve symbolic function: 
opti.subject_to(x_var**2 + y_var**2 == 1) 
opti.subject_to(x_var + y_var >= 1)  
#5. Solve the NLP and print solution 
opti.solver('ipopt') 
sol = opti.solve() 
print('x:', sol.value(x_var)) 
print('y:', sol.value(y_var)) 
# EQC 
# IQC