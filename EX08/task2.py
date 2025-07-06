from casadi import *

# 1. Define / Create necessary symbolic variables: 
x = SX.sym('x',1) 
y = SX.sym('y',1) 
v = vertcat(x,y)

print(x,y,v)

#2. Define symbolic functions: 
f = sin(x) * cos(y) + x**2 - y**3 
J = jacobian(f, v)

print(f)
print(J)

#3. Define CasADi function objects:  
F = Function('F', [x, y], [f]) 
J_func = Function('J', [x, y], [J])

print(F)
print(J_func)

#4. Evaluate symbolic functions: 
x_val = pi/2 
y_val = sqrt(2)

print(f'Evaluation of F for x = {x_val} and y = {y_val}')
print(F(x_val,y_val))
print(f'Evaluation of J for x = {x_val} and y = {y_val}')
print(J_func(x_val,y_val))

'''
What do you observe for print(J)? What’s the difference compared to numerical 
calculation of the Jacobian and what could be the benefit of it?

print(J): [[((cos(y)*cos(x))+(x+x)), (-((sin(x)*sin(y))+(sq(y)+(y*(y+y)))))]]

CasADi offers exact derivatives through algorithmic differentiation, making it more 
accurate and efficient than numerical methods like finite differences, which rely on 
small perturbations and are prone to approximation errors due to step size and 
floating-point limitations. Unlike numerical methods that require multiple function 
evaluations per variable, CasADi computes Jacobians and gradients using a computational 
graph, enabling faster and more reliable optimization—especially important in nonlinear 
problems. In contrast, numerical Jacobians are simpler to implement but less precise 
and more computationally expensive for large problems.
'''