'''
Write a function that returns the value of the objective function F for a given 
reactor temperature! Check your implementation using the given control solution. 
Control: F (T = 340 K) = 3.03 €/s, cA,out = 0.182 mol/m³, cB,out = 0.134 mol/m³ 
'''

def F(T, CAin = 12, CAout = 0.182, CBout = 0.134, q = 0.12, pB = 7, pA = 2, pT = 0.06):

    earnings = q*CBout*pB

    expenses = q*(CAin-CAout)*pA + q*pT*(abs(T - 298))

    revenue = earnings - expenses

    # print(earnings,expenses)
    print(f'The Revenue for temperature {T}K is: {revenue} euros per second')
    return revenue


if __name__ == '__main__':

    revenue = F(340)