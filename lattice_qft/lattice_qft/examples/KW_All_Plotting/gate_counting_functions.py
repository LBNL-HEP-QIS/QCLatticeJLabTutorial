'''
    j= iteration
    r= # fractional qubits for arithmetic
    b= # total qubits for arithmetic, (b-r is number of integral qubits)
    d= degree of series approximations, e.g. d=2 is the series a0 + a1(mu')^2 + a2(mu')^4
'''

def large_sigma_alpha_count(j, d, r):
    if j == 1:
        return 11*d*r**2 + 10*d*r - 3*r**2 + 49*r - 27*d + 94
    elif j == 2:
        return 11*d*r**2 + 10*d*r - 3*r**2 + 71*r - 27*d + 196
    else: # j >= 3
        return 11*d*r**2 + 10*d*r - 3*r**2 + 27*r - 27*d - 73 + 22*j*r + 11*j**2 + 111*j


def int_sigma_alpha_count(j, d, r, b):
    '''
    Count the number of gates to compute alpha.

    Precons:
        j  = 1..k
        r >= j + 1
        b >  r + 1 (b= r+2 minimum, which is a ones place and a sign)
    '''
    total= d * (22*b*(r-2) - 11*r**2 + 10*r + 61) + 11*b**2 + 22*r*(b-r) + 15*r - 21*b + 42 + 12*j
    
    # Add the CCM gates
    total+= 4*b*(d+1)
    
    if j == 1:
        total+= 38
    elif j == 2:
        total+= 158
    else: # j >= 3
        total+= 22*j**2 + 95*j - 123
    
    if d > 1:
        total+= 44*b*r - 22*r**2 - 120*b + 20*r + 122 + 8*b
        
    return total


def small_sigma_alpha_count():
    return 2