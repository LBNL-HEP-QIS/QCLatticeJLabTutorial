''' Arithmetic circuits optimized for particular register sizes, used to compute alpha for large and intermediate sigma. '''


####################################################################################################
# General Imports                                                                                  #
####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from qiskit import(
  QuantumCircuit,
  QuantumRegister,
  ClassicalRegister,
  execute,
  Aer,
  circuit)
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import MCMT


####################################################################################################
# General Imports                                                                                  #
####################################################################################################
#from testing_utilities import *
from lattice_qft.core.arithmetic_pkg.arithmetic import *


####################################################################################################
# Arithmetic Functions                                                                             #
####################################################################################################
def muSquarer(j, r, circuit=False):
    '''
    Preconditions: r <= 2 * (j+1) , which is enough to store mu^2 without any underflow, and r >= j 
    '''
    anc= QuantumRegister(1, 'anc') # Ancillary qubit, to copy multiplier bits to
    mu= QuantumRegister(j+3, 'mu') # mu register. Has 1 sign qubit, 1 sign extension bit (for partial sums), j fractional qubits, and 1 classical bit (least significant). Also 1 extension bit
    # Classical bit currently implemented as a qubit
    prod= QuantumRegister(r-2, 'prod') # Result is guaranteed < 1/4, so has r-2 bits

    qc= QuantumCircuit(anc, mu, prod)


    # Add partial sums
    # Order of CAdder is (c, a, b, z)
    for n in range(j+2):
        #print('n= ' + str(n))
        qc.cx(mu[n], anc)

        # In the absence of register limits:
        #qc.compose(CAdder(r-j+n+1, modular=True, circuit=False), [anc] + mu[:] + prod[:r-j+n+1], inplace=True)
        right_room= 2*(j+1) - r - n  # = -n
        left_index= r - j + n + 1 # 5 + n

        if right_room >= 0 and left_index < r-2:
            #print('right_room >= 0 and left_index < r-2\n')

            if n == j+1: # last iteration use subtractor
                qc.compose(CSubtractor(j+3-right_room, modular=True, circuit=False), [anc] + mu[right_room:] + prod[:j + 3 - right_room] , inplace=True)
            else:
                qc.compose(CAdder(j+3-right_room, modular=True, circuit=False), [anc] + mu[right_room:] + prod[:j + 3 - right_room] , inplace=True)

        elif right_room >= 0 and left_index >= r-2:
            #print('right_room >= 0 and left_index >= r-2\n')

            if n == j+1: # last iteration use subtractor
                qc.compose(CSubtractor(r-2, modular=True, circuit=False), [anc] + mu[right_room:right_room + r - 2] + prod[:] , inplace=True)
            else:
                qc.compose(CAdder(r-2, modular=True, circuit=False), [anc] + mu[right_room:right_room + r - 2] + prod[:] , inplace=True)

        elif right_room < 0 and left_index < r-2:
            #print('right_room < 0 and left_index < r-2\n')

            if n == j+1: # last iteration use subtractor
                qc.compose(CSubtractor(j+3, modular=True, circuit=False), [anc] + mu[ : j + 3] + prod[-right_room : j + 3 - right_room] , inplace=True)
            else:
                qc.compose(CAdder(j+3, modular=True, circuit=False), [anc] + mu[ : j + 3] + prod[-right_room : j + 3 - right_room] , inplace=True)

        elif right_room < 0 and left_index >= r-2:
            #print('right_room < 0 and left_index >= r-2\n')

            if n == j+1: # last iteration use subtractor
                if r-2+right_room > 0:
                    qc.compose(CSubtractor(r-2+right_room, modular=True, circuit=False), [anc] + mu[: r - 2 + right_room] + prod[-right_room:] , inplace=True)
            else:
                if r-2+right_room > 0:
                    qc.compose(CAdder(r-2+right_room, modular=True, circuit=False), [anc] + mu[: r - 2 + right_room] + prod[-right_room:] , inplace=True)

        # Extend sign
        if left_index < r-2:
            qc.cx(prod[left_index-1], prod[left_index])

        qc.cx(mu[n], anc)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='muSquarer(j=%d, r=%d)' %(j, r))

####################################################################################################
def mu2_Multiplier(r, circuit=True):
    mu2= QuantumRegister(r-2, 'mu2')
    x= QuantumRegister(r+3, 'x') # One extension bit
    prod= QuantumRegister(r+2, 'p')

    qc= QuantumCircuit(mu2, x, prod)

    # First partial product
    for n in range(3):
        qc.ccx(mu2[0], x[r+n], prod[n])

    # Sign extension
    qc.cx(prod[2], prod[3])

    # Order of CAdder is (c, a, b, z)
    for n in range(r-3):
        qc.compose(CAdder(n+4, modular=True, circuit=False), [mu2[n+1]] + x[r-1-n:] + prod[:n+4], inplace=True)
        qc.cx(prod[n+3], prod[n+4]) # extend the sign

    qc.cx(prod[-2], prod[-1])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='mu^2_multiplier(%d)' %(r))

####################################################################################################
def mu2_Multiplier_b(b, r, circuit=True):
    '''
    r+2 --> b

    '''
    mu2= QuantumRegister(r-2, 'mu2')
    x= QuantumRegister(b+1, 'x') # One extension bit
    prod= QuantumRegister(b, 'p')

    qc= QuantumCircuit(mu2, x, prod)

    # First partial product
    for n in range(b-r+1):
        qc.ccx(mu2[0], x[r+n], prod[n])

    # Sign extension
    qc.cx(prod[b-r], prod[b-r+1])

    # Order of CAdder is (c, a, b, z)
    for n in range(r-3):
        qc.compose(CAdder(b-r+2+n, modular=True, circuit=False), [mu2[n+1]] + x[r-1-n:] + prod[:b-r+2+n], inplace=True)
        qc.cx(prod[b-r+1+n], prod[b-r+2+n]) # extend the sign

    qc.cx(prod[-2], prod[-1])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='mu^2_multiplier_b(%d)' %(r))

####################################################################################################
def mu2_CCM(r, multStr, circuit=False):
    '''
    Represent the multiplier on r-2 qubits

    '''
    if len(multStr) != r-2:
        print('Error. Returning None...')
        return

    anc= QuantumRegister(1, 'anc')
    mu2= QuantumRegister(r-2, 'mu^2') # Register storing mu^2
    prod= QuantumRegister(r+2, 'p') # Register storing the product

    qc= QuantumCircuit(anc, mu2, prod)

    for n in range(1, r-2):
        if multStr[r-2-n] == '1':
            #if n <= r-3:
            qc.compose(RippleAdder(n, modular=False, circuit=False, twos_comp=False), [anc] + mu2[r-2-n:] + prod[:n+1], inplace=True)
            #else:
            #    qc.compose(RippleAdder(r-2, modular=False, circuit=False), [anc] + mu2[:] + prod[n-r+2:n+1], inplace=True)

    # Subtract Last One
    if multStr[0] == '1':
        # Use two extra qubits of prod as an extension for mu2 (no sign extend required)
        qc.compose(Subtractor2(r, modular=True, circuit=False), [anc] + prod[:r] + mu2[:] + prod[-2:], inplace=True)

    # Then, extend the sign
    qc.cx(prod[r-1], prod[r])
    qc.cx(prod[r], prod[r+1])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='mu^2_CCM(%d)' %(r))

####################################################################################################
def mu2_CCM_b(b, r, multStr, circuit=False):
    '''
    Represent the multiplier on b qubits, r+2 --> b?

    '''
    if len(multStr) != b:
        print('Error. Returning None...')
        return

    anc= QuantumRegister(1, 'anc')
    mu2= QuantumRegister(r-2, 'mu^2') # Register storing mu^2
    prod= QuantumRegister(b, 'p') # Register storing the product

    qc= QuantumCircuit(anc, mu2, prod)

    for n in range(3, b-1):
        if multStr[b-n] == '1':
            if n > r: # r-n < 0
                qc.compose(RippleAdder(r-2, modular=False, circuit=False, twos_comp=False), [anc] + mu2[:] + prod[n-r:n-2], inplace=True)
            else:
                qc.compose(RippleAdder(n-2, modular=False, circuit=False, twos_comp=False), [anc] + mu2[r-n:] + prod[:n-2], inplace=True)
            #else:
            #    qc.compose(RippleAdder(r-2, modular=False, circuit=False), [anc] + mu2[:] + prod[n-r+2:n+1], inplace=True)

    # Subtract Last One
    if multStr[0] == '1':
        # Use two extra qubits of prod as an extension for mu2 (no sign extend required)
        qc.compose(Subtractor2(r-2, modular=True, circuit=False), [anc] + prod[b-1-r:b-3] + mu2[:], inplace=True)

    # Then, extend the sign
    qc.cx(prod[b-3], prod[b-2])
    qc.cx(prod[b-2], prod[b-1])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='mu^2_CCM(%d)' %(r))

####################################################################################################
def FinalMuMultiply(j, r, circuit=False):
    '''
    Here is where we only care about the fractional part, due to pre-conditions

    Another pre: r >= k >= j
    '''
    mu= QuantumRegister(j+2) # One integral qubit (sign)
    x= QuantumRegister(r+3) # One extension qubit
    prod= QuantumRegister(r) # Only care about fractional part of result

    qc= QuantumCircuit(mu, x, prod)

    # Subtract partial sums
    for n in range(j+1):
        if n + 2 >= j + 1: # n= j or j-1
            qc.compose(CSubtractor(r, modular=True, circuit=False), [mu[n]] + x[j+1-n:j+1-n+r] + prod[:], inplace=True)
        else:
            qc.compose(CSubtractor(r+2-j+n, modular=True, circuit=False), [mu[n]] + x[j+1-n:] + prod[:r+2-j+n], inplace=True)

        # Extend sign, if space
        if n + 2 < j:  # r+2-j+n < r
            qc.cx(prod[r+1-j+n], prod[r+2-j+n])

    # Add Last
    qc.compose(CAdder(r, modular=True, circuit=False), mu[-1:] + x[:r] + prod[:], inplace=True)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='final_multiply(%d, %d)' %(j, r))

####################################################################################################
def FinalMuMultiply_b(j, b, r, circuit=False):
    '''
    Here is where we only care about the fractional part, due to pre-conditions

    Another pre: r >= k >= j
    '''
    mu= QuantumRegister(j+2) # One integral qubit (sign)
    x= QuantumRegister(b+1) # One extension qubit r+3 --> b+1
    prod= QuantumRegister(r) # Only care about fractional part of result

    qc= QuantumCircuit(mu, x, prod)

    # Subtract partial sums
    for n in range(j+1):

        if j + 1 - n + r <= b: # full addition, start index + r = j+1-n+r <= b = max index 
            qc.compose(CSubtractor(r, modular=True, circuit=False), [mu[n]] + x[j+1-n:j+1-n+r] + prod[:], inplace=True)
        else: # allowed integer index is higher than actual number of integer bits, also b - j + n - 1 < r
            qc.compose(CSubtractor(b-j+n, modular=True, circuit=False), [mu[n]] + x[j+1-n:b+1] + prod[:b-j+n], inplace=True)
            
        # Extend sign, if space
        if b-j+n <  r:
            qc.cx(prod[b-j+n-1], prod[b-j+n])

    # Add Last
    qc.compose(CAdder(r, modular=True, circuit=False), mu[-1:] + x[:r] + prod[:], inplace=True)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='final_multiply(%d, %d)' %(j, r))