'''
    Contains some utility functions that are useful.
'''


####################################################################################################
# General Imports                                                                                  #
####################################################################################################
from qiskit.circuit import Qubit
from qiskit import QuantumCircuit
from math import *
import numpy as np


####################################################################################################
# Functions                                                                                        #
####################################################################################################
def decimal_to_binary(n, K, d):
    '''
    Converts a decimal number to a binary string, truncating insignificant bits.

    :param n: (float) a number to convert to binary, with
    :param K: (int)   total binary digits
    :param d: (int)   digits after the decimal
    :return:  (str)   Binary representation of n
    '''
    if n == 0:
        return '0' * K

    nScaled= int(np.floor(n * (2**d)))

    # Overflow inputs are returned as the extremal value (most positive or most negative)
    if nScaled >= 2**(K-1):
        print('Input %f out of bounds for K= %d, d=%d. Returning 0111...' %(n, K, d))
        return '0' + '1'*(K-1)
    
    elif nScaled < -2**(K-1):
        print('Input %f out of bounds for K= %d, d=%d. Returning 1000...' %(n, K, d))
        return '1' + '0'*(K-1)

    else:
        b= int_to_binary(nScaled)
        if len(b) < K:
            return (K-len(b))*b[0] + b
        else:
            return b

####################################################################################################
def decimal_to_binary2(n, K, d):
    '''
    Converts a decimal number to a binary string, rounding insignificant bits.
    
    :param n: (float) a number to convert to binary, with
    :param K: (int)   total binary digits
    :param d: (int)   digits after the decimal
    :return:  (str)   Binary representation of n
    '''
    if n == 0:
        return '0' * K

    nScaled= int(np.round(n * (2**d)))

    # Overflow inputs are returned as the extremal value (most positive or most negative)
    if nScaled >= 2**(K-1):
        print('Input %f out of bounds for K= %d, d=%d. Returning 0111...' %(n, K, d))
        return '0' + '1'*(K-1)
    
    elif nScaled < -2**(K-1):
        print('Input %f out of bounds for K= %d, d=%d. Returning 1000...' %(n, K, d))
        return '1' + '0'*(K-1)

    else:
        b= int_to_binary(nScaled)
        if len(b) < K:
            return int(K-len(b))*b[0] + b
        else:
            return b

####################################################################################################
def binary_to_int(b):
    '''
    Converts a binary string into the two's complement integer it represents.

    :param b: (str) Binary string
    :return:  (int) Two's complement integer represented by b
    '''
    return int(b[1:], 2) - int(b[0] + '0'*(len(b)-1), 2)

####################################################################################################
def posint_to_binary(n):
    '''
    Converts a binary string into the positive integer is represents.

    :param n: (int) Positive integer
    :return:  (str) Binary representation of n (positive, not two's complement)
    '''
    if n < 0 or type(n) != int:
        print('Input is not a positive integer. Returning 0...')
        return '0'
    else:
        return bin(n)[2:]

####################################################################################################
def int_to_binary(n):
    '''
    Converts an integer into it's two's complement binary representation.

    :param n: (int) Integer
    :return:  (str) Two's complement binary representation of n
    '''
    if type(n) == float:
        n= int(n)
    if type(n) != int:
        print('Input is not an integer or a float. Returning 0...')
        return '0'
    else:
        if n >= 0:
            return '0' + bin(n)[2:]
        else:
            signPos= ceil(log2(abs(n)))
            posComp= int(2**ceil(log2(abs(n))) + n)
            #print('signPos: ' + str(signPos))
            #print('posComp: ' + str(posComp))
            if signPos == 0:
                return '1'
            else:
                return '1' + format(posComp, '0%db' %(signPos))

####################################################################################################
def binary_to_decimal(b, d):
    '''
    Converts a binary string into the two's complement decimal number it represents.

    :param b: (str)   Binary string
    :param d: (int)   Number of digits after the decimal
    :return:  (float) Two's complement decimal represented by b
    '''
    return (int(b[1:], 2) - int(b[0] + '0'*(len(b)-1), 2)) * (2**(-d))

####################################################################################################
def Qinit(qc, qubits, n):
    '''
    Initializes a quantum circuit to a particular basis state.

    :param qc:     (QuantumCircuit)
    :param qubits: (list(Qubit))    Qubits to initialize
    :param n:      (int)            computational basis state to initialize, in two's complement   

    :return:       None
    '''
    if isinstance(qubits, Qubit):
        N= 1
    else:
        N= len(qubits)
    state_vector= [0]*(2**N)

    if n >= 2**(N-1) or n < -2**(N-1):
        print('n=%d not in range. Initializing to zero.' %(n))
        state_vector[0]= 1
        qc.initialize(state_vector, qubits)
        return

    if n < 0:
        n+= 2**N

    state_vector= [0]*(2**N)
    state_vector[n]= 1
    
    qc.initialize(state_vector, qubits)
    return
def encode(K, aBinary, circuit=False):
    '''
    Constructs a new QuantumCircuit that encodes a binary string by applying NOTs at the locations
    of ones in the string.

    :param K:       (int)                   Register length
    :param aBinary: (str)                   Binary string to encode on qubits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; 
                                                if false, returns a the circuit as a Gate
    :return:        (QuantumCircuit / Gate)
    '''
    pm= len(aBinary)
    qc= QuantumCircuit(K)
    for j in range(pm):
        if aBinary[pm-j-1] == '1':
            qc.x(j)
            
    if circuit:
        return qc
    else:
        return qc.to_gate(label='Encode')

####################################################################################################