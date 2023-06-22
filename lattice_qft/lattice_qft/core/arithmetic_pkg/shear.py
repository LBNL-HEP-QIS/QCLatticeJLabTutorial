'''
    Contains functions to construct a shearing circuit.

    The action of construct_shear_circuit_theoretical() is replicated by the function
            createKWground()
    in classical.py.
'''


####################################################################################################
# General Imports                                                                                  #
####################################################################################################
from qiskit import *
import sys
import numpy as np

####################################################################################################
# Local Imports                                                                                  #
####################################################################################################

import lattice_qft.core.arithmetic_pkg.arithmetic as arithmetic
import lattice_qft.core.arithmetic_pkg.series as series

#import lattice

####################################################################################################
# QuantumCircuit Utility Functions                                                                 #
####################################################################################################
def extend_sign(K, circuit=False):
    '''
    Constructs a new QuantumCircuit that extends a sign qubit K times by applying K CNOTs. 
    The 0-indexed qubit is the sign bit.

    :param K:       (int)                 Binary precision of inputs
    :param circuit: (bool)                If true, returns the circuit as a QuantumCircuit; 
                                              if false, returns a the circuit as a Gate
    :return:        (QuantumCircuit(K+1))
    '''
    if K <= 0:
        print('Error: K must be postive.')
        return

    qc= QuantumCircuit(K+1)
    for k in range(K):
        qc.cx(k, k+1)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='extend_sign')


####################################################################################################
# Utility Functions                                                                                #
####################################################################################################
def generate_Abin(M, i, k, r):
    '''
    Generates the binary representation of row i of the shear matrix, with k integral bits 
    and r decimal bits. Only off-diagonal entries are included in the output, as the diagonal
    is populated by ones.

    :param M: (Array)     Shear Matrix
    :param i: (int)       Row
    :param k: (int)       Number of binary digits before decimal / Number of qubits per lattice site
    :param r: (int)       Number of digits after decimal

    :return:  (List[str]) Off-diagonal entries of row i of M, as binary strings
    '''
    row= M[i, i+1:]
    Ab= []
    for j in range(row.size):
        Ab.append(series.decimal_to_binary2(row[j], k+r, r))
    return Ab


####################################################################################################
# Shearing Circuit Generation                                                                      #
####################################################################################################
def construct_shear_row_theoretical(S, i, k, r, A, circuit=False):
    '''
    Constructs a new QuantumCircuit that implements the product of row i of the shearing matrix with vector |n>.

    :param S:       (int)                   Shear matrix dimension = Number of lattice sites
    :param i:       (int)                   Row
    :param k:       (int)                   Number of qubits per lattice site
    :param r:       (int)                   Number of fractional qubits
    :param A:       (List[string])          Off-diagonal entries of row i of the shear matrix, as binary strings
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; 
                                                if false, returns a the circuit as a Gate
    :return:        (QuantumCircuit / Gate)
    '''
    if i == S-1: # identity transformation
        print('i= S-1 is the identity. Returning empty QuantumCircuit(p+m).')
        return QuantumCircuit(k+r)
    if i >= S:
        print('i must be less than S: returning None')
        return

    if r > 0:
        anc= QuantumRegister(1, 'anc')
        dec= QuantumRegister(r, 'dec')
        ext= QuantumRegister(r, 'ext')
        qc= QuantumCircuit(anc, dec, ext)

        # Add |n> registers
        nDic= {}
        for j in range(S-i):
            nDic[i+j]= QuantumRegister(k, name='n%d' %(i+j))
            qc.add_register(nDic[i+j])

        # Add 1/2 to n_i -- Really we want to shear half integers, in mean centered coordinates
        qc.x(dec[r-1])
        # Add 1/2 to the n_j by appending an additional '1' to the right. To do this, we use ext[-1]
        # but that's how it's implemented in this update.
        qc.x(ext[-1])

        for j in range(S-i-1):
            # Sign extend n_j through |ext>, so |ext>|nj> = |nj> combined is k + r bits. 
            # However, leave the last bit of |ext> open, as it is only needed for n=0 below.
            qc.compose(extend_sign(r-1), [nDic[j+i+1][-1]] + ext[:-1], inplace=True)

            for n in range(k+r):
                if A[j][k+r-n-1] == '1': # Check if the nth bit of a_ij is one
                    if n <= r:
                        if n == 0:
                            # Make ext[-1] usable.
                            qc.x(ext[-1])
                            qc.cx(nDic[j+i+1][-1], ext[-1])

                            qc.compose(arithmetic.RippleAdder(k+r, modular=True, circuit=False), anc[:] + nDic[j+i+1][:] + ext[:r] + dec[:r] + nDic[i][:], inplace=True) # k + m - n
                            
                            # Return ext[-1] to the |1> state
                            qc.cx(nDic[j+i+1][-1], ext[-1])
                            qc.x(ext[-1])
                        else: # n >= 1
                            # Can use ext[-1] as the 1/2 place |1> for nj, as ext[-1] is only used as an extension qubit for n= 0.
                            qc.compose(arithmetic.RippleAdder(k+r-n+1, modular=True, circuit=False), anc[:] + ext[-1:] + nDic[j+i+1][:] + ext[:r-n] + dec[n-1:r] + nDic[i][:], inplace=True)
                    else:
                        if n == k + r - 1: # subtract
                            qc.compose(arithmetic.Subtractor2(k+r-n+1, modular=True, circuit=False), anc[:] + nDic[i][n-r-1:] + ext[-1:] + nDic[j+i+1][:k+r-n], inplace=True)
                        else:
                            qc.compose(arithmetic.RippleAdder(k+r-n+1, modular=True, circuit=False), anc[:] + ext[-1:] + nDic[j+i+1][:k+r-n] + nDic[i][n-r-1:], inplace=True) # k + m - n

            # Undo sign extension
            qc.compose(extend_sign(r-1).reverse_ops(), [nDic[j+i+1][k-1]] + ext[:-1], inplace=True)

        # The half integers are represented by their floors in two's complement. 
        # Note that rounding to the nearest half integer, then taking the floor, is the same as just taking the floor.
        # Therefore, no physical operations are required to "round" the results;
        # The integer part of the result, stored in nDic[i] is just the floor of the result, which is what we want. 

        # Uncompute m least significant bits of addition: |dec> register
        for j in reversed(range(S-i-1)):
            # Sign extend n_j through |ext>, so |ext>|nj> = |nj> combined is k + m bits
            if k < r:
                qc.compose(extend_sign(r-k), [nDic[j+i+1][k-1]] + ext[:r-k], inplace=True)

            # k > 0 by defintion, so ext[-1] is never used for extension here. Therefore it can be used as the 1/2-place qubit of the n_j.    

            for n in reversed(range(r+1)):
                if A[j][k+r-n-1] == '1': # Check if the nth bit of a_ij is one
                    if n == r:
                        qc.compose(arithmetic.RippleAdder(r-n+1, modular=True, circuit=False).reverse_ops(), anc[:] + ext[-1:] + dec[n-1:r], inplace=True)
                    elif r - k >= n:
                        if n == 0:
                            qc.compose(arithmetic.RippleAdder(r, modular=True, circuit=False).reverse_ops(), anc[:] + nDic[j+i+1][:] + ext[:r-n-k] + dec[:], inplace=True) # m - n
                        else:
                            qc.compose(arithmetic.RippleAdder(r-n+1, modular=True, circuit=False).reverse_ops(), anc[:] + ext[-1:] + nDic[j+i+1][:] + ext[:r-n-k] + dec[n-1:r], inplace=True) # m - n
                    else:
                        qc.compose(arithmetic.RippleAdder(r-n+1, modular=True, circuit=False).reverse_ops(), anc[:] + ext[-1:] + nDic[j+i+1][:r-n] + dec[n-1:r], inplace=True) # m - n

            # Undo sign extension
            if k < r:
                qc.compose(extend_sign(r-k).reverse_ops(), [nDic[j+i+1][k-1]] + ext[:r-k], inplace=True)

        # Subtract 1/2 from n_i (uncompute)
        qc.x(dec[-1])
        # Uncompute
        qc.x(ext[-1])

    else: # r = 0, never happens in practice.
        anc= QuantumRegister(1, 'anc')
        qc= QuantumCircuit(anc)

        # Add |n> registers
        nDic= {}
        for j in range(S-i):
            nDic[i+j]= QuantumRegister(k, name='n%d' %(i+j))
            qc.add_register(nDic[i+j])

        for j in range(S-i-1):
            for n in range(k):
                if A[j][k-n-1] == '1': # Check if the nth bit of a_ij is one
                    qc.compose(arithmetic.RippleAdder(k-n, modular=True, circuit=False), anc[:] + nDic[j+i+1][:k-n] + nDic[i][n:], inplace=True) # k + m - n

        # No Rounding needed if there are no decimal qubits

    if circuit:
        return qc
    else:
        return qc.to_gate(label='Compute x%d' %(i))

####################################################################################################
def construct_shear_circuit_theoretical(S, k, M, dim= 1, dx= 1, circuit=False, subcircuit=False):
    '''
    Construct a new QuantumCircuit that applies the entire shearing matrix.

    :param S:          (int)                   Shear matrix dimension = Number of lattice sites
    :param k:          (int)                   Number of qubits per lattice site
    :param M:          (int)                   Matrix M from the MDM decomposition
    :param circuit:    (bool)                  If true, returns the circuit as a QuantumCircuit; 
                                                   if false, returns a the circuit as a Gate
    :param subcircuit: (bool)                  If true, returns the circuit as a QuantumCircuit, 
                                                   which component gates also expanded into circits
    :return:           (QuantumCircuit / Gate)
    '''

    # Compute number of fractional qubits to use.
    if S <= 2: # Never the case in practice, but this is a precondition that ensures r >= k.
        r= k
    else:
        r= int(k - 1 + np.ceil(np.log2(S-1)))# Note: r >= k
        #r= int(np.ceil(np.log2(2**(k-1) + 1) + np.log2(S-1)))# NEW

    anc= QuantumRegister(1, 'anc')
    qc= QuantumCircuit(anc)
    if r > 0:
        dec= QuantumRegister(r, 'dec')
        ext= QuantumRegister(r, 'ext') # Extension Register
        qc.add_register(dec)
        qc.add_register(ext)

    nDic= {}    # Now registers are added in order from 0 to S-1
    for j in range(S):
        nDic[j]= QuantumRegister(k, name='n%d' %(j))
        qc.add_register(nDic[j])
    
    if r > 0:
        for i in range(S-1):
            regList= anc[:] + dec[:] + ext[:]
            for j in range(S-i):
                regList+= nDic[i+j]
            qc.compose(construct_shear_row_theoretical(S, i, k, r, generate_Abin(M, i, k, r), subcircuit), regList, inplace=True)

    else: # r = 0, never happens in practice
        for i in range(S-1):
            regList= anc[:]
            for j in range(S-i):
                regList+= nDic[i+j]
            qc.compose(construct_shear_row_theoretical(S, i, k, 0, generate_Abin(M, i, k, 0), subcircuit), regList, inplace=True)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='Shearing')