'''
    Contains functions to construct circuits to evaluate alpha using series approximations.
'''


####################################################################################################
# General Imports                                                                                  #
####################################################################################################
import numpy as np


####################################################################################################
# Local Imports                                                                                    #
####################################################################################################
from arithmetic import *
from alpha_arithmetic import *
from testing_utilities import int_to_binary, decimal_to_binary2


####################################################################################################
# Functions                                                                                        #
####################################################################################################
def Parallel_Series_2coeffs(b, r, j, d, aList1, aList2, circuit=False, test_counts=False):
    '''
    Constructs a new QuantumCircuit that evaluates a series expansion according to
    some given coefficients and input value.

    :param b:           (int)                   Binary precision of arithmetic
    :param r:           (int)                   Number of bits to the right of the decimal
    :param j:           (int)                   Iteration number
    :param d:           (int)                   Degree of series, >= 2
    :param aList1:      (List[float])           of coefficients of an n-degree series -- [a1(n), a1(n-1), ... , a1(1), a1(0)]
    :param aList2:      (List[float])           of coefficients of an n-degree series -- [a2(n), a2(n-1), ... , a2(1), a2(0)]
    :param circuit:     (bool)                  If True, returns a QuantumCircuit. If False, returns a Gate.
    :param test_counts: (bool)                  If True, the coefficients are all '11...11' to obtain the maximum possible gate count

    :return:            (QuantumCircuit / Gate)

    Label key:
        00 = top,     mu'  > muc
        01 = bottom,  mu'  < muc
        10 = middle, |mu'| < muc        
    '''
    anc= QuantumRegister(1)
    mu= QuantumRegister(j+3)
    mu2= QuantumRegister(r-2)
    a= QuantumRegister(b)
    lab= QuantumRegister(2)

    qc= QuantumCircuit(anc, mu, mu2, a, lab)

    if len(aList1) != d+1 or len(aList2) != d+1:
        print('Coefficient list has incorrect length. Exiting...')
        return

    xDic= {}
    for i in range(1, d+1):
        xDic[i]= QuantumRegister(b, name='x%d' %(i))
        qc.add_register(xDic[i])

    # Transform the aList to binary
    aList1_bin= []
    aList2_bin= []
    for i in range(d+1):
        if test_counts == False:
            aList1_bin.append(decimal_to_binary2(aList1[i], b, r))
            aList2_bin.append(decimal_to_binary2(aList2[i], b, r))
        else:
            aList1_bin.append('1'*b)
            aList2_bin.append('1'*b)

    # TODO: Initialize label

    # Subtract 1/2 from mu register, to take -mu --> -mu'
    qc.cx(mu[-3], mu[-2])
    qc.x(mu[-2])
    qc.x(mu[-3])
    qc.cx(mu[-2], mu[-1])

    # Square mu'
    qc.compose(muSquarer(j, r, circuit=False), anc[:] + mu[:] + mu2[:], inplace=True)

    # Horner iteration
    for i in range(1, d+1):
        if i == 1:
            # Encode a_d
            qc.compose(encode_coeff(b, r, aList1_bin[-1], aList2_bin[-1]), a[:] + lab[:1], inplace=True)
            qc.cx(a[-1], anc)
            qc.compose(mu2_Multiplier_b(b, r, circuit=False), mu2[:] + a[:] + anc[:] + xDic[i][:], inplace= True)
            qc.cx(a[-1], anc)
            qc.compose(encode_coeff(b, r, aList1_bin[-1], aList2_bin[-1]), a[:] + lab[:1], inplace=True)
        else:
            qc.cx(xDic[i-1][-1], anc)
            qc.compose(mu2_Multiplier_b(b, r, circuit=False), mu2[:] + xDic[i-1][:] + anc[:] + xDic[i][:], inplace= True)
            qc.cx(xDic[i-1][-1], anc)

        qc.compose(encode_coeff(b, r, aList1_bin[-i-1], aList2_bin[-i-1]), a[:] + lab[:1], inplace=True)
        qc.compose(RippleAdder(b, modular=True, circuit=False), anc[:] + a[:] + xDic[i][:], inplace=True)
        qc.compose(encode_coeff(b, r, aList1_bin[-i-1], aList2_bin[-i-1]), a[:] + lab[:1], inplace=True)

    # The "controlled"-multiply

    # Extend the multiplicand by one qubit
    qc.cx(xDic[d][-1], anc)

    # Free up a register to house the multiplier if d > 1. Otherwise house the multiplier in a new register
    if d > 1:
        qc.compose(encode_coeff(b, r, aList1_bin[-1], aList2_bin[-1]), a[:] + lab[:1], inplace=True)
        qc.cx(a[-1], anc)
        qc.compose(mu2_Multiplier_b(b, r, circuit=False).reverse_ops(), mu2[:] + a[:] + anc[:] + xDic[1][:], inplace= True)
        qc.cx(a[-1], anc)
        qc.compose(encode_coeff(b, r, aList1_bin[-1], aList2_bin[-1]), a[:] + lab[:1], inplace=True)

        # Initialize xDic[1] register to 1, -1, or mu'
        qc.compose(encode_multiplier(b, r, j), lab[:] + xDic[1][:] + mu[:j+2], inplace=True)
        qc.compose(ModularMultiplier3(b, r, circuit=False), xDic[1][:] + xDic[d][:] + anc[:] + a[:] , inplace=True)
        qc.compose(encode_multiplier(b, r, j).reverse_ops(), lab[:] + xDic[1][:] + mu[:j+2], inplace=True)

        # Return xDic[1] register to first partial sum
        qc.compose(encode_coeff(b, r, aList1_bin[-1], aList2_bin[-1]), a[:] + lab[:1], inplace=True)
        qc.cx(a[-1], anc)
        qc.compose(mu2_Multiplier_b(b, r, circuit=False), mu2[:] + a[:] + anc[:] + xDic[1][:], inplace= True)
        qc.cx(a[-1], anc)
        qc.compose(encode_coeff(b, r, aList1_bin[-1], aList2_bin[-1]), a[:] + lab[:1], inplace=True)

    else: # House the multiplier in a new register if d= 1.
        extra= QuantumRegister(b)
        qc.add_register(extra)
        qc.compose(encode_multiplier(b, r, j), lab[:] + extra[:] + mu[:j+2], inplace=True)
        qc.compose(ModularMultiplier3(b, r, circuit=False), extra[:] + xDic[d][:] + anc[:] + a[:] , inplace=True)
        qc.compose(encode_multiplier(b, r, j).reverse_ops(), lab[:] + extra[:] + mu[:j+2], inplace=True)

    qc.cx(xDic[d][-1], anc)

    # For the last little addition, only care about fractional part. Thus only have to add 1/2 if in middle region
    qc.cx(lab[0], a[r-1])

    if circuit:
        return qc
    else:
        return qc.to_gate(label= '%d-degree Series' %(d))

####################################################################################################
def encode_coeff(b, r, coeff1_bin, coeff2_bin, circuit=False):
    '''
    Encodes coeff1 on a b-qubit register conditioned on the label qubit (lab) being 1.
    Encodes coeff2 on a b-qubit register conditioned on the label qubit (lab) being 0.
    
    :param b:          (int)                   Binary precision of arithmetic
    :param r:          (int)                   Number of bits to the right of the decimal
    :param coeff1_bin: (str)                   Binary string representing coefficient 1
    :param coeff2_bin: (str)                   Binary string representing coefficient 2

    :return:           (QuantumCircuit / Gate)
    '''
    a= QuantumRegister(b)
    lab= QuantumRegister(1)

    qc= QuantumCircuit(a, lab)

    for i in range(b):
        if coeff1_bin[-i] == '1':
            qc.cx(lab, a[i])

    qc.x(lab)
    for i in range(b):
        if coeff2_bin[-i] == '1':
            qc.cx(lab, a[i])
    qc.x(lab)

    if circuit:
        return qc
    else:
        return qc.to_gate(label= 'Encode Coeff.')

####################################################################################################
def encode_multiplier(b, r, j, circuit=False):
    '''
    Depending on the label, encode:
        00 --> 1
        01 --> -1
        10 --> mu'

    Precondition: r >= j+1, b > r

    :param b:         (int)                   Binary precision of arithmetic
    :param r:         (int)                   Number of bits to the right of the decimal
    :param j:         (int)                   Iteration number

    :return:          (QuantumCircuit / Gate)
    '''

    if r < j+1:
        print('r < j+1 does not work. Exiting...')
        return

    lab= QuantumRegister(2)
    x= QuantumRegister(b)
    mu= QuantumRegister(j+2) # The extra sign qubit isn't needed
    
    qc= QuantumCircuit(lab, x, mu)

    # First, do lab= 10
    for i in range(j+2):
        qc.ccx(lab[0], mu[i], x[i+r-j-1]) # Note: goes up to ones place

    # Extend sign to two's place
    if b > r+1:
        qc.cx(x[r], x[r+1])

    # Now do lab= 00 nad lab= 01
    qc.x(lab[0])
    qc.cx(lab[0], x[r])
    if b > r+1:
        qc.ccx(lab[0], lab[1], x[r+1])
    qc.x(lab[0])

    # Extend sign
    for i in range(2, b-r):
        qc.cx(x[r+1], x[r+i])

    if circuit:
        return qc
    else:
        return qc.to_gate(label= 'Encode Mult.')

####################################################################################################
def coeffs(n, sigma):
    '''
    Generate an array containing the first n coefficients of the scaled sine series

    :param n:     (n)     Number of coefficients to generate
    :param sigma: (float) Std. Dev. of Gaussian, used in the prefactor

    :return:      (Array) of coefficients
    '''
    prefact= (-2 / np.pi) * np.exp(-(np.pi**2 * sigma**2) / 2)
    
    powerArray= np.arange(1, 2*n + 1, 2)
    factArray= np.zeros(powerArray.size)
    
    for n in range(powerArray.size):
        factArray[n]= np.math.factorial(powerArray[n])

    coArray=  (-1)**((powerArray-1)/2) * (np.pi ** powerArray) / factArray
    return prefact * coArray

####################################################################################################
def alpha_large_sigma(j, r, sigma, d, circuit=False, test_counts=False):
    '''
    Constructs a new QuantumCircuit that alpha in the large sigma regime.

    :param j:           (int)                   Iteration number
    :param r:           (int)                   Number of bits to the right of the decimal
    :param sigma:       (float)                 sigma_j
    :param d:           (int)                   Degree of series, >= 2
    :param circuit:     (bool)                  If True, returns a QuantumCircuit. If False, returns a Gate.
    :param test_counts: (bool)                  If True, the coefficients are all '11...11' to obtain the maximum possible gate count

    :return:            (QuantumCircuit / Gate)

    Label key:
        00 = top,     mu'  > muc
        01 = bottom,  mu'  < muc
        10 = middle, |mu'| < muc        
    '''
    mu= QuantumRegister(j+3, 'mu') # mu register. Has 1 sign qubit, j fractional qubits, and 1 classical bit (least significant). Also 1 extension bit
    anc= QuantumRegister(1, 'anc') # Ancillary qubit, to copy multiplier bits to
    square= QuantumRegister(r-2, 'prod') # Result is guaranteed < 1/4, so has r-2 bits
    CCMprod= QuantumRegister(r+2, 'p')
    alpha= QuantumRegister(r+2, 'alpha')

    qc= QuantumCircuit(mu, anc, square, CCMprod)

    xDic= {}
    for i in range(2, d+1):
        xDic[i]= QuantumRegister(r+2, name='x%d' %(i))
        qc.add_register(xDic[i])

    qc.add_register(alpha)

    coeff_list= coeffs(d+1, sigma)
    #print(coeff_list)
    if test_counts == False: # Actual Mode
        a_d= decimal_to_binary2(coeff_list[-1], r-2, r-3)
    else: # Make the CCM coefficient all ones
        a_d= '1' * (r-2)

    # Subtract 1/2 from mu register
    qc.cx(mu[-3], mu[-2])
    qc.x(mu[-2])
    qc.x(mu[-3])
    qc.cx(mu[-2], mu[-1])

    # Square mu
    qc.compose(muSquarer(j, r, circuit=False), anc[:] + mu[:] + square[:], inplace=True)
    
    # First Horner iterate
    if d >= 1:
        qc.compose(mu2_CCM(r, a_d, circuit=False), anc[:] + square[:] + CCMprod[:], inplace=True)
        a_d1= decimal_to_binary2(coeff_list[-2], r+2, r)
        qc.compose(encode(r+2, a_d1, circuit=False), alpha[:], inplace=True)
        qc.compose(RippleAdder(r+2, modular=True, circuit=False), anc[:] + alpha[:] + CCMprod[:], inplace=True)
        qc.compose(encode(r+2, a_d1, circuit=False), alpha[:], inplace=True)

    # Remaining Horner iterates
    for i in range(2, d+1): # d
        if i == 2:
            # Extend x by one
            qc.cx(CCMprod[-1], anc)
            qc.compose(mu2_Multiplier(r, circuit=False), square[:] + CCMprod[:] + anc[:] + xDic[i][:], inplace=True)
            qc.cx(CCMprod[-1], anc)
        else:
            # Extend x by one
            qc.cx(xDic[i-1][-1], anc)
            qc.compose(mu2_Multiplier(r, circuit=False), square[:] + xDic[i-1][:] + anc[:] + xDic[i][:], inplace=True)
            qc.cx(xDic[i-1][-1], anc)

        # Add next coefficient
        a_i= decimal_to_binary2(coeff_list[-i], r+2, r)
        if i < d:
            qc.compose(encode(r+2, a_i, circuit=False), xDic[i+1][:], inplace=True)
            qc.compose(RippleAdder(r+2, modular=True, circuit=False), anc[:] + xDic[i+1][:] + xDic[i][:], inplace=True)
            qc.compose(encode(r+2, a_i, circuit=False), xDic[i+1][:], inplace=True)
        else:
            qc.compose(encode(r+2, a_i, circuit=False), alpha[:], inplace=True)
            qc.compose(RippleAdder(r+2, modular=True, circuit=False), anc[:] + alpha[:] + xDic[i][:], inplace=True)
            qc.compose(encode(r+2, a_i, circuit=False), alpha[:], inplace=True)

    # Multiply the even-powered series by mu
    if d >= 2:
        qc.cx(xDic[d][-1], anc)
        qc.compose(FinalMuMultiply(j, r, circuit=False), mu[:j+2] + xDic[d][:] + anc[:] + alpha[:r], inplace=True)
        qc.cx(xDic[d][-1], anc)
    elif d == 1:
        qc.cx(CCMprod[-1], anc)
        qc.compose(FinalMuMultiply(j, r, circuit=False), mu[:j+2] + CCMprod[:] + anc[:] + alpha[:r], inplace=True)
        qc.cx(CCMprod[-1], anc)

    # Add 1/2
    qc.x(alpha[r-1])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='alpha_large_sigma(j=%d, r=%d)' %(j, r))