####################################################################################################
# General Imports                                                                                  #
####################################################################################################
import unittest
import sys
import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm
import math
from qiskit import QuantumCircuit, QuantumRegister, execute, quantum_info, Aer


####################################################################################################
# Local Imports                                                                                    #
####################################################################################################

from lattice_qft.Scalar_Field_Theory.classical import *
import lattice_qft.core.basic_circuits as bc
import lattice_qft.Scalar_Field_Theory.basic_operator_implementations as boi



####################################################################################################
# Testing class                                                                                    #
####################################################################################################
class TestCaseUtils(unittest.TestCase):
    '''
        Class that contains useful functions for testing.

        This class is inherited by all testing modules in this package.
    '''


    def checkEqual(self, n1, n2, msg=None):
        with self.subTest():
            self.assertEqual(n1, n2, msg=msg)

    ####################################################################################################
    def compareFidelity(self, sv1, sv2, delta=1e-16, msg=None):
        '''
        Assert that the fidelity between two statevectors is "almost" 1, that is within
        delta of 1.

        :param sv1:   (1D array) 1st statevector
        :param sv2:   (1D array) 2nd statevector
        :param delta: (float)    Tolerated error
        :param msg:   (str)      Eerror message to display if the assertion is False
        '''
        with self.subTest():
            self.assertAlmostEqual(quantum_info.state_fidelity(sv1, sv2), 1.0, msg= msg, delta=delta)

    ####################################################################################################
    def compareSV(self, sv1, sv2, delta=1e-16, msg=None):
        '''
        Assert that two statevectors are "almost" the same, that is the squared absolute value of the 
        inner product of their difference is less than delta.

        :param sv1:   (1D array) 1st statevector
        :param sv2:   (1D array) 2nd statevector
        :param delta: (float)    Tolerated error
        :param msg:   (str)      Error message to display if the assertion is False
        '''
        with self.subTest():
            self.assertAlmostEqual(abs(np.conj(sv1-sv2).dot(sv1-sv2))**2, 0., msg=msg, delta=delta)

    ####################################################################################################
    def compareOperators(self, mat1, mat2, delta=1e-16, msg=None):
        '''
        Assert that two matrices/operators are "almost" the same, that is the Frobenius/Hilbert-Schmidt norm,
        (normalized by the matrix size) of their difference is less than delta.

        :param mat1:   (1D array) 1st matrix
        :param mat2:   (1D array) 2nd matrix
        :param delta: (float)     Tolerated error
        :param msg:   (str)       Error message to display if the assertion is False
        '''
        with self.subTest():
            self.assertAlmostEqual(np.sqrt(np.sum(np.conj(mat1-mat2)*(mat1-mat2)))/mat1.size, 0., msg= msg, delta=delta)

    ####################################################################################################
    @staticmethod
    def DFT(nQ):
        '''
        Contructs the DFT matrix with standard indexing: [0, ... , N-1].

        :param nQ: (int)      Number of qubits --> DFT length is 2^n
        :return:   (2D array) DFT matrix
        '''
        # n = number of qubits
        N= 2**nQ
        i, j = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, N-1, N))
        omega = np.exp( 2 * np.pi * 1j / N )
        W = np.power( omega, i * j ) / np.sqrt(N)
        return W

    ####################################################################################################
    @staticmethod
    def bit_reverse(n):
        '''
        Contructs a mapping from the indices of a statevector to the indicies of that statevector but with reversed bits.
         e.g. for a 2-bit statevector, the map is [0, 1, 2, 3] --> [0, 2, 1, 3], as reversing the bit order leaves 
         00 and 11 in the same position, but swaps 01 and 10.

        :param n: (int)      Number of qubits
        :return:  (1D array) Map
        '''
        out= np.array([0])
        for j in range(n):
            out= np.concatenate((2*out, 2*out + 1))
        return out