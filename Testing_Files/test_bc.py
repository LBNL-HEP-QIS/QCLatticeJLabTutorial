####################################################################################################
# General Imports                                                                                  #
####################################################################################################
import unittest
import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm
import math
import sys
sys.path.append('./')
sys.path.append('../')
from qiskit import QuantumCircuit, QuantumRegister, execute, quantum_info, Aer


####################################################################################################
# Local Imports                                                                                    #
####################################################################################################
from test_base import TestCaseUtils # custom class that includes utility functions for testing
from Scalar_Field_Theory.classical import *
import modules.basic_circuits as bc
import Scalar_Field_Theory.basic_operator_implementations as boi
import modules.lattice as lattice

global simulator_state
simulator_state= Aer.get_backend('statevector_simulator')


####################################################################################################
# Testing class                                                                                    #
####################################################################################################
class bcTestCase(TestCaseUtils):
    '''
        Class for testing the functions defined in basic_circuits.py.
    '''


    def test_fourier_transform(self):
        '''
        Things to test:
            odd vs. even length            -- check
            inverse vs. regular            -- check
            swap vs. no swap               -- check

        Add edge cases? Probably not, random statevectors are sufficiently general.

        Note: Quantum circuit output has to be reversed (swap=True) to align with the DFT matrix product.
        '''
        print('\nTesting fourier_transform...')
        # Test forward transform
        for nQ in range(2, 5):
            for swap in [True, False]:
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)
                qc.compose(bc.fourier_transform(q, inv=False, swap=swap), inplace=True)

                mat_qc= quantum_info.Operator(qc).data    # Quantum Circuit
                mat_exp= TestCaseUtils.DFT(nQ)               # Expected (classical)

                if not swap: # Quantum Fourier transform output states are reverse-indexed by qubit
                    bit_map= TestCaseUtils.bit_reverse(nQ)
                    mat_exp= mat_exp[bit_map,:]

                msg= 'QuantumCircuit does not implement the expected Fourier transform operator. Params: nQ= %d, swap= %s' %(nQ, swap)
                self.compareOperators(mat_qc, mat_exp, delta=1e-14, msg= msg)

        # Test inverse transform
        for nQ in range(2, 5):
            for swap in [True, False]:
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)
                qc.compose(bc.fourier_transform(q, inv=True, swap=swap), inplace=True)

                mat_qc= quantum_info.Operator(qc).data    # Quantum Circuit
                mat_exp= TestCaseUtils.DFT(nQ)               # Expected (classical)

                if not swap: # Quantum Fourier transform output states are reverse-indexed by qubit
                    bit_map= TestCaseUtils.bit_reverse(nQ)
                    mat_exp= mat_exp[bit_map,:]
                
                mat_exp= inv(mat_exp)

                msg= 'QuantumCircuit does not implement the expected inverse Fourier transform operator. Params: nQ= %d, swap= %s' %(nQ, swap)
                self.compareOperators(mat_qc, mat_exp, delta=1e-14, msg= msg)

    ####################################################################################################
    def test_phase(self):
        '''
        Things to test:
            different phase values    -- <-2π, -2π, <0, 0, <2π, 2π, >2π
            odd vs. even circuit size -- check

        Add edge cases? Probably not, random statevectors are sufficiently general.
        '''
        print('\nTesting phase...')
        for nQ in range(1, 5):
            for theta in np.pi * np.array([-2.43, -2, -1.38, 0, 1.15, 2, 4.56]):
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)
                qc.compose(bc.phase(q, theta), inplace=True)
                
                mat_qc= quantum_info.Operator(qc).data
                mat_exp= np.exp(1j * theta) * np.identity(2**nQ)

                msg= 'QuantumCircuit does not implement the expected phase operator. Params: nQ= %d, θ= %.4f' %(nQ, theta)
                self.compareOperators(mat_qc, mat_exp, delta=1e-14, msg= msg)

    ####################################################################################################
    def test_fourier_transform_symmetric(self):
        '''
        Things to test:
            odd vs. even length            -- check
            inverse vs. regular            -- check
            swap vs. no swap               -- check

        Add edge cases? Probably not, random statevectors are sufficiently general.

        Note: Quantum circuit output has to be reversed (swap=True) to align with the DFT matrix product.
        '''
        print('\nTesting fourier_transform_symmetric...')
        # Test forward transform
        for nQ in range(2, 5):
            for swap in [True, False]:
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)
                qc.compose(bc.fourier_transform_symmetric(q, inv=False, swap=swap), inplace=True)

                mat_qc= quantum_info.Operator(qc).data
                mat_exp= DFT_phi(2**nQ)
                if not swap:
                    bit_map= TestCaseUtils.bit_reverse(nQ)
                    mat_exp= mat_exp[bit_map,:]

                msg= 'QuantumCircuit does not implement the expected Fourier transform operator. Params: nQ= %d, swap= %s' %(nQ, swap)
                self.compareOperators(mat_qc, mat_exp, delta=1e-14, msg= msg)

        # Test inverse transform
        for nQ in range(2, 5):
            for swap in [True, False]:
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)
                qc.compose(bc.fourier_transform_symmetric(q, inv=True, swap=swap), inplace=True)

                mat_qc= quantum_info.Operator(qc).data    # Quantum Circuit
                mat_exp= iDFT_phi(2**nQ)                         # Expected (classical)

                if not swap: # Quantum Fourier transform output states are reverse-indexed by qubit
                    bit_map= TestCaseUtils.bit_reverse(nQ)
                    mat_exp= mat_exp[:, bit_map] # Input is reverse-qubit order of QC for the inverse transform, so the columns have to be reordered

                msg= 'QuantumCircuit does not implement the expected inverse Fourier transform operator. Params: nQ= %d, swap= %s' %(nQ, swap)
                self.compareOperators(mat_qc, mat_exp, delta=1e-14, msg= msg)

    ####################################################################################################
    def test_exp_pauli_product(self):
        '''
        Note: this just tests exp_pauli_product against an equivalent qiskit circuit that applies X, Y gates directly. 
          (No comparison against a classical computation.)
        Things to test:
            different q size          -- check
            different theta values    -- check
            different Pauli operators -- check

        Add edge cases? Probably not, random statevectors are sufficiently general.
        '''
        print('\nTesting exp_pauli_product...')
        for pauli in ['IZ', 'ZI', 'XI', 'IY', 'XY', 'ZIZ', 'ZIX', 'YYZ']:
            for theta in [0, 0.434, 1, 1.67, -1.67, -1, -0.434]:
                pauli_list= bc.get_pauli_list(pauli) # MSB is index 0 in the Pauli strings
                nQ= len(pauli)
                q= QuantumRegister(nQ)
                qc= QuantumCircuit(q)
                qc.compose(bc.exp_pauli_product(q, theta, pauli_list), inplace=True)

                mat_qc= quantum_info.Operator(qc).data
                mat_exp= np.array([1])
                for i in range(len(pauli)):
                    op= pauli[i]
                    if op == 'I':
                        mat_exp= np.kron(mat_exp, np.identity(2))
                    elif op == 'Z':
                        mat_exp= np.kron(mat_exp, np.array([[1, 0], [0, -1]]))
                    elif op == 'X':
                        mat_exp= np.kron(mat_exp, np.array([[0, 1], [1, 0]]))
                    elif op == 'Y':
                        mat_exp= np.kron(mat_exp, np.array([[0, -1j], [1j, 0]]))
                mat_exp= expm(-1j * mat_exp * theta)

                msg= msg= 'QuantumCircuit does not implement the expected e^(i * pauli) operator. Params: pauli= %s, θ= %.4f' %(pauli, theta)
                self.compareOperators(mat_qc, mat_exp, delta=1e-14, msg= msg)


########################################################################################################
if __name__ == '__main__':
    unittest.main()