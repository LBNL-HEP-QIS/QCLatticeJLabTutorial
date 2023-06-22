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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, quantum_info, Aer


####################################################################################################
# Local Imports                                                                                    #
####################################################################################################
sys.path.append('Scalar_Field_Theory')
sys.path.append('arithmetic_pkg')
from Testing_Files.test_base import TestCaseUtils # custom class that includes utility functions for testing
from Scalar_Field_Theory.classical import *
import testing_utilities as tu
import arithmetic_pkg.alpha_arithmetic as aa
import arithmetic_pkg.series as series
import modules.lattice as lattice
#import settings as settings

global simulator_state
simulator_state= Aer.get_backend('statevector_simulator')


####################################################################################################
# Testing class                                                                                    #
####################################################################################################
class arithTestCase(TestCaseUtils):
    '''
        Class for testing the functions defined in alpha_arithmetic.py and series.py.
    '''

    #@unittest.skip('temp')
    def test_mu2_Multiplier(self):
        '''
        Tests method: aa.mu2_Multiplier().

        Things to test:
            Input size               -- 
        '''
        print('\nTesting mu2_Multiplier()...')

        for r in range(3, 5):
            mu2= QuantumRegister(r-2, 'mu2')
            x= QuantumRegister(r+3, 'x') # One extension bit
            prod= QuantumRegister(r+2, 'p')
            cl= ClassicalRegister(r+2)

            for n1 in range(-2**(r-3), 2**(r-3), 1):
                for n2 in range(-2**(r+1), 2**(r+1), 1):
                    qc= QuantumCircuit(mu2, x, prod, cl)

                    tu.Qinit(qc, mu2, n1)
                    tu.Qinit(qc, x, n2)
                    tu.Qinit(qc, prod, 0)

                    qc.compose(aa.mu2_Multiplier(r, circuit=True), qc.qubits[:], inplace=True)
                    qc.measure(prod, cl)
                    counts= execute(qc, simulator_state).result().get_counts()
                    
                    # In this case, mu^2 is not in two's complement, so need to correct n1.
                    if n1 < 0:
                        n1val= n1 + 2**(r-2)
                    else:
                        n1val= n1
                        
                    n1val*= 2**(-r)
                    n2val= n2 * 2**(-r)

                    prod_qc= tu.binary_to_decimal(list(counts.keys())[0], r)
                    prod_cl= n1val * n2val

                    msg= 'n1= %.4f, n2= %.4f -- Expected product= %.4f. Computed product= %.4f.' %(n1val, n2val, prod_cl % (2**r), prod_qc % (2**r))
                    self.checkEqual(len(counts), 1)
                    with self.subTest():
                        self.assertAlmostEqual(prod_qc, prod_cl, delta= (r-2) * 2**(-r), msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_mu2_Multiplier_b(self):
        '''
        Tests method: aa.mu2_Multiplier_b().

        Things to test:
            Input size               -- 
        '''
        print('\nTesting mu2_Multiplier_b()...')

        for r in range(3, 5):
            b= r + 2
            mu2= QuantumRegister(r-2, 'mu2')
            x= QuantumRegister(b+1, 'x') # One extension bit
            prod= QuantumRegister(b, 'p')
            cl= ClassicalRegister(b)

            for n1 in range(-2**(r-3), 2**(r-3), 1):
                for n2 in range(-2**(b-1), 2**(b-1), 1):
                    qc= QuantumCircuit(mu2, x, prod, cl)

                    tu.Qinit(qc, mu2, n1)
                    tu.Qinit(qc, x, n2)
                    tu.Qinit(qc, prod, 0)

                    qc.compose(aa.mu2_Multiplier_b(b, r, circuit=True), qc.qubits[:], inplace=True)
                    qc.measure(prod, cl)
                    counts= execute(qc, simulator_state).result().get_counts()

                    # In this case, mu^2 is not in two's complement, so need to correct n1.
                    if n1 < 0:
                        n1val= n1 + 2**(r-2)
                    else:
                        n1val= n1
                        
                    n1val*= 2**(-r)
                    n2val= n2 * 2**(-r)
                    prod_qc= tu.binary_to_decimal(list(counts.keys())[0], r)
                    prod_cl= n1val * n2val

                    msg= 'n1= %.4f, n2= %.4f -- Expected product= %.4f. Computed product= %.4f.' %(n1val, n2val, prod_cl % (2**r), prod_qc % (2**r))
                    self.checkEqual(len(counts), 1)
                    with self.subTest():
                        self.assertAlmostEqual(prod_qc, prod_cl, delta= (r-2) * 2**(-r), msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_FinalMuMultiply(self):
        '''
        Tests method: aa.FinalMuMultiply().

        Things to test:
            Input size               -- 
        '''
        print('\nTesting FinalMuMultiply()...')

        for r in range(1, 4):
            x= QuantumRegister(r+3, 'x') # One extension bit
            prod= QuantumRegister(r, 'p')
            cl= ClassicalRegister(r)

            for j in range(0, r):
                mu= QuantumRegister(j+2, 'mu2')

                for n1 in range(-2**(j+1), 2**(j+1), 1):
                    for n2 in range(-2**(r+1), 2**(r+1), 1):
                        qc= QuantumCircuit(mu, x, prod, cl)

                        tu.Qinit(qc, mu, n1)
                        tu.Qinit(qc, x, n2)
                        tu.Qinit(qc, prod, 0)

                        qc.compose(aa.FinalMuMultiply(j, r, circuit=False), qc.qubits[:], inplace=True)
                        qc.measure(prod, cl)
                        counts= execute(qc, simulator_state).result().get_counts()

                        n1val= -n1 * 2**(-j-1)
                        n2val= n2 * 2**(-r)
                        prod_qc= int(list(counts.keys())[0], 2) * 2**(-r)
                        #resultval2= binary_to_decimal(list(counts.keys())[0], r)
                        prod_cl= (n1val * n2val) % 1
                        #if not (np.sign(n1val * n2val) != np.sign(prod_qc) and  abs(abs(prod_qc-1) - prod_cl) < (j) * 2**(-r)):
                        #    if abs(prod_qc - prod_cl) > (j) * 2**(-r):
                        #        print('Error!!!')
                        #        print('(%f) * (%f) = %f, Actual= %f. Tolerance= %.4f' %(n1val, n2val, n1val * n2val, prod_qc, (j) * 2**(-r)))

                        self.checkEqual(len(counts), 1)
                        with self.subTest():
                            msg= 'n1= %.4f, n2= %.4f -- Expected product= %.4f. Computed product= %.4f.' %(n1val, n2val, prod_cl % (2**r), prod_qc % (2**r))
                            try:
                                self.assertAlmostEqual(prod_qc, prod_cl, delta= (j+1) * 2**(-r), msg= msg)
                            except AssertionError:
                                self.assertAlmostEqual(1-prod_qc, prod_cl, delta= (j+1) * 2**(-r), msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_FinalMuMultiply_b(self):
        '''
        Tests method: aa.FinalMuMultiply_b().

        Things to test:
            Input size               -- 
        '''
        print('\nTesting FinalMuMultiply_b()...')

        for r in range(1, 4):
            prod= QuantumRegister(r, 'p')
            cl= ClassicalRegister(r)

            for b in range(r, r+2):
                x= QuantumRegister(b+1, 'x') # One extension bit

                for j in range(0, r):
                    mu= QuantumRegister(j+2, 'mu2')

                    for n1 in range(-2**(j+1), 2**(j+1), 1):
                        for n2 in range(-2**(b-1), 2**(b-1), 1):
                            qc= QuantumCircuit(mu, x, prod, cl)

                            tu.Qinit(qc, mu, n1)
                            tu.Qinit(qc, x, n2)
                            tu.Qinit(qc, prod, 0)

                            qc.compose(aa.FinalMuMultiply_b(j, b, r, circuit=False), qc.qubits[:], inplace=True)
                            qc.measure(prod, cl)
                            counts= execute(qc, simulator_state).result().get_counts()

                            n1val= -n1 * 2**(-j-1)
                            n2val= n2 * 2**(-r)
                            prod_qc= int(list(counts.keys())[0], 2) * 2**(-r)
                            #resultval2= binary_to_decimal(list(counts.keys())[0], r)
                            prod_cl= (n1val * n2val) % 1
                            if not (np.sign(n1val * n2val) != np.sign(prod_qc) and  abs(abs(prod_qc-1) - prod_cl) < (j+2) * 2**(-r)):
                                if abs(prod_qc - prod_cl) > (j+2) * 2**(-r):
                                    print('Error!!!')
                                    print('(%f) * (%f) = %f, %f' %(n1val, n2val, prod_qc, n1val * n2val))

                            self.checkEqual(len(counts), 1)
                            with self.subTest():
                                msg= 'n1= %.4f, n2= %.4f -- Expected product= %.4f. Computed product= %.4f.' %(n1val, n2val, prod_cl % (2**r), prod_qc % (2**r))
                                try:
                                    self.assertAlmostEqual(prod_qc, prod_cl, delta= (j+1) * 2**(-r), msg= msg)
                                except AssertionError:
                                    self.assertAlmostEqual(1-prod_qc, prod_cl, delta= (j+1) * 2**(-r), msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_mu2_CCM(self):
        '''
        Tests method: aa.mu2_CCM().

        Things to test:
            Input size               -- 
        '''
        print('\nTesting mu2_CCM()...')

        for r in range(3, 7):
            anc= QuantumRegister(1, 'anc')
            mu2= QuantumRegister(r-2, 'mu2')
            prod= QuantumRegister(r+2, 'p')
            cl= ClassicalRegister(r+2)

            for n1 in range(-2**(r-3), 2**(r-3), 1):
                for n2 in range(-2**(r-3), 2**(r-3), 1):
                    qc= QuantumCircuit(anc, mu2, prod, cl)

                    #multStr= int_to_binary(n1).zfill(r-2)
                    multStr= tu.decimal_to_binary2(n1, r-2, 0)
                    tu.Qinit(qc, mu2, n2)
                    tu.Qinit(qc, prod, 0)

                    qc.compose(aa.mu2_CCM(r, multStr, circuit=False), qc.qubits[:], inplace=True)
                    qc.measure(prod, cl)
                    counts= execute(qc, simulator_state).result().get_counts()
                    
                    n1val= n1 * 2**(-r+3)

                    # Here, n2 represents mu^2, which is not two's complement. n2 just takes negative 
                    # values to initialize using tu.Qinit, so we fix it here.
                    if n2 < 0:
                        n2val= n2 + 2**(r-2)
                    else:
                        n2val= n2
                        
                    n2val*= 2**(-r)

                    prod_qc= tu.binary_to_decimal(list(counts.keys())[0], r)
                    prod_cl= n1val * n2val
                    
                    msg= 'n1= %.4f, n2= %.4f -- Expected product= %.4f. Computed product= %.4f.' %(n1val, n2val, prod_cl % (2**r), prod_qc % (2**r))
                    self.checkEqual(len(counts), 1)
                    with self.subTest():
                        self.assertAlmostEqual(prod_qc, prod_cl, delta=(r-1) * 2**(-r), msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_mu2_CCM_b(self):
        '''
        Tests method: aa.mu2_CCM_b().

        Things to test:
            Input size               -- 
        '''
        print('\nTesting mu2_CCM_b()...')

        anc= QuantumRegister(1, 'anc')
        for r in range(3, 7):
            mu2= QuantumRegister(r-2, 'mu2')

            for b in range(r+1, 8):
                prod= QuantumRegister(b, 'p')
                cl= ClassicalRegister(b)

                for n1 in range(-2**(b-1), 2**(b-1), 1):
                    for n2 in range(-2**(r-3), 2**(r-3), 1):
                        qc= QuantumCircuit(anc, mu2, prod, cl)

                        #multStr= int_to_binary(n1).zfill(r-2)
                        multStr= tu.decimal_to_binary2(n1, b, 0)
                        tu.Qinit(qc, mu2, n2)
                        tu.Qinit(qc, prod, 0)

                        #print(qc2= aa.mu2_CCM_b(b, r, multStr, circuit=False).num_qubits())
                        #print(qc
                        qc.compose(aa.mu2_CCM_b(b, r, multStr, circuit=False), qc.qubits[:], inplace=True)
                        qc.measure(prod, cl)
                        #print(qc.draw())
                        counts= execute(qc, simulator_state).result().get_counts()
                        
                        n1val= n1 * 2**(-r+3)

                        # Here, n2 represents mu^2, which is not two's complement. n2 just takes negative 
                        # values to initialize using tu.Qinit, so we fix it here.
                        if n2 < 0:
                            n2val= n2 + 2**(r-2)
                        else:
                            n2val= n2
                            
                        n2val*= 2**(-r)

                        prod_qc= tu.binary_to_decimal(list(counts.keys())[0], r)
                        prod_cl= n1val * n2val
                        
                        msg= 'n1= %.4f, n2= %.4f -- Expected product= %.4f. Computed product= %.4f.' %(n1val, n2val, prod_cl % (2**r), prod_qc % (2**r))
                        self.checkEqual(len(counts), 1)
                        with self.subTest():
                            self.assertAlmostEqual(prod_qc, prod_cl, delta=(b-3) * 2**(-r), msg= msg)

                        if abs(prod_cl - prod_qc) > (b-3) * 2**(-r):
                            print('Error!!!')
                            print('n1= %.4f, n2= %.4f. Computed product ! %.4f, but expected= %.4f\n. Also n2orig= %.4f' %(n1val, n2val, prod_qc, prod_cl, n2))

    ####################################################################################################
    # TODO
    @unittest.skip('temp')
    def test_muSquarer(self):
        '''
        Tests method: aa.muSquarer().

        Things to test:
            Input size               -- 
        '''
        print('\nTesting muSquarer()...')

        anc= QuantumRegister(1, 'anc') # Ancillary qubit, to copy multiplier bits to

        for j in range(1, 6):
            mu= QuantumRegister(j+3, 'mu') # mu register. Has 1 sign qubit, j fractional qubits, and 1 classical bit (least significant). Also 1 extension bit

            for r in range(3, 2*(j+1)):
                # Classical bit currently implemented as a qubit
                prod= QuantumRegister(r-2, 'prod') # Result is guaranteed < 1/4, so has r-2 bits
                cl= ClassicalRegister(r-2)
                #cl= ClassicalRegister(j+3)

                for mu_int in range(-2**(j) + 1, 2**(j)):
                    qc= QuantumCircuit(anc, mu, prod, cl)

                    tu.Qinit(qc, anc, 0)
                    tu.Qinit(qc, mu, mu_int)
                    tu.Qinit(qc, prod, 0)
                    
                    qc.compose(aa.muSquarer(j, r, circuit=True), qc.qubits[:], inplace=True)
                    qc.measure(prod, cl)
                    #qc.measure(mu, cl)
                    counts= execute(qc, simulator_state).result().get_counts()

                    mu_val= mu_int * 2**(-j-1)
                    prod_cl= mu_val ** 2
                    prod_qc= int(list(counts.keys())[0], 2) / (2**r)
                    
                    msg= 'mu= %.4f -- Expected product= %.4f. Computed product= %.4f.' %(mu_val, prod_cl % (2**r), prod_qc % (2**r))
                    self.checkEqual(len(counts), 1)
                    with self.subTest():
                        self.assertAlmostEqual(prod_qc, prod_cl, delta= (j+2) * 2**(-r), msg= msg)
                    # There are a couple instances where doing arithmetic like this just fails...

                    #if abs(prod_cl - prod_qc) > (j+2) * 2**(-r):
                    #    print('Error!!!')
                    #    print('(%f)^2 != %f, but %f\n.     j= %d, r= %d' %(mu_val, prod_qc, prod_cl, j, r))

    ####################################################################################################
    # TODO
    @unittest.skip('temp')
    def test_alpha_large_sigma(self):
        '''
        Tests method: series.alpha_large_sigma().

        Things to test:
            Input size               -- 
        '''
        print('\nTesting alpha_large_sigma()...')

    ####################################################################################################
    # TODO
    @unittest.skip('temp')
    def test_Parallel_Series_2coeffs(self):
        '''
        Tests method: series.Parallel_Series_2coeffs().

        Things to test:
            Input size               -- 
        '''
        print('\nTesting Parallel_Series_2coeffs()...')


########################################################################################################
if __name__ == '__main__':
    unittest.main()