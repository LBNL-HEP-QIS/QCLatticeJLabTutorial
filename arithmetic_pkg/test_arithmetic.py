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
from Testing_Files.test_base import TestCaseUtils # custom class that includes utility functions for testing
from Scalar_Field_Theory.classical import *
import arithmetic_pkg.testing_utilities as tu
import arithmetic_pkg.arithmetic as ar
import modules.lattice as lattice
#import harmonic_oscillator.settings as settings

global simulator_state
simulator_state= Aer.get_backend('statevector_simulator')


####################################################################################################
# Testing class                                                                                    #
####################################################################################################
class arithTestCase(TestCaseUtils):
    '''
        Class for testing the functions defined in arithmetic.py.
    '''


    #@unittest.skip('temp')
    def test_RippleAdder(self):
        '''
        Tests method: arithmetic.RippleAdder().

        Things to test:
            Input size               -- check
            Positive/negative inputs -- check
            Modular True/False       -- check
        '''
        print('\nTesting RippleAdder()...')

        # Test RippleAdder(modular= False) for two's complement numbers:
        # Works

        for k in range(2, 5):

            a= QuantumRegister(k, 'a')
            anc= QuantumRegister(1, 'anc')

            # modular == True
            b= QuantumRegister(k, 'b')
            cl= ClassicalRegister(k)

            for n1 in range(-2**(k-1), 2**(k-1), 1):
                for n2 in range(-2**(k-1), 2**(k-1), 1):
                    qc= QuantumCircuit(anc, a, b, cl)

                    tu.Qinit(qc, a, n1)
                    tu.Qinit(qc, anc, 0)
                    tu.Qinit(qc, b, n2)

                    qc.compose(ar.RippleAdder(k, modular=True, circuit=False), anc[:] + a[:] + b[:], inplace= True)
                    qc.measure(b, cl)
                    counts = execute(qc, simulator_state).result().get_counts()
                    
                    sum_qc= tu.binary_to_int(list(counts.keys())[0])
                    sum_cl= n1 + n2

                    msg= 'n1= %.3f, n2= %.3f -- Expected sum= %.3f. Computed sum= %.3f' %(n1, n2, sum_cl % 2**(k-1), sum_qc % 2**(k-1))
                    self.checkEqual(len(counts), 1)
                    self.checkEqual(sum_cl % 2**(k-1), sum_qc % 2**(k-1), msg= msg)
    
            # modular == False
            b= QuantumRegister(k+1, 'b')
            cl= ClassicalRegister(k+1)

            for n1 in range(-2**(k-1), 2**(k-1), 1):
                for n2 in range(-2**(k-1), 2**(k-1), 1):
                    qc= QuantumCircuit(anc, a, b, cl)

                    tu.Qinit(qc, a, n1)
                    tu.Qinit(qc, anc, 0)
                    tu.Qinit(qc, b[:k], n2)
                    tu.Qinit(qc, b[k], 0)

                    qc.compose(ar.RippleAdder(k, modular=False, circuit=False), anc[:] + a[:] + b[:], inplace= True)
                    qc.measure(b, cl)
                    counts = execute(qc, simulator_state).result().get_counts()
                    
                    sum_qc= tu.binary_to_int(list(counts.keys())[0])
                    sum_cl= n1 + n2

                    msg= 'n1= %.3f, n2= %.3f -- Expected sum= %.3f. Computed sum= %.3f' %(n1, n2, sum_cl, sum_qc)
                    self.checkEqual(len(counts), 1)
                    self.checkEqual(sum_cl, sum_qc, msg= msg)

    ####################################################################################################
    #@unittest.skip('temp')
    def test_Subtractor(self):
        '''
        Tests method: arithmetic.Subtractor().
        '''
        print('\nTesting Subtractor()...')

        for k in range(2, 5):

            a= QuantumRegister(k, 'a')
            anc= QuantumRegister(1, 'anc')

            # modular == True
            b= QuantumRegister(k, 'b')
            cl= ClassicalRegister(k)

            for n1 in range(-2**(k-1), 2**(k-1), 1):
                for n2 in range(-2**(k-1), 2**(k-1), 1):
                    qc= QuantumCircuit(anc, a, b, cl)

                    tu.Qinit(qc, a, n1)
                    tu.Qinit(qc, anc, 0)
                    tu.Qinit(qc, b, n2)

                    qc.compose(ar.Subtractor(k, modular=True, circuit=False), anc[:] + a[:] + b[:], inplace= True)
                    qc.measure(b, cl)
                    counts= execute(qc, simulator_state).result().get_counts()

                    diff_qc= tu.binary_to_int(list(counts.keys())[0])
                    diff_cl= n1 - n2

                    msg= 'n1= %.3f, n2= %.3f -- Expected difference= %.3f. Computed difference= %.3f' %(n1, n2, diff_cl % 2**(k-1), diff_qc % 2**(k-1))
                    self.checkEqual(len(counts), 1)
                    self.checkEqual(diff_qc % 2**(k-1), diff_cl % 2**(k-1), msg= msg)

            # modular == False
            b= QuantumRegister(k+1, 'b')
            cl= ClassicalRegister(k+1)

            for n1 in range(-2**(k-1), 2**(k-1), 1):
                for n2 in range(-2**(k-1), 2**(k-1), 1):
                    qc= QuantumCircuit(anc, a, b, cl)

                    tu.Qinit(qc, a, n1)
                    tu.Qinit(qc, anc, 0)
                    tu.Qinit(qc, b[:k], n2)
                    tu.Qinit(qc, b[k], 0)

                    qc.compose(ar.Subtractor(k, modular=False, circuit=False), anc[:] + a[:] + b[:], inplace= True)
                    qc.measure(b, cl)
                    counts= execute(qc, simulator_state).result().get_counts()
                    
                    diff_qc= tu.binary_to_int(list(counts.keys())[0])
                    diff_cl= n1 - n2

                    msg= 'n1= %.3f, n2= %.3f -- Expected difference= %.3f. Computed difference= %.3f' %(n1, n2, diff_cl, diff_qc)
                    self.checkEqual(len(counts), 1)
                    self.checkEqual(diff_qc, diff_cl, msg= msg)

    ####################################################################################################
    #@unittest.skip('temp')
    def test_Subtractor2(self):
        '''
        Tests method: arithmetic.Subtractor2().
        '''
        print('\nTesting Subtractor2()...')

        for k in range(2, 5):

            b= QuantumRegister(k, 'a')
            anc= QuantumRegister(1, 'anc')

            # modular == True
            a= QuantumRegister(k, 'b')
            cl= ClassicalRegister(k)

            for n1 in range(-2**(k-1), 2**(k-1), 1):
                for n2 in range(-2**(k-1), 2**(k-1), 1):
                    qc= QuantumCircuit(anc, a, b, cl)

                    tu.Qinit(qc, a, n1)
                    tu.Qinit(qc, anc, 0)
                    tu.Qinit(qc, b, n2)

                    qc.compose(ar.Subtractor2(k, modular=True, circuit=False), anc[:] + a[:] + b[:], inplace= True)
                    qc.measure(a, cl)
                    counts= execute(qc, simulator_state).result().get_counts()

                    diff_qc= tu.binary_to_int(list(counts.keys())[0])
                    diff_cl= n1 - n2

                    msg= 'n1= %.3f, n2= %.3f -- Expected difference= %.3f. Computed difference= %.3f' %(n1, n2, diff_cl % 2**(k-1), diff_qc % 2**(k-1))
                    self.checkEqual(len(counts), 1)
                    self.checkEqual(diff_qc % 2**(k-1), diff_cl % 2**(k-1), msg= msg)

            # modular == False
            a= QuantumRegister(k+1, 'b')
            cl= ClassicalRegister(k+1)

            for n1 in range(-2**(k-1), 2**(k-1), 1):
                for n2 in range(-2**(k-1), 2**(k-1), 1):
                    qc= QuantumCircuit(anc, a, b, cl)

                    tu.Qinit(qc, a[:k], n1)
                    tu.Qinit(qc, a[k], 0)
                    tu.Qinit(qc, anc, 0)
                    tu.Qinit(qc, b, n2)

                    qc.compose(ar.Subtractor2(k, modular=False, circuit=False), anc[:] + a[:] + b[:], inplace= True)
                    qc.measure(a, cl)
                    counts= execute(qc, simulator_state).result().get_counts()

                    diff_qc= tu.binary_to_int(list(counts.keys())[0])
                    diff_cl= n1 - n2

                    msg= 'n1= %.3f, n2= %.3f -- Expected difference= %.3f. Computed difference= %.3f' %(n1, n2, diff_cl, diff_qc)
                    self.checkEqual(len(counts), 1)
                    self.checkEqual(diff_qc, diff_cl, msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_CAdder(self):
        '''
        Tests method: arithmetic.CAdder().

        Things to test:
            Input size               -- check
            Positive/negative inputs -- check
            Modular True/False       -- check
            Control True/False       -- check
        '''
        print('\nTesting CAdder()...')

        for k in range(2, 5):
            a= QuantumRegister(k, 'a')
            b= QuantumRegister(k, 'b')
            c= QuantumRegister(1, 'ctrl')

            for modular in [True, False]:
                for n1 in range(-2**(k-1), 2**(k-1), 1):
                    for n2 in range(-2**(k-1), 2**(k-1), 1):
                        for c0 in [0, -1]:

                            if modular:
                                cl= ClassicalRegister(k)
                                qc= QuantumCircuit(c, a, b, cl)
                            else:
                                z= QuantumRegister(2, 'z')
                                cl= ClassicalRegister(k+1)
                                qc= QuantumCircuit(c, a, b, z, cl)
                                tu.Qinit(qc, z, 0)

                            tu.Qinit(qc, a, n1)
                            tu.Qinit(qc, b, n2)
                            tu.Qinit(qc, c, c0)

                            if modular:
                                qc.compose(ar.CAdder(k, modular, circuit=False), c[:] + a[:] + b[:], inplace= True)
                            else:
                                qc.compose(ar.CAdder(k, modular, circuit=False), c[:] + a[:] + b[:] + z[:], inplace= True)
                    
                            qc.measure(b, cl[:k])
                            if not modular:
                                qc.measure(z[0], cl[k])
                            counts= execute(qc, simulator_state).result().get_counts()

                            sum_qc= tu.binary_to_int(list(counts.keys())[0])
                            sum_cl= n2 - (c0 * n1)

                            self.checkEqual(len(counts), 1)
                            #if c0 == 0: 
                            #    assert sum_qc == n2
                            #if c0 == -1:  # controlled addition
                            #    if modular:
                            #        assert sum_qc % 2**(k-1) == (n1 + n2) % 2**(k-1)
                            #    else:
                            #        assert sum_qc == n1 + n2

                            # Equivalent of lines 256-262
                            if modular == True:
                                msg= 'n1= %.3f, n2= %.3f -- Expected sum= %.3f. Computed sum= %.3f' %(n1, n2, sum_cl % 2**(k-1), sum_qc % 2**(k-1))
                                self.checkEqual(sum_qc % 2**(k-1), sum_cl % 2**(k-1), msg= msg)

                            if modular == False:
                                msg= 'n1= %.3f, n2= %.3f -- Expected sum= %.3f. Computed sum= %.3f' %(n1, n2, sum_cl, sum_qc)
                                self.checkEqual(sum_qc, sum_cl, msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_CSubtractor(self):
        '''
        Tests method: arithmetic.CSubtractor().
        '''
        print('\nTesting CSubtractor()...')

        for k in range(2, 5):
            a= QuantumRegister(k, 'a')
            b= QuantumRegister(k, 'b')
            c= QuantumRegister(1, 'ctrl')

            for modular in [True, False]:
                for n1 in range(-2**(k-1), 2**(k-1), 1):
                    for n2 in range(-2**(k-1), 2**(k-1), 1):
                        for c0 in [0, -1]:

                            if modular:
                                cl= ClassicalRegister(k)
                                qc= QuantumCircuit(c, a, b, cl)
                            else:
                                z= QuantumRegister(2, 'z')
                                cl= ClassicalRegister(k+1)
                                qc= QuantumCircuit(c, a, b, z, cl)
                                tu.Qinit(qc, z, 0)

                            tu.Qinit(qc, a, n1)
                            tu.Qinit(qc, b, n2)
                            tu.Qinit(qc, c, c0)

                            if modular:
                                qc.compose(ar.CSubtractor(k, modular, circuit=False), c[:] + a[:] + b[:], inplace= True)
                            else:
                                qc.compose(ar.CSubtractor(k, modular, circuit=False), c[:] + a[:] + b[:] + z[:], inplace= True)
                    
                            qc.measure(b, cl[:k])
                            if not modular:
                                qc.measure(z[0], cl[k])
                            counts= execute(qc, simulator_state).result().get_counts()

                            diff_qc= tu.binary_to_int(list(counts.keys())[0])
                            diff_cl= n2 + (c0 * n1)

                            self.checkEqual(len(counts), 1)
                            #if c0 == 0:
                            #    self.checkEqual(diff_qc, n2)
                            ##if c0 == -1:  # controlled subtraction
                            #    if modular:
                            #        self.checkEqual(diff_qc % 2**(k-1), (n2 - n1) % 2**(k-1))
                            #    else:
                            #        self.checkEqual(diff_qc, n2 - n1)

                            # Equivalent of lines 308-314
                            if modular == True:
                                msg= 'n1= %.3f, n2= %.3f -- Expected difference= %.3f. Computed difference= %.3f' %(n1, n2, diff_cl % 2**(k-1), diff_qc % 2**(k-1))
                                self.checkEqual(diff_qc % 2**(k-1), diff_cl % 2**(k-1), msg= msg)

                            if modular == False:
                                msg= 'n1= %.3f, n2= %.3f -- Expected difference= %.3f. Computed difference= %.3f' %(n1, n2, diff_cl, diff_qc)
                                self.checkEqual(diff_qc, diff_cl, msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_Multiplier(self):
        '''
        Tests method: arithmetic.Multiplier().
        '''
        print('\nTesting Multiplier()...')

        for k in range(2, 5):
            a= QuantumRegister(k, 'a')
            b= QuantumRegister(k+1, 'b')
            p= QuantumRegister(2*k, 'p')
            cl= ClassicalRegister(2*k, 'cl')

            for n1 in range(-2**(k-1), 2**(k-1), 1):
                for n2 in range(-2**(k-1), 2**(k-1), 1):
                    qc= QuantumCircuit(a, b, p, cl)
                    
                    tu.Qinit(qc, a, n1)
                    tu.Qinit(qc, b, n2)
                    tu.Qinit(qc, p, 0)
                    
                    qc.compose(ar.Multiplier(k), a[:] + b[:] + p[:], inplace= True)
                    qc.measure(p, cl)
                    counts= execute(qc, simulator_state).result().get_counts()

                    prod_qc= tu.binary_to_int(list(counts.keys())[0])
                    prod_cl= n1 * n2

                    msg='n1= %.3f, n2= %.3f -- Expected product= %.3f. Computed productr= %.3f.' %(n1, n2, prod_cl % (2**k), prod_qc % (2**k))
                    self.checkEqual(len(counts), 1)
                    self.checkEqual(prod_qc, prod_cl, msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_AddSub(self):
        '''
        Tests method: arithmetic.AddSub().
        '''
        print('\nTesting AddSub()...')

        for k in range(2, 5):
            b= QuantumRegister(k, 'b')
            c= QuantumRegister(1, 'ctrl')

            for modular in [True, False]:
                for n1 in range(-2**(k-1), 2**(k-1), 1):
                    for n2 in range(-2**(k-1), 2**(k-1), 1):
                        for c0 in [0, -1]:

                            if modular == True:
                                a= QuantumRegister(k, 'a')
                                cl= ClassicalRegister(k)
                            if modular == False:
                                a= QuantumRegister(k+1, 'a')
                                cl= ClassicalRegister(k+1)

                            qc= QuantumCircuit(c, a, b, cl)
                            tu.Qinit(qc, a, n1)
                            tu.Qinit(qc, b, n2)
                            tu.Qinit(qc, c, c0)

                            qc.compose(ar.AddSub(k, modular, circuit=False), c[:] + a[:] + b[:], inplace= True)
                            qc.measure(a, cl)
                            counts= execute(qc, simulator_state).result().get_counts()

                            sum_qc= tu.binary_to_int(list(counts.keys())[0])
                            sum_cl= n1 + n2 + (2 * c0 * n2)

                            self.checkEqual(len(counts), 1)
                            #print('n1= %d, n2= %d, ctrl= %d, modular= ' %(n1, n2, c0) + str(modular))
                            #print(sum_qc, n1 + n2, n1 - n2)
                            
                            if modular == True:
                                msg= 'n1= %.3f, n2= %.3f -- Expected sum= %.3f. Computed sum= %.3f' %(n1, n2, sum_cl % 2**(k-1), sum_qc % 2**(k-1))
                                self.checkEqual(sum_qc % 2**(k-1), sum_cl % 2**(k-1), msg= msg)

                            if modular == False:
                                msg= 'n1= %.3f, n2= %.3f -- Expected sum= %.3f. Computed sum= %.3f' %(n1, n2, sum_cl, sum_qc)
                                self.checkEqual(sum_qc, sum_cl, msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_Divider(self):
        '''
        Tests method: arithmetic.Divider() and arithmetic.Divider2().
        '''
        print('\nTesting Divider()...')

        for k in range(2, 5):
            a= QuantumRegister(k, 'a')
            b= QuantumRegister(k, 'b')
            q= QuantumRegister(k-1, 'q')
            anc= QuantumRegister(1, 'anc')
            cl_int= ClassicalRegister(k)
            cl_rem= ClassicalRegister(k-1)

            for n1 in range(1, 2**(k-1), 1):
                for n2 in range(1, n1, 1):
                    for divider in [ar.Divider, ar.Divider2]: # check both register arrangements
                        qc= QuantumCircuit(a, q, b, anc, cl_int, cl_rem)
                        #qc.initialize([1,0], anc)
                        tu.Qinit(qc, anc, 0)
                        tu.Qinit(qc, a, n1)
                        tu.Qinit(qc, b, n2)
                        tu.Qinit(qc, q, 0)

                        #qc.compose(divider(k, circuit=True), a[:] + q[:] + b[:] + anc[:], inplace=True)
                        qc.compose(divider(k, circuit=True), qc.qubits[:], inplace=True)

                        qc.measure([a[k-1]] + q[:], cl_int)
                        qc.measure(a[:k-1], cl_rem)

                        counts= execute(qc, simulator_state).result().get_counts()
                        counts_split= list(counts.keys())[0].split()
                        quo_qc= tu.binary_to_int(counts_split[1])
                        rem_qc= tu.binary_to_int('0' + counts_split[0])

                        msg_quo= 'n1= %d, n2= %d -- Expected quotient / remainder= %d r %d. Computed quotient / remainder= %d r %d.' %(n1, n2, int(n1 / n2), n1 % n2, quo_qc, rem_qc)
                        msg_rem= 'n1= %d, n2= %d -- Expected quotient / remainder= %d r %d. Computed quotient / remainder= %d r %d.' %(n1, n2, int(n1 / n2), n1 % n2, quo_qc, rem_qc)
                        self.checkEqual(len(counts), 1)
                        self.checkEqual(quo_qc, int(n1 / n2), msg= msg_quo)
                        self.checkEqual(rem_qc, n1 % n2, msg= msg_rem)

    ####################################################################################################
    @unittest.skip('temp')
    def test_ModularMultiplier(self):
        '''
        Tests method: arithmetic.ModularMultiplier().
        '''
        print('\nTesting ModularMultiplier()...')

        for k in range(1, 4):
            for m in range(1, 3):
                p= k+m
                a= QuantumRegister(p, 'a')
                b= QuantumRegister(p+1, 'b')
                prod= QuantumRegister(p+m, 'p')
                cl= ClassicalRegister(p+m, 'cl')

                for n1 in range(-2**(p-1), 2**(p-1), 1):
                    for n2 in range(-2**(p-1), 2**(p-1), 1):
                        qc= QuantumCircuit(a, b, prod, cl) 
                        
                        tu.Qinit(qc, a, n1)
                        tu.Qinit(qc, b, n2)
                        tu.Qinit(qc, prod, 0)
                        
                        qc.compose(ar.ModularMultiplier(k, m), a[:] + b[:] + prod[:], inplace= True)
                        qc.measure(prod, cl)
                        counts= execute(qc, simulator_state).result().get_counts()

                        prod_qc= tu.binary_to_decimal(list(counts.keys())[0], 2*m)

                        n1val= n1/(2**m)
                        n2val= n2/(2**m)
                        prod_cl= n1val * n2val
                        #print('(%.1f) * (%.1f) = %.2f' %(n1val, n2val, prod_qc))

                        msg= 'n1= %.3f, n2= %.3f -- Expected product= %.3f. Computed product= %.3f.' %(n1val, n2val, prod_cl % (2**k), prod_qc % (2**k))
                        self.checkEqual(len(counts), 1)
                        self.checkEqual(prod_qc % (2**k), prod_cl % (2**k), msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_ModularMultiplier2(self):
        '''
        Tests method: arithmetic.ModularMultiplier2().
        '''
        print('\nTesting ModularMultiplier2()...')

        for k in range(1, 4):
            for m in range(1, 3):
                p= k+m
                a= QuantumRegister(p, 'a')
                b= QuantumRegister(k+1, 'b')
                prod= QuantumRegister(p, 'p')
                cl= ClassicalRegister(p, 'cl')

                for n1 in range(-2**(p-1), 2**(p-1), 1):
                    for n2 in range(-2**(k-1), 2**(k-1), 1):
                        qc= QuantumCircuit(a, b, prod, cl) 
                        
                        tu.Qinit(qc, a, n1)
                        tu.Qinit(qc, b, n2)
                        tu.Qinit(qc, prod, 0)
                        
                        qc.compose(ar.ModularMultiplier2(k, m), a[:] + b[:] + prod[:], inplace= True)
                        qc.measure(prod, cl)
                        counts= execute(qc, simulator_state).result().get_counts()

                        prod_qc= tu.binary_to_decimal(list(counts.keys())[0], m)
                        n1val= n1/(2**m)
                        prod_cl= n1val * n2
                        #print('(%.1f) * (%.1f) = %.2f' %(n1val, n2val, prod_qc))

                        msg= 'n1= %.3f, n2= %.3f -- Expected product= %.3f. Computed product= %.3f.' %(n1val, n2, prod_cl % (2**k), prod_qc % (2**k))
                        self.checkEqual(len(counts), 1)
                        self.checkEqual(prod_qc % (2**k), prod_cl % (2**k), msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_ModularMultiplier3(self):
        '''
        Tests method: arithmetic.ModularMultiplier3().
        '''
        print('\nTesting ModularMultiplier3()...')

        for k in range(1, 4):
            for m in range(1, 3):
                p= k+m
                a= QuantumRegister(p, 'a')
                b= QuantumRegister(p+1, 'b')
                prod= QuantumRegister(p, 'p')
                cl= ClassicalRegister(p, 'cl')

                for n1 in range(-2**(p-1), 2**(p-1), 1):
                    for n2 in range(-2**(p-1), 2**(p-1), 1):
                        qc= QuantumCircuit(a, b, prod, cl) 
                        
                        tu.Qinit(qc, a, n1)
                        tu.Qinit(qc, b, n2)
                        tu.Qinit(qc, prod, 0)
                        
                        qc.compose(ar.ModularMultiplier3(p, m), a[:] + b[:] + prod[:], inplace= True)
                        qc.measure(prod, cl)
                        counts= execute(qc, simulator_state).result().get_counts()

                        prod_qc= tu.binary_to_decimal(list(counts.keys())[0], m)

                        n1val= n1/(2**m)
                        n2val= n2/(2**m)
                        prod_cl= n1val * n2val
                        #print('(%.3f) * (%.3f) = %.2f' %(n1val, n2val, prod_qc))

                        self.checkEqual(len(counts), 1)
                        with self.subTest():
                            if np.sign(prod_cl) != np.sign(prod_qc) and abs(prod_qc - prod_cl) < m * 2**(-m):
                                msg='n1= %.3f, n2= %.3f -- Expected product= %.3f. Computed product= %.3f.' %(n1val, n2val, prod_cl % (2**k), prod_qc % (2**k))
                                self.assertAlmostEqual(prod_qc, prod_cl, delta= m * 2**(-m), msg= msg)

                            else:
                                msg= 'n1= %.3f, n2= %.3f -- Expected product= %.3f. Computed product= %.3f.' %(n1val, n2val, prod_cl % (2**k), prod_qc % (2**k))
                                self.assertAlmostEqual(prod_qc % (2**k), prod_cl % (2**k), delta= m * 2**(-m), msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_ModularSquarer(self):
        '''
        Tests method: arithmetic.ModularSquarer().
        '''
        print('\nTesting ModularSquarer()...')

        for k in range(1, 4):
            for r in range(1, 3):
                p= k + r
                a= QuantumRegister(p+1, 'a')
                anc= QuantumRegister(1, 'anc')
                prod= QuantumRegister(p, 'p')
                cl= ClassicalRegister(p, 'cl')

                for n1 in range(-2**(p-1), 2**(p-1), 1):
                    qc= QuantumCircuit(a, anc, prod, cl) 
                    
                    tu.Qinit(qc, a, n1)
                    tu.Qinit(qc, anc, 0)
                    tu.Qinit(qc, prod, 0)
                    
                    qc.compose(ar.ModularSquarer(p, r), a[:] + anc[:] + prod[:], inplace= True)
                    qc.measure(prod, cl)
                    counts= execute(qc, simulator_state).result().get_counts()

                    prod_qc= tu.binary_to_decimal(list(counts.keys())[0], r)
                    n1val= n1/(2**r)
                    #print('(%.3f)^2 = %.2f' %(n1val, prod_qc))

                    self.checkEqual(len(counts), 1)
                    with self.subTest():
                        if np.sign(n1val ** 2) != np.sign(prod_qc) and abs(prod_qc - (n1val ** 2)) < r * 2**(-r):
                            msg= 'n1= %.3f -- Expected product= %.3f. Computed productr= %.3f.' %(n1val, n1val ** 2, prod_qc)
                            self.assertAlmostEqual(prod_qc, n1val ** 2, delta= r * 2**(-r), msg= msg)

                        else:
                            msg= 'n1= %.3f -- Expected product= %.3f. Computed productr= %.3f.' %(n1val, n1val ** 2, prod_qc)
                            self.assertAlmostEqual(prod_qc % (2**k), (n1val ** 2) % (2**k), delta= r * 2**(-r), msg= msg)

    ####################################################################################################
    @unittest.skip('temp')
    def test_CCM_general(self):
        '''
        Tests method: arithmetic.CCM_general().
        '''
        print('\nTesting CCM_general()...')

        for k in range(1, 4):
            for r in range(1, 3):
                p= k + r
                b= QuantumRegister(k, 'b')
                anc= QuantumRegister(1, 'anc')
                ext= QuantumRegister(r, 'ext')
                prod= QuantumRegister(p, 'p')
                cl= ClassicalRegister(p, 'cl')

                for n1 in range(-2**(p-1), 2**(p-1), 1):
                    for n2 in range(-2**(k-1), 2**(k-1), 1):
                        a= tu.decimal_to_binary2(n1/(2**r), p, r)
                        
                        qc= QuantumCircuit(b, ext, anc, prod, cl)
                        
                        tu.Qinit(qc, b, n2)
                        tu.Qinit(qc, prod, 0)
                        
                        qc.compose(ar.CCM_general(a, p, r, int_input=True), qc.qubits[:], inplace= True)
                        qc.measure(prod, cl)
                        
                        counts= execute(qc, simulator_state).result().get_counts()
                        self.checkEqual(len(counts), 1)

                        prod_qc= tu.binary_to_decimal(list(counts.keys())[0], r)
                        n1val= n1/(2**r)

                        #print('(%.4f) * (%.4f) = %.4f' %(n1val, n2, prod_qc))
                        
                        prod_cl= n1val * n2

                        with self.subTest():
                            msg= 'n1= %.3f -- Expected product= %.3f. Computed productr= %.3f.' %(n1val, n1val ** 2, prod_qc)
                            self.assertAlmostEqual(prod_qc % (2**k), (n1val * n2) % (2**k), delta= r * 2**(-r), msg= msg)


########################################################################################################
if __name__ == '__main__':
    unittest.main()