####################################################################################################
# General Imports                                                                                  #
####################################################################################################
import unittest
import sys
import math
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statistics as stat
from scipy.stats import norm
####################################################################################################
# Local Imports                                                                                    #
####################################################################################################
sys.path.append('modules/deprecated')
sys.path.append('Scalar_Field_Theory')

from distributions import *
import states

simulator = Aer.get_backend('statevector_simulator')


####################################################################################################
# Testing class                                                                                    #
####################################################################################################
class TestDistributions1(unittest.TestCase):
    """
    Test build initialized quantum circuit method: distributions>UnivariateDistribution(class)>build(method).

    Prepare single qubit circuit with pauli x and compare with quantum circuit initialized with statevector = [0,1]
    """
    probabilities = [0,1]
    num_target_qubits = [0]
    def test_build(self):
        #Build Single Qubit Quantum Circuit
        qc = QuantumCircuit(1)
        UnivariateDistribution.build(self, qc)
        qc1 = QuantumCircuit(1)
        qc1.x(0)
        result_test_class = execute(qc, simulator).result().get_statevector()
        result_standard_compare = execute(qc1, simulator).result().get_statevector()
        #Test with assertEquals
        self.assertEqual(result_test_class, result_standard_compare)

class TestDistributions2(unittest.TestCase): 
    """
    Test build initialized quantum circuit method: distributions>UnivariateDistribution(class)>build(method).

    Prepare two qubit circuit with haddamard gate on each qubit and compare with quantum circuit initialized with statevector = [1/2, 1/2, 1/2, 1/2]
    """ 

    statevector = [0.5, 0.5, 0.5, 0.5]      
    probabilities = statevector/np.linalg.norm(statevector)
    num_target_qubits = [0,1]

    def test_build_edgecase(self):
        #Build Two Qubit Quantum Circuit
        qc = QuantumCircuit(2)
        UnivariateDistribution.build(self, qc)
        qc1 = QuantumCircuit(2)
        qc1.h([0,1])
        result_test_class = execute(qc, simulator).result().get_statevector()
        result_standard_compare = execute(qc1, simulator).result().get_statevector()
        #Test with assertEquals
        self.assertEqual(result_test_class, result_standard_compare)

class TestPDFtoProbabilties(unittest.TestCase):
    """
    Test pdf_to_probabilities method: distributions>UnivariateDistribution(class)>pdf_to_probabilities(method).

    Create standard gaussian distribution from scipy.stat.norm method. Normalize standard gaussian for truncated dist. [-1,1] and for 1000 x values.
    Create truncated discretized gaussian with normalization using pdf_to_probabilities
    """ 
    @staticmethod
    def pdf_normal(x):
        mean = 0
        stdev = 1
        return (1/(math.sqrt(2*math.pi*(stdev**2))))*math.exp(-(((x-mean)**2)/(2*(stdev**2))))

    def test_PDF(self):
        total = .0
        probabilty_values_compare_normalized = []
        probabilty_values = UnivariateDistribution.pdf_to_probabilities(self.pdf_normal, -1,1,1000)
        probabilty_values_compare = stats.norm(0,1).pdf(probabilty_values[1])
        for y in probabilty_values_compare:
            total += y
        for prob in probabilty_values_compare:
            prob /= total
            probabilty_values_compare_normalized.append(prob)
        for i in range(0,len(probabilty_values)):
            self.assertAlmostEqual(probabilty_values_compare_normalized[i],probabilty_values[0].tolist()[0],delta=0.001)
        plt.show()

class TestPDFtoProbabilties2(unittest.TestCase):
    """
    Test pdf_to_probabilities_2 method: distributions>UnivariateDistribution(class)>pdf_to_probabilities_2(method).

    Create standard gaussian distribution from scipy.stat.norm method and square values. Normalize standard square gaussian for truncated dist. [-1,1] and for 1000 x values.
    Create truncated discretized square gaussian with normalization using pdf_to_probabilities_2
    """ 
    @staticmethod
    def pdf_normal(x):
        """
        Normal Distribution Probabilty Distribution Function with variable x. X is input np.array.
        """
        mean = 0
        stdev = 1
        return (1/(math.sqrt(2*math.pi*(stdev**2))))*math.exp(-(((x-mean)**2)/(2*(stdev**2))))

    def test_PDF2(self):
        total = .0
        probabilty_values_compare_normalized = []
        probabilty_values = UnivariateDistribution.pdf_to_probabilities_2(self.pdf_normal, -1,1,1000)
        probabilty_values_compare = stats.norm(0,1).pdf(probabilty_values[1])
        for y in probabilty_values_compare:
            total += y**2
        for prob in probabilty_values_compare:
            prob = (prob**2)/total
            probabilty_values_compare_normalized.append(prob)
        

        for i in range(0,len(probabilty_values)):
            self.assertAlmostEqual(probabilty_values_compare_normalized[i],probabilty_values[0].tolist()[0],delta=0.001)
    
class TestCreateFreeVacuum(unittest.TestCase):
    """
    Test _create_free_vacuum method: states>VacuumStatePrep(class)>_create_free_vacuum(method).

    Initialize gaussian to (circ_size: positive int) qubit circuit
    Run simulation with aer_simulator and compare mean with what was calculated in VacuumStatePrep based on phi_max
    """     
    def stdev(self, data, mean, shots):
        temp = 0
        for key in data:
            temp += (shots*data[key]*((key-mean)**2))
        return math.sqrt(temp/(shots-1))

    def test_Vacuum_Mean_Stdev(self):
        circ_size = 4
        n_shots = 1e4
        vac = states.VacuumStatePrep(QuantumRegister(circ_size), 2)
        vac_qc = vac._create_free_vacuum()
        vac_qc.measure_all()
        simulator = Aer.get_backend('aer_simulator')
        result = execute(vac_qc, simulator, shots=n_shots).result()
        counts = result.get_counts()    
        for k in range(0,2**circ_size):
            state = str(bin(k).replace("0b", "")).zfill(circ_size)
            if state in counts.keys():
                counts[state] = (counts[state]/n_shots)
            else:
                counts[state] = 0
            #counts_test.append(counts[state])
            counts[k] = counts.pop(state)
        mu, sigma = stat.mean(counts), stat.stdev(counts)
        y = norm.pdf(np.linspace(0,2**circ_size,1000), mu, self.stdev(counts, mu, n_shots))
        plt.plot(np.linspace(0,2**circ_size,1000), y, 'r--', linewidth=2)
        plt.bar(counts.keys(), counts.values(), color='g')
        #print(self.stdev(counts, mu, n_shots))
        self.assertAlmostEqual(mu,vac.mu)
        #self.assertAlmostEqual(sigma,self.stdev(counts, mu, n_shots),delta=1)


if __name__ == '__main__':
    unittest.main()