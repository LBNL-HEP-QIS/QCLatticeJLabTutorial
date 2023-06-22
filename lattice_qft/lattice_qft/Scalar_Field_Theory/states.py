import math
# import sys
# sys.path.append('modules')
# sys.path.append('modules/Deprecated')
#from qiskit.aqua.utils.circuit_factory import CircuitFactory
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from lattice_qft.core import trotterized_evolution
from lattice_qft.core import basic_circuits
from lattice_qft.core import  distributions
from lattice_qft.Scalar_Field_Theory import basic_operator_implementations
from lattice_qft.core import hamiltonian
from qiskit import Aer, transpile, assemble, execute
from qiskit.tools.visualization import plot_histogram, plot_state_city, plot_distribution
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
import statistics as stat

class FreeHarmonicOscillatorState():
    """
    Circuit factory that builds Quantum Circuit for generating eigenstates of the free harmonic oscillator.
    """
    def __init__(self, trotter_order, trotter_steps, phi_max):
        """
        Initializes parameters.
        :param trotter_order: (int) Order of trotterization for state generation.
        :param trotter_steps: (int) Number of trotter steps for state generation.
        """
        self.order = trotter_order
        self.steps = trotter_steps
        self.phi_max = phi_max

    def get_trotter_order(self):
        return self.order

    def set_trotter_order(self, order):
        self.order = order

    def get_trotter_steps(self):
        return self.steps

    def set_trotter_order(self, steps):
        self.steps = steps

    def build(self, qc, q, q_ancillas, params=None):
        """
        Builds Quantum Circuit for generating n-th excited state of harmonic oscillator.
        :param qc: (QuantumCircuit) Quantum Circuit to build on.
        :param q: (QuantumRegister) Quantum Register in qc to build on.
        :param q_ancillas: (QuantumRegister) Contains a single ancilla qubit.
        :param params: (int) number >= 0, eigenstate number.
        :return: None, operates on quantum circuit in-place.
        """
        n_state = params
        Nq = q.size
        gs = VacuumStatePrep(q, self.phi_max)
        qc.compose(gs._create_free_vacuum(), inplace=True)
        #print(qc.draw())
        print("n_state = ", n_state)
        if n_state > 0:
            phi_x = basic_operator_implementations.PhiTensorXOperator(self.phi_max)
            pi_y = basic_operator_implementations.PiTensorYOperator(self.phi_max)
            excite_hamiltonian = hamiltonian.TimeEvolutionOfHamiltonian()
            excite_hamiltonian.clear_all()
            excite_hamiltonian.add_term(1 / math.sqrt(2), phi_x)
            excite_hamiltonian.add_term(-1 / math.sqrt(2), pi_y)
            print('terms: ',excite_hamiltonian.get_terms())
            excite_trotter = trotterized_evolution.TrotterizedEvolution(excite_hamiltonian, self.order, 0, self.steps, True)
            for i in range(n_state):
                time = math.pi / 2 / math.sqrt(i + 1)
                excite_trotter.set_time(time)
                print('time: ',excite_trotter.time)
                excite_trotter.build(qc, q, q_ancillas)
                qc.x(q_ancillas)
            phase_offset = gs.phase_free_excite(Nq, n_state)
            circ_phase = basic_circuits.phase(q, -phase_offset)
            qc.compose(circ_phase, inplace=True)
        return qc

####################################################################
####     Private functions and classes      ########################
####################################################################

####################################################################
class VacuumStatePrep():
    def __init__(self, q, phi_max):
        self.q = q
        self.Nq = q.size
        self.Ns = 2**(self.Nq)
        self.sigma = (self.Ns-1)/(2 * phi_max)
        self.mu = (self.Ns-1)/2
        self.low = 0
        self.high = self.Ns-1
        self.phi_max = phi_max



####################################################################
    def _create_free_vacuum(self):
        """
        Constructs quantum circuit to generate state whose wavefunction is gaussian with purely real components.
        Contructs circuit in-place.
        :param q: (QuantumRegister) Quantum register used to store wavefunction
        :param mean: (flaot) The man value of the Gaussian
        :param std_dev: (float) The standard deviation of the Gaussian
        :param low: (int) lower bound of wavefunction (corresponding to state |0>, normally 0)
        :param high: (int) upper bound (corresponding to state |1>, normally 2**(number of qubits))
        :return: (QuantumCircuit) The quantum circuit
        """
        qc = QuantumCircuit(self.q)
        Nq = self.q.size
        normal = distributions.NormalDistributionWF(Nq, self.mu, self.sigma, self.low, self.high)
        qc1 = normal.build(qc, self.q)
        return qc1

####################################################################
    def phase_free_excite(self, Nq, n):
        """
        Returns the phase
        :param Nq: (int) The number of qubits in the wavefunction digitization
        :param n: (int) The number of the excited state
        :return: (float) The overall phase
        The term phase_Rz is coming from the Rz gates, while the second term comes from the factor of -i (see JLP)
        """
        pre = 0
        for i in range(n):
            pre += 1 / math.sqrt(i+1)
        phase_Rz = (2**(self.Nq-1)) * (basic_operator_implementations._dphi(Nq, self.phi_max) - basic_operator_implementations._dpi(Nq, self.phi_max)) \
                * math.pi / 4 / math.sqrt(2) * pre
        phase_i = - math.pi / 2 * n
        return phase_Rz + phase_i

# import numpy as np

# circ_size = 4
# #print(np.sum(np.power([1/np.sqrt(2),1/np.sqrt(2)],2)))
# vac = vacuum_state_prep(QuantumRegister(circ_size), 2)._create_free_vacuum()
# vac.measure_all()

# simulator = Aer.get_backend('qasm_simulator')
# result = execute(vac, simulator, shots=100000).result()
# counts = result.get_counts()
# counts_test = []
# for k in range(0,2**circ_size):
#     state = str(bin(k).replace("0b", "")).zfill(circ_size)
#     if state in counts.keys():
#         counts[state] = (counts[state]/100000)
#     else:
#         counts[state] = 0
#     #counts_test.append(counts[state])
#     counts[k] = counts.pop(state)

# print(counts)

# mu, sigma = stat.mean(counts), stat.stdev(counts)

# print(mu, sigma)

# y = norm.pdf(np.linspace(0,2**circ_size,1000), mu, sigma)

# plt.plot(np.linspace(0,2**circ_size,1000), y, 'r--', linewidth=2)
# #_, bins, _ = plt.hist(counts_test, 10, density=1, alpha=0.5)
# print(counts)
# plt.bar(counts.keys(), counts.values(), color='g')
# plt.show()


# excitedstate = FreeHarmonicOscillatorState(1,10,phi_max=2)
# q = QuantumRegister(4)

# q_ancilla = QuantumRegister(1)
# cr = ClassicalRegister(4)
# qc = QuantumCircuit(q,q_ancilla,cr)

# qc_test = excitedstate.build(qc,q, q_ancilla, params=2)
# qc_test.barrier(q)
# qc_test.measure(q,cr)
# simulator = Aer.get_backend('aer_simulator')
# result = execute(qc_test, simulator, shots=100000).result()
# counts = result.get_counts()
# plot_histogram(counts)
# print(qc_test.draw())
# plt.show()