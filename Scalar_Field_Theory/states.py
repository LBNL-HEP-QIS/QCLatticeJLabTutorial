import math
import sys
sys.path.append('modules')
sys.path.append('modules/Deprecated')
#from qiskit.aqua.utils.circuit_factory import CircuitFactory
from qiskit import QuantumCircuit
import trotterized_evolution
import basic_circuits
from Deprecated import distributions
import basic_operator_implementations
from modules import hamiltonian


class FreeHarmonicOscillatorState():
    """
    Circuit factory that builds Quantum Circuit for generating eigenstates of the free harmonic oscillator.
    """
    def __init__(self, trotter_order, trotter_steps):
        """
        Initializes parameters.
        :param trotter_order: (int) Order of trotterization for state generation.
        :param trotter_steps: (int) Number of trotter steps for state generation.
        """
        self.order = trotter_order
        self.steps = trotter_steps

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
        qc += _create_free_vacuum(q)
        if n_state > 0:
            phi_x = basic_operator_implementations.PhiTensorXOperator()
            pi_y = basic_operator_implementations.PiTensorYOperator()
            excite_hamiltonian = hamiltonian.TimeEvolutionOfHamiltonian()
            excite_hamiltonian.clear_all()
            excite_hamiltonian.add_term(1 / math.sqrt(2), phi_x)
            excite_hamiltonian.add_term(-1 / math.sqrt(2), pi_y)
            excite_trotter = trotterized_evolution.TrotterizedEvolution(excite_hamiltonian, self.order, 0, self.steps, True)
            for i in range(n_state):
                time = math.pi / 2 / math.sqrt(i + 1)
                excite_trotter.set_time(time)
                excite_trotter.build(qc, q, q_ancillas)
                qc.x(q_ancillas[0])
            phase_free_excite(Nq, n_state)
            phase_offset = phase_free_excite(Nq, n_state)
            circ_phase = basic_circuits.phase(q, -phase_offset)
            qc += circ_phase

####################################################################
####     Private functions and classes      ########################
####################################################################

####################################################################
def _create_free_vacuum(q, phi_max):
   """
   Constructs quantum circuit for generating ground state of free harmonic oscillator Hamiltonian.
   :param q: (QuantumRegister) QuantumRegister to build circuit on
   :return: Result of calling gaussian_wavefunction in core.gaussian
   """
   qc = QuantumCircuit(q)
   Nq = q.size
   Ns = 2**(Nq)
   sigma = (Ns-1)/(2 * phi_max)
   mu = (Ns-1)/2
   low = 0
   high = Ns-1
   return _gaussian_wavefunction(q, mu, sigma, low, high)

####################################################################
def _gaussian_wavefunction(q, mean, std_dev, low, high):
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
   qc = QuantumCircuit(q)
   Nq = q.size
   normal = distributions.NormalDistributionWF(Nq, mean, std_dev, low, high)
   normal.build(qc, q)
   return qc

####################################################################
def phase_free_excite(Nq, n, phi_max):
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
   phase_Rz = (2**Nq-1) * (basic_operator_implementations._dphi(Nq, phi_max) - basic_operator_implementations._dpi(Nq, phi_max)) \
          * math.pi / 4 / math.sqrt(2) * pre
   phase_i = - math.pi / 2 * n
   return phase_Rz + phase_i