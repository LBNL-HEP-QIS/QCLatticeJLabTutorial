### Utility functions for constructing and visualizing quantum circuits in Qiskit

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math
import operator

####################################################################
def measure_all(qc, q, c):
   """
   Constructs measurement circuit on quantum register.
   :param qc: (QuantumCircuit) Quantum Circuit to add measurement circuit to.
   :param q: (QuantumRegister) Quantum Register in qc to measure.
   :param c: (QuantumRegister) Classical Register in qc to readout q.
   :return: None.
   """
   qc.measure(q, c)
   return None

####################################################################
def measure_all(q, c):
   """
   Constructs measurement circuit on quantum register.
   :param qc: (QuantumCircuit) Quantum Circuit to add measurement circuit to.
   :param q: (QuantumRegister) Quantum Register in qc to measure.
   :param c: (QuantumRegister) Classical Register in qc to readout q.
   :return: None.
   """
   qc = QuantumCircuit(q, c)
   qc.measure(q, c)
   return qc

####################################################################
def swap_all(qc, q):
   """
   Swaps all qubits in a quantum register. Reverses qubit order.
   :param qc: (QuantumCircuit) Quantum Circuit to add swap circuit to.
   :param q: (QuantumRegister) Quantum Register in qc to swap.
   :return: None.
   """
   n = q.size
   for i in range(n//2):
     qc.swap(q[i], q[n-i-1])
   return None

####################################################################
def controlled_swap_all(qc, q, control, control_index):
   """
   Swaps all qubits in a quantum register. Reverses qubit order.
   :param qc: (QuantumCircuit) Quantum Circuit to add swap circuit to.
   :param q: (QuantumRegister) Quantum Register in qc to swap.
   :return: None.
   """
   n = q.size
   for i in range(n//2):
     qc.cswap(control[control_index], q[i], q[n-i-1])
   return None

####################################################################
def draw_circ(qc):
   """
   Draws quantum circuit.
   :param qc: (QuantumCircuit) Quantum circuit to draw.
   :return: None.
   """
   return qc.draw(output = 'mpl')

####################################################################
def get_energy_from_phase(phase, time):
   """
   Computes energy value from phase.
   :param phase: (str) binary value of phase register.
   :param time: (float) time length for which Hamiltonian is acting.
   :return: (float) energy value.
   """
   if time == 0:
      return 0
   else:
      return (-2*math.pi*parse_bin(phase))/time

####################################################################
def parse_bin(s):
   """
   Converts binary fraction notation to float.
   :param s: (str) value in binary fraction notation.
   :return: (float) float value of binary fraction.
   """
   total = 0
   for n in range(len(s)):
      total += 1/(2**(n+1))*int(s[n])
   return total

####################################################################
def get_energy_from_result(result, time):
   """
   computes most likely energy returned from simulation.
   :param result: (dict) result from running run_circuit.run_qasm.
   :param time: (float) time length for which hamiltonian is evolved.
   :return: (float) energy.
   """
   return get_energy_from_phase(max(result.items(), key=operator.itemgetter(1))[0],time)
