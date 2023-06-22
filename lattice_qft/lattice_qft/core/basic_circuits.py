import math
from qiskit import QuantumCircuit
import numpy as np




#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))

import sys
print(sys.path)



import lattice_qft.core.utils as utils

####################################################################
def fourier_transform(q, inv=False, swap=True):
    """
    Constructs regular quantum fourier transform circuit.
    :param q: (QuantumRegister) Quantum Register in qc to build circuit on
    :param inv: (bool) False if regular QFT, True if Inverse QFT
    :return: (QuantumCircuit) Circuit implementing QFT
    """
    qc = QuantumCircuit(q)
    Nq = q.size
    if inv:
        if swap:
            utils.swap_all(qc, q)
        for i in (range(Nq)):
            if i > 0:
                for j in (range(i)):
                    qc.cp(-math.pi/(2**(i-j)),q[i],q[j])
            qc.h(q[i])
    else:
        for i in reversed(range(Nq)):
            qc.h(q[i])
            if i > 0:
                for j in reversed(range(i)):
                    qc.cp(math.pi/(2**(i-j)),q[i],q[j])
        if swap:
            utils.swap_all(qc, q)
    return qc

####################################################################
def fourier_transform_symmetric(q, inv=False, swap=True):
    """
    Constructs symmetric quantum fourier transform circuit, where momentum values
    are symmetrically distributed about 0.
    :param q: (QuantumRegister) Quantum Register in qc to build circuit on
    :param inv: (bool) False if regular QFT, True if Inverse QFT
    :return: (QuantumCircuit) Circuit implementing symmetric QFT
    """
    qc = QuantumCircuit(q)
    Nq = q.size
    m= 2**Nq - 1

    #if inv:
    if False:
        pass
    #    qc += fourier_transform(q, inv=True, swap= swap)
    #    for i in (range(Nq)):
    #        qc.p((m * math.pi / (2 ** (i + 1))), q[Nq-i-1])
    else:
        for i in (range(Nq)):
            qc.p((-m * math.pi / (2 ** (i + 1))), q[Nq-i-1])
        
        theta= np.pi * (2**Nq - 1)**2 / 2**Nq / 2.
        qc.compose(phase(q, theta), inplace=True) # uses phase() !!

        qc.compose(fourier_transform(q, inv=False, swap= False), inplace=True)

        for i in (range(Nq)):
            qc.p((-m * math.pi / (2 ** (i + 1))), q[i]) # Note the indexing

        if swap:
            utils.swap_all(qc, q)

    if inv:
        return qc.inverse()
    else:
        return qc

####################################################################
def phase(q, theta):
   """
      Multiplies by the overall phase Exp[I theta].
      :param q: (QuantumRegister) Quantum Register that stores the wavefunction
      :param theta: (float) the value of the phase
      :return: (QuantumCircuit) Circuit implementing phase rotation
      """
   qc = QuantumCircuit(q)
   qc.p(theta, q[0])
   qc.x(q[0])
   qc.p(theta, q[0])
   qc.x(q[0])
   return qc

####################################################################
def phase_estimation(q, phase_qubits, phase_classical, unitary):
    """
    Constructs quantum circuit for quantum phase estimation of a unitary operator
    :param q: (QuantumRegister) Register on which unitary operates
    :param phase_qubits: (QuantumRegister) Register for storing qubits for phase readout
    :param phase_classical: (ClassicalRegister) Register for storing classical qubits for phase readout
    :param unitary: (CircuitFactory) Unitary operator Circuit Factory
    :return: (QuantumCircuit) Circuit implementing phase estimation
    """
    _qc = QuantumCircuit(q, phase_qubits, phase_classical)
    num_phase_bits = phase_qubits.size

    for i in range(num_phase_bits):
        _qc.h(phase_qubits[i])
    for i in (range(num_phase_bits)):
        j = 2 ** (i)
        for k in range(j):
            unitary.build_controlled(_qc, q, phase_qubits[i]) ## WARNING
    _qc += fourier_transform(phase_qubits, inv=True)
    _qc.measure(phase_qubits, phase_classical)
    return _qc


####################################################################
def exp_pauli_product(q, theta, pauli_list):
    """
    Constructs quantum circuit for exp[-I theta pauli_i pauli_j ...]
    :param q: (QuantumRegister) Register on which unitary operates
    :param theta: (float) the value of theta
    :param pauli_list: List of pairs of [P, i], where P = {I, X,Y,Z} 
                       denotes the pauli matrix and i the position at which it operates
                       Note that 'I' does not need to be specified (can be omitted from list)
    :return: (QuantumCircuit) The resulting quantum circuit

    Predcondition: len(pauli_list) > 0
    """
    qc = QuantumCircuit(q)
    #Do the basis rotation to the Z basis
    for pauli in pauli_list:
        if pauli[0] == 'I' or pauli[0] == 'Z':
            continue
        if pauli[0] == 'X':
            qc.h(pauli[1])
        if pauli[0] == 'Y':
            qc.sdg(pauli[1])
            qc.h(pauli[1])             
        
    theta *= 2 #RZ = diag[1, exp(i theta)]
               #   = exp(i theta/2) diag[exp(-i theta/2, exp(i theta/2))] 

    for i in range(0, len(pauli_list)-1, 1):
        qc.cx(pauli_list[i][1], pauli_list[i+1][1])
    qc.rz(theta, pauli_list[-1][1])
    for i in range(len(pauli_list)-1, 0, -1):
        qc.cx(pauli_list[i-1][1], pauli_list[i][1])

    #Do the basis rotation from the Z basis
    for pauli in pauli_list:
        if pauli[0] == 'I' or pauli[0] == 'Z':
            continue
        if pauli[0] == 'X':
            qc.h(q[pauli[1]])
        if pauli[0] == 'Y':
            qc.h(pauli[1])
            qc.s(pauli[1])

    return qc

####################################################################
def get_pauli_list(sequence):
   """
   Scans string of pauli matrix sequence and returns a list of the position of z gates
   :param sequence: (str) string which indicates order of pauli matrices used in rotations. MSB first.
   :return: (list) of Tuples (str: pauli operation, int: qubit position)
   """
   return [(sequence[s], len(sequence)-1 -s) for s in range(len(sequence)) if sequence[s] != 'I']




