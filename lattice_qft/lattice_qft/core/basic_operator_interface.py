from abc import ABC, abstractmethod


class BasicOperator(ABC):
    """
    Abstract base class which implements time-evolved operator quantum circuit.
    """
    def __init__(self, num_params):
        self.num_params = num_params

    @staticmethod
    def check_params(input_params, num_params):
        """
        Checks if number of parameters for Operator matches function input.
        :param input_params: (int) number of input parameters.
        :param num_params: (int) number of parameters operator accepts.
        :return: None
        """
        if len(input_params) != num_params:
            raise ValueError('Expected {n} parameters for prefactor, but got {input}.'.format(n=num_params, input = len(input_params)))

    #@abstractmethod
    #def phase(self, Nq, prefactor):
        """
        Calculates phase error induced by Qiskit Rz implementation.
        :param Nq: (int) number of qubits in wavefunction quantum circuit.
        :param prefactor: (float) prefactor to multiply operator by.
        :param time: (float) time step length for operator.
        :return: (float) phase error.
        """
    #    raise NotImplementedError

    @abstractmethod
    def build_operator_circuit(self, qregs, ancilla_qubits, params, correct_phase, qreg_offset=0):
        """
        Builds quantum circuit for time-evolved operator quantum circuit.
        :param qregs: (List of QuantumRegisters) Quantum registers that operator circuit acts on.
        :param ancilla_qubits: (QuantumRegister) Ancilla qubit register.
        :param params: (List) Parameters needed to construct operator.
        :param time: (Float) Length of time to evolve with operator over.
        :param correct_phase: (Bool) True if phase error is to be corrected.
        :return: None, operates on circuit in-place.
        """
        raise NotImplementedError

    #@abstractmethod
    #def build_phase_correction_circuit(self, qregs, params, qreg_offset=0):
        """
        Builds quantum circuit to correct for phase error caused by Qiskit implementation of Rz gate.
        :param qregs: (List of QuantumRegisters) Quantum registers that operator circuit acts on.
        :param params: (List) Parameters needed to construct operator.
        :param time: (Float) Length of time to evolve with operator over.
        :return: None, operates on circuit in-place.
        """
    #    raise NotImplementedError
