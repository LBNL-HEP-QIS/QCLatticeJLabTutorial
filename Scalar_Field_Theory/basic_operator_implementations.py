import math
import numpy as np
from qiskit import QuantumCircuit

import sys

sys.path.append("Scalar_Field_Theory")
sys.path.append("modules")
import basic_circuits
from basic_operator_interface import BasicOperator

####################################################################
###############Implementations of Base Class Operators##############
####################################################################
class PhiOperator(BasicOperator):
    def __init__(self, phi_max):
        self.phi_max = phi_max
        super().__init__(1)

    def build_operator_circuit(self, qregs, ancilla_qubits, params):
        self.check_params(params, self.num_params)
        qreg = qregs[0]
        param = params[0]
        return PhiOperator.phi(self, qreg, param)

    #@staticmethod
    def phi(self, q, prefactor):
        """
        Constructs quantum circuit for the operator Exp[-I prefactor * phi * t]
        :param q: (QuantumRegister) Quantum Register that stores the wavefunction.
        :param prefactor: (float) factor to multiply by.
        :return: (QuantumCircuit) Circuit that implements Exp[-I prefactor * phi^2 * t].
        """
        qc = QuantumCircuit(q)
        Nq = q.size
        theta = _dphi(Nq, self.phi_max) * prefactor
        coeffs = _phi_coeffs_dict.get(Nq)
        for n in range(len(coeffs)):
            if coeffs[n] != 0:
                qc.compose(
                    basic_circuits.exp_pauli_product(
                        q,
                        theta * coeffs[n],
                        basic_circuits.get_pauli_list(_pauli_sequences.get(Nq)[n]),
                    ),
                    inplace=True,
                )
        return qc

    @staticmethod
    def phi_phase(self, Nq, prefactor):
        """
        Calculates phase angle to rotate over in implementation of phi^2 circuit.
        :param Nq: (int) number of qubits in wavefunction digitization.
        :param prefactor: (float) prefector to multiply by.
        :return: (float) value of phase angle to rotate over.
        """
        return ((2**Nq - 1) / 2 * _dphi(Nq, self.phi_max)) * prefactor


class Phi2Operator(BasicOperator):
    def __init__(self, phi_max):
        self.phi_max = phi_max
        super().__init__(1)

    def build_operator_circuit(self, qregs, ancilla_qubits, params):
        self.check_params(params, self.num_params)
        qreg = qregs[0]
        param = params[0]
        return Phi2Operator.phi2(self, qreg, param)

    #@staticmethod
    def phi2(self, q, prefactor):
        """
        Constructs quantum circuit for the operator Exp[-I prefactor * phi^2 * t]
        :param q: (QuantumRegister) Quantum Register that stores the wavefunction.
        :param prefactor: (float) factor to multiply by.
        :return: (QuantumCircuit) Circuit that implements Exp[-I prefactor * phi^2 * t].
        """
        qc = QuantumCircuit(q)
        Nq = q.size
        theta = _dphi(Nq, self.phi_max) ** 2 * prefactor
        coeffs = _phi2_coeffs_dict.get(Nq)  # Terms with Z operators
        coeffsI = _phi_coeffs_dict.get(Nq)  # Terms proportional to the identity

        for n in range(len(coeffs)):
            # Append terms with Z operators
            if coeffs[n] != 0:
                qc.compose(
                    basic_circuits.exp_pauli_product(
                        q,
                        theta * coeffs[n],
                        basic_circuits.get_pauli_list(_pauli_sequences.get(Nq)[n]),
                    ),
                    inplace=True,
                )
            # Append terms proportional to the identity
            if coeffsI[n] != 0:
                qc.compose(
                    basic_circuits.phase(q, -theta * coeffsI[n] ** 2), inplace=True
                )

        return qc

    @staticmethod
    def phi2_phase(self, Nq, prefactor):
        """
        Calculates phase angle to rotate over in implementation of phi^2 circuit.
        :param Nq: (int) number of qubits in wavefunction digitization.
        :param prefactor: (float) prefector to multiply by.
        :return: (float) value of phase angle to rotate over.
        """
        return ((2**Nq - 1) / 2 * _dphi(Nq, self.phi_max)) ** 2 * prefactor


####################################################################
class Pi2Operator(BasicOperator):
    def __init__(self, phi_max):
        self.phi_max = phi_max
        super().__init__(1)

    def build_operator_circuit(self, qregs, ancilla_qubits, params):
        self.check_params(params, self.num_params)
        qreg = qregs[0]
        param = params[0]
        return Pi2Operator.pi2(self, qreg, param)

    
    def pi2(self, q, prefactor, swap=False):
        """
        Constructs quantum circuit for the operator Exp[-I prefactor * pi^2 * t].
        :param q: (QuantumRegister) Quantum Register that stores the wavefunction.
        :param prefactor: (float) factor to multiply by.
        :return: (QuantumCircuit) Circuit that implements Exp[-I prefactor * pi^2 * t].
        """
        qc = QuantumCircuit(q)
        Nq = q.size
        theta = _dpi(Nq, self.phi_max) ** 2 * prefactor
        qc.compose(
            basic_circuits.fourier_transform_symmetric(q, inv=False, swap=swap),
            inplace=True,
        )
        if swap == False:
            qc.qubits.reverse()

        coeffs = _phi2_coeffs_dict.get(Nq)  # Terms with Z operators
        coeffsI = _phi_coeffs_dict.get(Nq)  # Terms proportional to the identity

        for n in range(len(coeffs)):
            # Append terms with Z operators
            if coeffs[n] != 0:
                qc.compose(
                    basic_circuits.exp_pauli_product(
                        q,
                        theta * coeffs[n],
                        basic_circuits.get_pauli_list(_pauli_sequences.get(Nq)[n]),
                    ),
                    qc.qubits,
                    inplace=True,
                )

            # Append terms proportional to the identity
            if coeffsI[n] != 0:
                qc.compose(
                    basic_circuits.phase(q, -theta * coeffsI[n] ** 2), inplace=True
                )

        if swap == False:
            qc.qubits.reverse()
        qc.compose(
            basic_circuits.fourier_transform_symmetric(q, inv=True, swap=swap),
            inplace=True,
        )

        return qc

    
    def pi2_phase(self, Nq, prefactor):
        """
        Calculates phase angle to rotate over in implementation of pi^2 circuit.
        :param Nq: (int) number of qubits in wavefunction digitization.
        :param prefactor: (float) prefector to multiply by.
        :return: (float) value of phase angle to rotate over.
        """
        return ((2**Nq - 1) / 2 * _dpi(Nq, self.phi_max)) ** 2 * prefactor


####################################################################
class Phi4Operator(BasicOperator):
    def __init__(self, phi_max):
        self.phi_max = phi_max
        super().__init__(1)

    def build_operator_circuit(self, qregs, ancilla_qubits, params):
        self.check_params(params, self.num_params)
        qreg = qregs[0]
        param = params[0]
        return Phi4Operator.phi4(self, qreg, param)

    #@staticmethod
    def phi4(self, q, prefactor):
        """
        Constructs quantum circuit for the operator Exp[-I prefactor * phi^4 * t].
        :param q: (QuantumRegister) Quantum Register that stores the wavefunction.
        :param prefactor: (float) factor to multiply by.
        :return: (QuantumCircuit) Circuit that implements Exp[-I prefactor * phi^4 * t].
        """
        qc = QuantumCircuit(q)
        Nq = q.size
        theta = _dphi(Nq, self.phi_max) ** 4 * prefactor
        phi4coeff = _phi4_coeffs_dict.get(Nq)

        #########
        phicoeff = _phi_coeffs_dict.get(Nq)
        phi2coeff = _phi2_coeffs_dict.get(Nq)
        #########

        for n in range(len(_phi4_coeffs_dict.get(Nq))):
            if phi4coeff[n] != 0:
                qc.compose(
                    basic_circuits.exp_pauli_product(
                        q,
                        theta * phi4coeff[n],
                        basic_circuits.get_pauli_list(_pauli_sequences.get(Nq)[n]),
                    ),
                    inplace=True,
                )

            #########
            if phi2coeff[n] != 0:
                qc.compose(
                    basic_circuits.phase(q, -theta * phi2coeff[n] ** 2), inplace=True
                )
            #########

        #########
        qc.compose(
            basic_circuits.phase(q, -theta * np.sum(np.array(phicoeff) ** 2) ** 2),
            inplace=True,
        )
        #########
        return qc

    #@staticmethod
    def phi4_phase(self, Nq, prefactor):
        """
        Calculates phase angle to rotate over in implementation of phi^4 circuit.
        :param Nq: (int) number of qubits in wavefunction digitization.
        :param prefactor: (float) prefector to multiply by.
        :return: (float) value of phase angle to rotate over.
        """
        return ((2**Nq - 1) / 2 * _dphi(Nq, self.phi_max)) ** 4 * prefactor


####################################################################
class Phi2Phi4Operator(BasicOperator):  # prefactor is a 2 element list
    def __init__(self, phi_max):
        self.phi_max = phi_max
        super().__init__(2)

    def build_operator_circuit(self, qregs, ancilla_qubits, params):
        self.check_params(params, self.num_params)
        prefactor2 = params[0]
        prefactor4 = params[1]
        qreg = qregs[0]
        return Phi2Phi4Operator.phi2_phi4(qreg, prefactor2, prefactor4)

    #@staticmethod
    def phi2_phi4(self, q, prefactor2, prefactor4):
        """
        Constructs quantum circuit for the operator Exp[-I (prefactor2 * phi^2 + prefactor4 * phi^4) * t]
        :param q: (QuantumRegister) Quantum Register that stores the wavefunction.
        :param prefactor: (float) factor to multiply by.
        :return: (QuantumCircuit) Circuit that implements operator.
        """
        qc = QuantumCircuit(q)
        Nq = q.size
        assert len(_phi2_coeffs_dict.get(Nq)) == len(_phi4_coeffs_dict.get(Nq))
        ##########
        phi1coeff = _phi_coeffs_dict.get(Nq)
        ##########
        phi2coeff = _phi2_coeffs_dict.get(Nq)
        phi4coeff = _phi4_coeffs_dict.get(Nq)
        for n in range(len(_phi4_coeffs_dict.get(Nq))):
            if phi2coeff[n] != 0 or phi4coeff[n] != 0:  # CHANGED FROM and TO or
                theta = _dphi(Nq, self.phi_max) ** 2 * prefactor2 * phi2coeff[n]
                theta += _dphi(Nq, self.phi_max) ** 4 * prefactor4 * phi4coeff[n]

                qc.compose(
                    basic_circuits.exp_pauli_product(
                        q,
                        theta,
                        basic_circuits.get_pauli_list(_pauli_sequences.get(Nq)[n]),
                    ),
                    inplace=True,
                )

            if phi2coeff[n] != 0:
                qc.compose(
                    basic_circuits.phase(
                        q,
                        -_dphi(Nq, self.phi_max) ** 4
                        * prefactor4
                        * phi2coeff[n] ** 2,
                    ),
                    inplace=True,
                )
            if phi1coeff[n] != 0:
                qc.compose(
                    basic_circuits.phase(
                        q,
                        -_dphi(Nq, self.phi_max) ** 2
                        * prefactor2
                        * phi1coeff[n] ** 2,
                    ),
                    inplace=True,
                )

        qc.compose(
            basic_circuits.phase(
                q,
                -_dphi(Nq, self.phi_max) ** 4
                * prefactor4
                * np.sum(np.array(phi1coeff) ** 2) ** 2,
            ),
            inplace=True,
        )

        return qc

    #@staticmethod
    def phi2_phi4_phase(self, Nq, prefactor2, prefactor4):
        """
        Calculates phase angle to rotate over in implementation of phi^4 circuit.
        :param Nq: (int) number of qubits in wavefunction digitization.
        :param prefactor: (float) prefector to multiply by.
        :return: (float) value of phase angle to rotate over.
        """
        phase_offset = (
            (2**Nq - 1) / 2 * _dphi(Nq, self.phi_max)
        ) ** 2 * prefactor2
        phase_offset += (
            (2**Nq - 1) / 2 * _dphi(Nq, self.phi_max)
        ) ** 4 * prefactor4
        return phase_offset


####################################################################
class PhiTensorXOperator(BasicOperator):  # remember to add phase circuit from states.py
    def __init__(self, phi_max):
        self.phi_max = phi_max
        assert False, "This class is untested."
        self.num_params = 1

    def build_operator_circuit(self, qregs, ancilla_qubits, params):
        self.check_params(params, self.num_params)
        qreg = qregs[0]
        param = params[0]
        return PhiTensorXOperator.phi_tensor_x(qreg, ancilla_qubits, param)

    @staticmethod
    def phi_tensor_x(self, q, a, prefactor):
        """
        Constructs quantum circuit for Exp[-I (phi x sigmax) t]
        :param q: (QuantumRegister) Quantum Register that stores the wavefunction.
        :param a: (Qubit) ancillary qubit.
        :param prefactor: (float) factor to multiply by.
        :return: (QuantumCircuit) Circuit implementing operator.
        """
        qc = QuantumCircuit(q, a)
        Nq = q.size
        qc.h(a[0])
        for n in reversed(range(Nq)):
            qc.cx(a[0], q[n])
            qc.rz(_dphi(Nq, self.phi_max) * 2**n * prefactor, q[n])
            qc.cx(a[0], q[n])
        qc.h(a[0])
        return qc


####################################################################
class PiTensorYOperator(BasicOperator):
    def __init__(self, phi_max):
        self.phi_max = phi_max
        assert False, "This class is untested."
        self.num_params = 1

    def build_operator_circuit(self, qregs, ancilla_qubits, params):
        self.check_params(params, self.num_params)
        qreg = qregs[0]
        param = params[0]
        return PiTensorYOperator.pi_tensor_y(qreg, ancilla_qubits, param)

    @staticmethod
    def pi_tensor_y(self, q, a, prefactor, swap=False):
        """
        Constructs quantum circuit for Exp[-I (pi x sigmay) t]
        :param q: (QuantumRegister) Quantum Register that stores the wavefunction.
        :param a: (Qubit) ancillary qubit.
        :param prefactor: (float) factor to multiply by.
        :return: (QuantumCircuit) Circuit implementing operator.
        """
        qc = QuantumCircuit(q, a)
        Nq = q.size
        qc.h(a[0])

        # qc.compose(basic_circuits.fourier_transform_symmetric(q), inplace=True)
        qc.compose(
            basic_circuits.fourier_transform_symmetric(q, inv=False, swap=swap),
            inplace=True,
        )
        if not swap:
            qc.qubits.reverse()

        for n in reversed(range(Nq)):
            qc.sdg(a[0])
            qc.h(a[0])
            qc.cx(a[0], q[n])
            qc.rz(_dpi(Nq, self.phi_max) * 2**n * prefactor, q[n])
            qc.cx(a[0], q[n])
            qc.h(a[0])
            qc.s(a[0])

        if not swap:
            qc.qubits.reverse()
        qc.compose(
            basic_circuits.fourier_transform_symmetric(q, inv=True, swap=swap),
            inplace=True,
        )

        qc.h(a[0])
        return qc


####################################################################
####     Private functions and classes      ########################
####################################################################

####################################################################
"""
Dictionary for coefficient used in building circuits for phi and pi operators.
These are created with the Mathematica file QuantumCircuits on the Dropbox/Christian
"""
_phi_coeffs_dict = {
    1: [-1 / 2],
    2: [-1 / 2, -1, 0],
    3: [-1 / 2, -1, 0, -2, 0, 0, 0],
    4: [-1 / 2, -1, 0, -2, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0],
    5: [
        -1 / 2,
        -1,
        0,
        -2,
        0,
        0,
        0,
        -4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    6: [
        -1 / 2,
        -1,
        0,
        -2,
        0,
        0,
        0,
        -4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -16,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
}


####################################################################
"""
Dictionary for coefficient used in building circuits for phi^2 and pi^2 operators.
These are created with the Mathematica file QuantumCircuits on the Dropbox/Christian
"""
_phi2_coeffs_dict = {
    1: [0],
    2: [0, 0, 1],
    3: [0, 0, 1, 0, 2, 4, 0],
    4: [0, 0, 1, 0, 2, 4, 0, 0, 4, 8, 0, 16, 0, 0, 0],
    5: [
        0,
        0,
        1,
        0,
        2,
        4,
        0,
        0,
        4,
        8,
        0,
        16,
        0,
        0,
        0,
        0,
        8,
        16,
        0,
        32,
        0,
        0,
        0,
        64,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    6: [
        0,
        0,
        1,
        0,
        2,
        4,
        0,
        0,
        4,
        8,
        0,
        16,
        0,
        0,
        0,
        0,
        8,
        16,
        0,
        32,
        0,
        0,
        0,
        64,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        16,
        32,
        0,
        64,
        0,
        0,
        0,
        128,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        256,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
}

####################################################################
"""
Dictionary for coefficient used in building circuits for phi^4 operators.
These are created with the Mathematica file QuantumCircuits on the Dropbox/Christian
"""
_phi4_coeffs_dict = {
    1: [0],
    2: [0, 0, 5 / 2],
    3: [0, 0, 53 / 2, 0, 29, 46, 0],
    4: [0, 0, 245 / 2, 0, 221, 430, 0, 0, 250, 476, 0, 760, 0, 0, 96],
    5: [
        0,
        0,
        1013 / 2,
        0,
        989,
        1966,
        0,
        0,
        1786,
        3548,
        0,
        6904,
        0,
        0,
        96,
        0,
        2036,
        4024,
        0,
        7664,
        0,
        0,
        192,
        12256,
        0,
        0,
        384,
        0,
        768,
        1536,
        0,
    ],
    6: [
        0,
        0,
        4085 / 2,
        0,
        4061,
        8110,
        0,
        0,
        7930,
        15836,
        0,
        31480,
        0,
        0,
        96,
        0,
        14324,
        28600,
        0,
        56816,
        0,
        0,
        192,
        110560,
        0,
        0,
        384,
        0,
        768,
        1536,
        0,
        0,
        16360,
        32624,
        0,
        64480,
        0,
        0,
        384,
        122816,
        0,
        0,
        768,
        0,
        1536,
        3072,
        0,
        196480,
        0,
        0,
        1536,
        0,
        3072,
        6144,
        0,
        0,
        6144,
        12288,
        0,
        24576,
        0,
        0,
        0,
    ],
}

####################################################################
"""
Stores the sequence of Pauli gates used in building circuits for phi^2 and pi^2 operators.
These are created with the Mathematica file QuantumCircuits on the Dropbox/Christian
"""
_pauli_sequences = {
    1: ["Z"],
    2: ["IZ", "ZI", "ZZ"],
    3: ["IIZ", "IZI", "IZZ", "ZII", "ZIZ", "ZZI", "ZZZ"],
    4: [
        "IIIZ",
        "IIZI",
        "IIZZ",
        "IZII",
        "IZIZ",
        "IZZI",
        "IZZZ",
        "ZIII",
        "ZIIZ",
        "ZIZI",
        "ZIZZ",
        "ZZII",
        "ZZIZ",
        "ZZZI",
        "ZZZZ",
    ],
    5: [
        "IIIIZ",
        "IIIZI",
        "IIIZZ",
        "IIZII",
        "IIZIZ",
        "IIZZI",
        "IIZZZ",
        "IZIII",
        "IZIIZ",
        "IZIZI",
        "IZIZZ",
        "IZZII",
        "IZZIZ",
        "IZZZI",
        "IZZZZ",
        "ZIIII",
        "ZIIIZ",
        "ZIIZI",
        "ZIIZZ",
        "ZIZII",
        "ZIZIZ",
        "ZIZZI",
        "ZIZZZ",
        "ZZIII",
        "ZZIIZ",
        "ZZIZI",
        "ZZIZZ",
        "ZZZII",
        "ZZZIZ",
        "ZZZZI",
        "ZZZZZ",
    ],
    6: [
        "IIIIIZ",
        "IIIIZI",
        "IIIIZZ",
        "IIIZII",
        "IIIZIZ",
        "IIIZZI",
        "IIIZZZ",
        "IIZIII",
        "IIZIIZ",
        "IIZIZI",
        "IIZIZZ",
        "IIZZII",
        "IIZZIZ",
        "IIZZZI",
        "IIZZZZ",
        "IZIIII",
        "IZIIIZ",
        "IZIIZI",
        "IZIIZZ",
        "IZIZII",
        "IZIZIZ",
        "IZIZZI",
        "IZIZZZ",
        "IZZIII",
        "IZZIIZ",
        "IZZIZI",
        "IZZIZZ",
        "IZZZII",
        "IZZZIZ",
        "IZZZZI",
        "IZZZZZ",
        "ZIIIII",
        "ZIIIIZ",
        "ZIIIZI",
        "ZIIIZZ",
        "ZIIZII",
        "ZIIZIZ",
        "ZIIZZI",
        "ZIIZZZ",
        "ZIZIII",
        "ZIZIIZ",
        "ZIZIZI",
        "ZIZIZZ",
        "ZIZZII",
        "ZIZZIZ",
        "ZIZZZI",
        "ZIZZZZ",
        "ZZIIII",
        "ZZIIIZ",
        "ZZIIZI",
        "ZZIIZZ",
        "ZZIZII",
        "ZZIZIZ",
        "ZZIZZI",
        "ZZIZZZ",
        "ZZZIII",
        "ZZZIIZ",
        "ZZZIZI",
        "ZZZIZZ",
        "ZZZZII",
        "ZZZZIZ",
        "ZZZZZI",
        "ZZZZZZ",
    ],
}

####################################################################
def _dphi(Nq, phi_max):
    """
    Computes the distance between the phi digitization.
    :param Nq: (int) Number of qubits in wavefunction representation.
    :param phi_max: (float) phi_max value.
    :return: (float) dphi value.
    """
    Ns = 2**Nq
    return 2 * phi_max / (Ns - 1)


####################################################################
def _dpi(Nq, phi_max):
    """
    Computes the distance between the pi digitization.
    :param Nq: (int) Number of qubits in wavefunction representation.
    :param phi_max: (float) phi_max value.
    :return: (float) dpi value.
    """
    Ns = 2**Nq
    print(phi_max)
    return math.pi / _dphi(Nq, phi_max) * 2 / Ns
