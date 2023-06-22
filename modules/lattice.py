import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, execute, Aer
import math
from numpy.linalg import cholesky, inv
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from operator import add 

import sys
sys.path.append('modules')
import basic_circuits

class Lattice():
    """
    Class that implements quantum circuit for Suzuki-Trotter expanded time evolution of a Hamiltonian.
    """
    def __init__(self, dimension, nL, dx, nQ, num_ancilla, twisted=True): #Generic
        """
        Initializes parameters.
        :param dimension: (int) Number of dimensions of the Lattice
        :param nL:        (int)   Number of sites per dimension
        :param nQ:        (int)   Number of qubits per site
        :param twisted:   (float) Whether to use twisted boundary conditions
        """
        assert(dimension == 1) #Currently only implemented for 1-d lattices
        self._dimension = dimension
        self._theta = np.pi * twisted
        self._nL = nL
        self._dx = dx
        self._nQ = nQ # self._nQ = nQ
        self._num_ancilla = num_ancilla
        self._twisted = twisted
        self._q_register = QuantumRegister(nQ * nL ** dimension)
        self._a_register = QuantumRegister(num_ancilla)
    @property
    def nQ(self):
        return self._nQ
    @nQ.setter
    def nQ(self, nQ): #SCF
        self._nQ= nQ
    def is_twisted(self):
        return self._twisted
    def get_a_register(self):
        return self._a_register
    def get_q_register(self):
        return self._q_register
    def dp(self):
        return 2 * math.pi / self._dx / self._nL
    def p_max(self):
        if self._nL % 2 == 0:
            return self.dp() * self._nL / 2
        else:
            return self.dp() * (self._nL - 1) / 2
    def x_max(self):
        if self._nL % 2 == 0:
            return self._dx * self._nL / 2
        else:
            return self._dx * (self._nL - 1) / 2
    def x_list(self): # Edited
        return np.arange(0,self._nL)*self._dx - self.x_max()
    def p_list(self):
        return self.dp() * (np.arange(0,self._nL) - self.p_max() / self.dp() - self._theta / 2 / math.pi)
    def _subregister_start(self,lattice_point): #Generic
        """
        Compute starting point of the subregister
        :param lattice_point: The lattice point (list of length dimension)
        :return: the starting value of the register
        """
        assert(len(lattice_point) == self._dimension)
        result=0   
        for i, point in enumerate(lattice_point):
            result += point * self._dimension**(self._dimension - i - 1)
        return self._nQ * result
    
    def _subregister_map(self,lattice_point):
        """
        Compute subregister map
        :param lattice_point: The lattice point (list of length dimension)
        :return: map of subregister onto full register
        """
        start = self._subregister_start(lattice_point)
        return list(range(start, start + self._nQ))
    
    def _get_subregister(self, lattice_point):
        """
        Compute subregister information
        :param lattice_point: The lattice point (list of length dimension)
        :return: [Quantum register of size _nQ, map of subregister onto full register]
        """
        qbit_map = self._subregister_map(lattice_point)
        q_register = QuantumRegister(self._nQ)
        return [q_register, qbit_map]
    def apply_single_operator(self, pos_op): #Generic
        """
        Applies an operators at given lattice position
        :param pos_op: Position and operators to apply
            Is in the follwing form
                The lattice point (list of length dimension)
                The operator (basic_operator_interface) to apply
                A list of parameters to apply
        """
        assert(len(pos_op)==3)
        position = pos_op[0]
        operator = pos_op[1]
        params = pos_op[2]
        point = [0] * self._dimension
        #Take the position modulo the number of lattice sites, and keep track of the wrap
        wrap = 0
        for i in range(self._dimension):
            (tmpwrap,point[i]) = divmod(position[i], self._nL)
            wrap+= tmpwrap
        if self._twisted:
            params *= (-1)**wrap
            
        [qbit_register, qbit_map] = self._get_subregister(point)
        qc_sub = operator.build_operator_circuit([qbit_register], self._a_register, params)
        qc = QuantumCircuit(self._q_register)
        qc.compose(qc_sub, qbit_map, inplace=True)
        return qc

    def apply_single_operator_list(self,pos_op_list): #Generic
        """
        Applies operators at given lattice sites
        :param pos_op_list: List of positions and operators to apply
            This list is of the following form:
                The lattice point (list of length dimension)
                The operator (basic_operator_interface) to apply
                A list of parameters to apply
        """
        qc = QuantumCircuit(self._q_register)
        wrap = 0 #Number holding the total wrap needed to implement the phase offset in the boundary condition
        for i, pos_op in  enumerate(pos_op_list):
            assert(len(pos_op)==3)
            position = pos_op[0]
            operator = pos_op[1]
            params = pos_op[2]
            point = [0] * self._dimension
            for i in range(self._dimension):
                (tmpwrap,point[i]) = divmod(position[i], self._nL)
                wrap += tmpwrap
            if self._twisted:
                #params *= (-1)**wrap
                for j in range(len(params)): params[j]*= (-1)**wrap

            [qbit_register, qbit_map] = self._get_subregister(point)
            qc_sub = operator.build_operator_circuit([qbit_register], self._a_register, params)
            qc.compose(qc_sub, qbit_map, inplace=True)
        return qc
    
    