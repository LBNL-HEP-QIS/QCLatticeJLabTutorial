import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, execute, Aer
import math
from numpy.linalg import cholesky, inv
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution
from operator import add 
from lattice_qft.core.error_checking_utils import *


import lattice_qft.core.basic_circuits as basic_circuits

class Multidimensional():
    def __init__(self, grid):
        self._grid = np.array(grid)
        self.new_grid = []
    def __add__(self, a):
        if type(a) == int or type(a) == float or type(a) == complex:
            for inner_elements in self._grid:
                self.new_grid.append(inner_elements + a)
            return self.new_grid
    def __sub__(self, a):
        if type(a) == int or type(a) == float or type(a) == complex:
            for inner_elements in self._grid:
                self.new_grid.append(inner_elements - a)
            return self.new_grid
    def __truediv__(self, a):
        if type(a) == int or type(a) == float or type(a) == complex:
            for inner_elements in self._grid:
                self.new_grid.append(inner_elements / a)
            return self.new_grid
    def __mul__(self, a):
        if type(a) == int or type(a) == float or type(a) == complex:
            for inner_elements in self._grid:
                self.new_grid.append(inner_elements * a)
            return self.new_grid 
      

        


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
        #assert(dimension == 1) #Currently only implemented for 1-d lattices
        
        #Custom function and file
        #typeerror_check(int, dimension, nL, nQ, num_ancilla)
        #real_positive_check(dimension, nL, dx, nQ, num_ancilla)
        #typeerror_check((int, float), dx)
        self._dimension = dimension
        self._theta = np.pi * twisted
        self._nL_site, self._nL_link, self._nL_plaquette = nL
        self._dx = dx
        self._nQ = nQ # self._nQ = nQ
        self._num_ancilla = [i for i in num_ancilla]
        self._twisted = twisted
       # self._q_register = QuantumRegister(nQ * nL ** dimension)
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
        x_list_temp = []
        for dim in range(self._dimension):
            x_list_temp.append(np.arange(0,self._nL)*self._dx - self.x_max())
        return Multidimensional(np.meshgrid(*np.array(x_list_temp)))
    def p_list(self):
        p_list_temp = []
        for dim in range(self._dimension):
            p_list_temp.append(self.dp() * (np.arange(0,self._nL) - self.p_max() / self.dp() - self._theta / 2 / math.pi))
        return Multidimensional(np.meshgrid(*np.array(p_list_temp)))
    def _subregister_start(self,lattice_point): #Generic
        """
        Compute starting point of the subregister
        :param lattice_point: The lattice point (list of length dimension)
        :return: the starting value of the register
        """
        if not isinstance(lattice_point, (list, type(np.array([])))):
            raise TypeError("<method 'apply_single_operator'> expects list or <class 'numpy.ndarray'> for pos_op parameter.")
        #assert(len(lattice_point) == self._dimension)
        result=0   
        for i, point in enumerate(lattice_point):
            result += point * self._dimension**(self._dimension - i - 1)
        return self._nQ * result
    def get_num_sites_per_dim(self):
        return self._nL
    def _subregister_map(self,lattice_point):
        """
        Compute subregister map
        :param lattice_point: The lattice point (list of length dimension)
        :return: map of subregister onto full register
        """
        typeerror_check((type(np.array([])),list),lattice_point)
        start = self._subregister_start(lattice_point)
        return list(range(start, start + self._nQ))
    
    def _get_subregister(self, lattice_point):
        #doxygen
        """
        Compute subregister information
        :param lattice_point: The lattice point (list of length dimension)
        :return: [Quantum register of size _nQ, map of subregister onto full register]
        """
        typeerror_check((type(np.array([])),list),lattice_point)
        qbit_map = self._subregister_map(lattice_point)
        q_register = QuantumRegister(self._nQ)
        return [q_register, qbit_map]
    def apply_single_operator(self, pos_op): #Generic
        """
        Applies an operators at given lattice position
        :param pos_op: Position and operators to apply
            Is in the following form
                The lattice point (list of length dimension)
                The operator (basic_operator_interface) to apply
                A list of parameters to apply
        """
        #typeerror_check((type(np.array([])),list), pos_op)
        assert(len(pos_op)==3)
        #DimensionError(pos_op, (3,null))
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
        typeerror_check((type(np.array([])),list), pos_op_list)
        qc = QuantumCircuit(self._q_register)
        wrap = 0 #Number holding the total wrap needed to implement the phase offset in the boundary condition
        for i, pos_op in  enumerate(pos_op_list):
            assert(len(pos_op)==3)#####
            position = pos_op[0]
            operator = pos_op[1]
            params = pos_op[2]
            point = [0] * self._dimension
            for i in range(self._dimension):
                (tmpwrap,point[i]) = divmod(position[i], self._nL)
                wrap += tmpwrap
            if self._twisted:
                for j in range(len(params)): params[j]*= (-1)**wrap

            [qbit_register, qbit_map] = self._get_subregister(point)
            qc_sub = operator.build_operator_circuit([qbit_register], self._a_register, params)
            qc.compose(qc_sub, qbit_map, inplace=True)
        return qc
    @staticmethod
    def grids_to_points(grids):
        return np.moveaxis(np.array(grids), 0, grids[0].ndim).reshape(-1, len(grids))
    

test = (Lattice(3, (2,3,4), 1, 1, (1,2,3))._num_ancilla)

print(test)

# testing = t1 - 2
# xv, yv, zv = testing
# xv, yv, zv = Lattice(3, 3, 1, 1, 1).p_list()._grid
# xv1, yv1, zv1 = Lattice(3, 3, 1, 1, 1).x_list()._grid
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax1 = fig.add_subplot(1, 2, 2,projection='3d')
# ax.scatter(xv, yv, zv, marker='o')
# ax1.scatter(xv1, yv1, zv1, marker='x', c='r')
# plt.show()
