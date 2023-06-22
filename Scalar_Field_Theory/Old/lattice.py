import itertools
import math
from operator import add 
import numpy as np
from numpy.linalg import cholesky, inv

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
#from qiskit.aqua.components.uncertainty_models import MultivariateNormalDistribution
from qiskit_finance.circuit.library.probability_distributions import NormalDistribution

import sys
sys.path.append('modules')
#sys.path.append('arithmetic_pkg')

import basic_circuits
import settings as settings
import basic_operator_implementations as basic_op_cf

import classical as classical


sys.path.append('arithmetic_pkg')
from shear import *

sys.path.append('Scalar_Field_Theory')


####################################################################
####################### Defintion of Lattice #######################
####################################################################
class Lattice(): #include building phase correction into build operator method?P
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
        self.nQ = nQ # self._nQ = nQ
        self._num_ancilla = num_ancilla
        self._twisted = twisted
        self._q_register = QuantumRegister(nQ * nL ** dimension)
        self._a_register = QuantumRegister(num_ancilla)

    def get_dimension(self):
        return self._dimension

    @property
    def nQ(self):
        return self._nQ

    @nQ.setter
    def nQ(self, nQ): #SCF
        self._nQ= nQ
    #    self._nPhi= 2**nQ
    #    self._phiMax= np.sqrt(1/np.average(self.omega_list()) * np.pi/2 * (self._nPhi - 1)**2/self._nPhi)
    #    settings.phi_max= self._phiMax

    def is_twisted(self):
        return self._twisted
   
    def get_q_register(self):
        return self._q_register
        
    def get_a_register(self):
        return self._a_register
    
    def dx(self):
        return self._dx
    
    def dp(self):
        return 2 * math.pi / self._dx / self._nL
    
    def x_max(self):
        if self._nL % 2 == 0:
            return self._dx * self._nL / 2
        else:
            return self._dx * (self._nL - 1) / 2

    def p_max(self):
        if self._nL % 2 == 0:
            return self.dp() * self._nL / 2
        else:
            return self.dp() * (self._nL - 1) / 2

    def x_list(self): # Edited
        return np.arange(0,self._nL)*self._dx - self.x_max()
 
    def p_list(self):
        return self.dp() * (np.arange(0,self._nL) - self.p_max() / self.dp() - self._theta / 2 / math.pi)
    
################################################################################################
    #def phi_list(self): #SFT
        #return np.linspace(-self._phiMax, self._phiMax, self._nPhi)

    #def pi_list(self): #SFT
        #piMax = np.pi/(2*self._phiMax) * (self._nPhi -1)**2/self._nPhi
        #return np.linspace(-piMax, piMax, self._nPhi)
################################################

    #def omega_list(self): #SFT
        #return np.abs(2. / self._dx * np.sin(self.p_list() * self._dx / 2.))
    
    #def ground_state_energy(self): #SFT
        #return np.sum(self.omega_list()) / 2.

    #def Gij_matrix(self): #SFT
        #'''
        #:return: (2D Array) Inverse of the correlation matrix, that gives the ground state of a 1D scalar field theory,
        #                        see https://arxiv.org/pdf/2102.05044.pdf Eq. S54.
        #'''
        #omegalist = self.omega_list()
        #correlation = np.zeros((self._nL, self._nL))
        #for i in range(self._nL):
        #    for j in range(self._nL):
        #        explist = np.exp(1.j * self.p_list() * (self.x_list()[i] - self.x_list()[j]))
        #        # Should be times dx in the equation below!
        #        correlation[i,j] = np.real(np.sum(np.abs(omegalist) * explist) * self._dx / self._nL)
        #return correlation

    #def KitaevWebbDMDecomposition(self): #SFT
    #    '''
    #    :return: (Tuple(2D array, 2D array)) D and M from MDM decomposition, see https://arxiv.org/pdf/0801.0342.pdf.
    #    '''
    #    g = np.linalg.cholesky(self.Gij_matrix())
    #    d = np.real(np.diag(np.diag(g)**2))
    #    l = np.real(g @ np.diag(np.diag(g)**-1))
    #    m = np.linalg.inv(l.conj().T)
    #    return d, m

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
    
    #def apply_double_phi(self, pos1, pos2, fact): #SFT
    #    """
    #    Applies the exponential of Exp[-i fact phi_pos1 phi_pos2]
    #    :param pos1: Lattice position of first phi
    #    :param pos2: Lattice position of second phi
    #    :param fact: The pre-factor 
    #    :return: The quantum circuit
    #    """
        #This uses the fact that the phi operator can be written as
        #phi^(n) ~ pre * Sum_i 2^i sigma^(n)_i
        #phi^(n) phi^(m) ~ pre^2 * Sum_{i>j} 2^{i+j+1} sigma^(n)_i sigma^(n)_j
        #with pre = phi_max / (2^nQ - 1)
    #    qc = QuantumCircuit(self._q_register)
    #    #Deal with the wrapping and boundary condition
    #    wrap = 0
    #    point1 = [0] * self._dimension
    #    point2 = [0] * self._dimension
    #    for i in range(self._dimension):
    #        (tmpwrap,point1[i]) = divmod(pos1[i], self._nL)
    #        wrap += tmpwrap
    #        (tmpwrap,point2[i]) = divmod(pos2[i], self._nL)
    #        wrap += tmpwrap

    #    if self._twisted:
    #        fact *= (-1)**wrap
            
    #    qbit_map = self._subregister_map(point1) + self._subregister_map(point2)
    #    qreg = QuantumRegister(2*self._nQ)
    #    prefact = fact * (self._phiMax / (2**self._nQ - 1))**2 

    #    for i in range(self._nQ):
    #        for j in range(self._nQ):
    #            fact = prefact * 2**(i+j)
    #            j += self._nQ #Different lattice site
    #            qc_sub = basic_circuits.exp_pauli_product(qreg, fact, [['Z',i], ['Z',j]]) #minus sign because we are doing exp[-I t]
    #            qc.compose(qc_sub, qbit_map, inplace=True)
    #    return qc                    

    def apply_single_prep(self, pos_op): #Generic
        """
        Applies a particular state preparation to a particular site
        :param pos_op: Position and operators to apply
            Is in the follwing form
                The lattice point (list of length dimension)
                The state to prepare, passed as an array of the right size
        :return: The quantum circuit
        """
        #################
        assert False, 'This method is untested.'
        #################
        assert(len(pos_op)==2)
        position = pos_op[0]
        operator = pos_op[1]
        point = [0] * self._dimension
        #Take the position modulo the number of lattice sites, and keep track of the wrap
        wrap = 0
        for i in range(self._dimension):
            point[i] = position[i] % self._nL
        
        [qbit_register, qbit_map] = self._get_subregister(point)
        qc_sub = QuantumCircuit(qbit_register)
        qc_sub.initialize(operator,qbit_register)
        qc = QuantumCircuit(self._q_register)
        qc.compose(qc_sub, qbit_map, inplace=True)
        return qc    
    
####################################################################
#################### Classes to create Circuits ####################
####################################################################
#from qiskit.aqua.utils.circuit_factory import CircuitFactory
"""
class ground_state(): #Specific
    #
    #Class hanlding the preparation of the ground state for the given scalar lattice.
    #

    def __init__(self, lattice, full_correlation=False, shear= True):
        '''
        :lattice:          (Lattice) Lattice on which the ground state is prepared
        :full_correlation: (bool)    If True, prepare a fully correlated state (exponential cost)
                                     If False, prepare a product state of uncorrelated 1D Gaussians,
                                        with widths determined by MDM decomposition
        :shear:            (bool)    If True, apply the shearing operation to a product of uncorrelated 1D Gaussians.
                                        Note: If full_correlation == True, then self._shear is set to False by default.
        '''
        self._lattice = lattice
        self._shear = shear
        self._full_correlation = full_correlation
        if full_correlation: self._shear = False # Override shear if preparing fully correlated state

        # Compute optimal number of arithmetic qubits for shearing
        if lattice._nL <= 2: # Never the case in practice, but this is a precondition that ensures r >= nQ.
            r= lattice._nQ
        else:
            r= int(lattice._nQ - 1 + np.ceil(np.log2(lattice._nL - 1)))

        self._shear_ancillas= (2 * r) +1


    def set_full_correlation(self, full_correlation):
        self._full_correlation = full_correlation
        
    def build(self, qc, q, q_ancillas=None, params=None):

        Gij= self._lattice.Gij_matrix() # inverse of the correlation matrix
        correlation= inv(Gij)
        D, M= self._lattice.KitaevWebbDMDecomposition()

        #print(D, M, g)
        #print(D1, M1, g1)

        #print('Shear? ' + str(self._shear))

        if self._full_correlation: # i.e. Qiskit default prep
            # Set up bounds list
            bound= (-self._lattice._phiMax, self._lattice._phiMax) # This seems bad, just have phi_max as a parameter
            bounds= []
            for j in range(self._lattice._nL):
                bounds.append(bound)

            # Prepare ground state
            ground_state = NormalDistribution(
                            num_qubits=[self._lattice._nQ]*self._lattice._nL,
                            mu=[0]*self._lattice._nL,
                            #sigma=correlation,
                            sigma= correlation / 2.,
                            bounds= bounds)

            # Apply ground state prep circuit
            qc.compose(ground_state, q[:], inplace=True)


        else: # i.e. Prepare 1D Gaussians using Qiskit, then shear

            covariance_diag = np.diagonal(D)

            # Prepare 1D Guassians
            ground_state = QuantumCircuit(q)   
            for i in range(self._lattice._nL):
                [qbit_register, qbit_map] = self._lattice._get_subregister([i])
                qc_sub = QuantumCircuit(qbit_register)
                qc_sub.compose(NormalDistribution(self._lattice._nQ,
                                                 mu=0.,
                                                 sigma=1/covariance_diag[i]/2,
                                                 #sigma=1/covariance_diag[i],
                                                 bounds= (-self._lattice._phiMax, self._lattice._phiMax)),
                               qbit_register[:], inplace=True)
                if self._shear == True:
                    qc_sub.x(qbit_register[-1]) # Put into two's complement for shearing
                ground_state.compose(qc_sub, qbit_map, inplace=True)

            # Apply 1D Gaussian prep circuits
            qc.compose(ground_state, q[:], inplace=True)

            # Schmear
            if self._shear == True:
                
                qc.compose(construct_shear_circuit_theoretical(self._lattice._nL, 
                                                               self._lattice._nQ, 
                                                               M,
                                                               dim= self._lattice._dimension, 
                                                               dx= self._lattice._dx), 
                            q_ancillas[:] + q[:], inplace=True)

                # Switch back to regular indexing (Not two's complement)
                for i in range(self._lattice._nL):
                    [qbit_register, qbit_map] = self._lattice._get_subregister([i])
                    qc.x(q[qbit_map[-1]])
"""



class evolution():
    """
    Class hanlding the time evolution for the given scalar lattice
    """

    def __init__(self, lattice, evolve_time=1., trotter_steps=1.):
        '''
        :lattice:       (Lattice) Lattice on which the time evolution operates
        :evolve_time:   (float)   amount of time to evolve for
        :trotter_steps: (int)     number of trotter steps
        '''
        self._lattice = lattice
        self._evolve_time = evolve_time
        self._trotter_steps = trotter_steps
        
    def set_evolve_time(self, evolve_time):
        self._evolve_time = evolve_time
        
    def set_trotter_steps(self, trotter_steps):
        self._trotter_steps = trotter_steps
                
    def build(self, qc, q, q_ancillas=None, params=None):
        #Create the circuits involving pi and phi fields
        evolve_H_Pi = QuantumCircuit(self._lattice.get_q_register())
        evolve_H_Phi = QuantumCircuit(self._lattice.get_q_register())
        phi2 = basic_op_cf.Phi2Operator()
        pi2 = basic_op_cf.Pi2Operator()

        if self._trotter_steps == 0:
            t= 0
        else: 
            t = self._evolve_time / self._trotter_steps
        
        #print('\n Evolve Time: ' + str(self._evolve_time))
        #print('Trotter Steps: ' + str(self._trotter_steps) + '\n')
        for i in range(self._lattice._nL): #Assumes a 1d Lattice
            evolve_H_Pi.compose(self._lattice.apply_single_operator([[i], pi2, [t * 0.5 * self._lattice._dx]]), inplace=True)
            evolve_H_Phi.compose(self._lattice.apply_single_operator([[i], phi2, [t * 1. / self._lattice._dx]]), inplace=True)
            evolve_H_Phi.compose(self._lattice.apply_double_phi([i-1], [i], t * (-1.) / self._lattice._dx), inplace=True)
            #evolve_H_Phi.compose(self._lattice.apply_double_phi([i+1], [i], t * (-.5) / self._lattice._dx), inplace=True)
            #evolve_H_Phi.compose(self._lattice.apply_double_phi([i-1], [i], t * (-.5) / self._lattice._dx), inplace=True)

        #Combine them using Suzuki-Trotter
        evolve_Trotter = QuantumCircuit(self._lattice.get_q_register())        
        for i in range(self._trotter_steps):
            evolve_Trotter.compose(evolve_H_Phi + evolve_H_Pi , inplace=True)


        qc.compose(evolve_Trotter , inplace=True)

    def build_connection(self, qc_evo, qc, qreg, areg, params=None):
        

        a_length = areg.size
        print(a_length)
        controlled_gate = qc_evo.to_gate().control(a_length)
        q_length = qreg.size
        qbit_list = [i for i in range(q_length)]
        qbit_list.insert(0, q_length)
        qc.append(controlled_gate, qbit_list)
        #print(qc.draw())
        
    
        


class wilson_line():
    """
    Class hanlding the creationg of the Wilson line operators
    """

    def __init__(self, lattice, dir1= [1], dir2= [-1], g= 1, trotter_steps_per_dt= 1):
        '''
        :lattice:              (Lattice)   Lattice on which this Wilson line operator acts
        :dir1:                 (list(int)) vector of size dimension giving the **positive** direction of the Wilson line
        :dir2:                 (list(int)) vector of size dimension giving the **negative** direction of the Wilson line
        :g:                    (float)     value of the coupling constant
        :trotter_steps_per_dt: (int)       number of Trotter steps used for one time step dt = dx

        '''
        self._lattice = lattice
        self._direction1= dir1
        self._direction2= dir2
        self._g= g
        self._trotter_steps_per_dt= trotter_steps_per_dt
        
    def build(self, qc, q, q_ancillas=None, params=None):
        #direction1 = params[0]
        #direction2 = params[1]
        #g = params[2]
        #trotter_steps_per_dt = params[3]

        #Create the circuit for the time evolution over a single time step
        evolve = evolution(self._lattice, evolve_time= -self._lattice._dx, trotter_steps= self._trotter_steps_per_dt)
        qc_evolution_step = QuantumCircuit(self._lattice.get_q_register())
        evolve.build(qc_evolution_step, self._lattice.get_q_register())
        #print("evolve params: time= %f, steps= %d" %(self._lattice._dx, self._trotter_steps_per_dt))
        point = [0] * self._lattice._dimension
        phi = basic_op_cf.PhiOperator()
        #Find the center of the lattice to puth the cusp of the lattice
        
        #center = self._lattice._nL % 2
        center = int((self._lattice._nL - 1) / 2)
        #Create a circuit for a single time evolution step with time dt
        
        qc_result = QuantumCircuit(self._lattice.get_q_register())
        pos1 = [center] * self._lattice._dimension
        pos2 = [center] * self._lattice._dimension
        tot_steps = 0
        while True:
            tot_steps += 1
            #Get the position of the fields for the Wilson lines
            pos1 = list(map(add, pos1, self._direction1)) 
            pos2 = list(map(add, pos2, self._direction2)) 

            #print(pos1, pos2)
            #Add a single evolution step
            qc_result.compose(qc_evolution_step.to_gate(label='Evolve'), inplace=True)
            #print('Forward time= ' + str(self._lattice._dx))

            #Add the exponential of the field operators
            # Note that apply_single_operator_list applies exp(op) with the opposite sign given
            qc_result.compose(self._lattice.apply_single_operator_list([[pos1, phi, [-self._g * self._lattice._dx]],
                                                                        [pos2, phi, [self._g * self._lattice._dx]]]), inplace=True)

            #Exit out of while loop if we have reached the end of the lattice
            for dim in range(self._lattice._dimension):
                finish = False
                if pos1[dim] == 0 or pos1[dim] == self._lattice._nL - 1:
                    finish = True
                if pos2[dim] == 0 or pos2[dim] == self._lattice._dimension - 1:
                    finish = True
            if finish:
                break
        #Add the full time evolution
        qc_devolution_step = QuantumCircuit(self._lattice.get_q_register())
        #
        #evolve.set_evolve_time(-1 * tot_steps * self._lattice._dx)
        evolve.set_evolve_time(1 * tot_steps * self._lattice._dx)
        #print(evolve._evolve_time)
        #
        evolve.set_trotter_steps(tot_steps * self._trotter_steps_per_dt)
        evolve.build(qc_devolution_step, self._lattice.get_q_register())
        
        #
        #evolve.build(qc_result, self._lattice.get_q_register())
        qc_result.compose(qc_devolution_step.to_gate(label='inv(Evolve)'), qc_result.qubits, inplace=True)
        #
        #print('Circuit total steps: %d' %(tot_steps * self._trotter_steps_per_dt))
        #print('Circuit total time: %d' %(tot_steps * self._lattice._dx))
        #print('Backward time= ' + str(-1 * tot_steps * self._lattice._dx))
        
        qc.compose(qc_result, inplace=True)