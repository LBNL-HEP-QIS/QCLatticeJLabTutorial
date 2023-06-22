import sys
sys.path.append('Scalar_Field_Theory')
from lattice import *






class latt_sft(Lattice):
    def __init__(self,lattice):
        self.lattice = lattice
        self._nL = lattice._nL
        self._nQ = lattice._nQ
        self._dx = lattice._dx
        self.p_list = lattice.p_list
        self.x_list = lattice.x_list
        self._nPhi= 2**lattice._nQ
        self._phiMax= np.sqrt(1/np.average(self.omega_list()) * np.pi/2 * (self._nPhi - 1)**2/self._nPhi)
        self._q_register = lattice._q_register
        #self._q_register = lattice._q_register
    #@Lattice.nQ.setter
    def nQ(self, nQ): #SCF
        self._nQ= nQ
        self._nPhi= 2**nQ
        self._phiMax= np.sqrt(1/np.average(self.omega_list()) * np.pi/2 * (self._nPhi - 1)**2/self._nPhi)
        settings.phi_max= self._phiMax
    def phi_list(self): #SFT
        return np.linspace(-self._phiMax, self._phiMax, self._nPhi)

    def pi_list(self): #SFT
        piMax = np.pi/(2*self._phiMax) * (self._nPhi -1)**2/self._nPhi
        return np.linspace(-piMax, piMax, self._nPhi)

    def omega_list(self): #SFT
        return np.abs(2. / self._dx * np.sin(self.p_list() * self._dx / 2.))
    
    def ground_state_energy(self): #SFT
        return np.sum(self.omega_list()) / 2.

    def Gij_matrix(self): #SFT
        '''
        :return: (2D Array) Inverse of the correlation matrix, that gives the ground state of a 1D scalar field theory,
                                see https://arxiv.org/pdf/2102.05044.pdf Eq. S54.
        '''
        omegalist = self.omega_list(self.lattice)
        correlation = np.zeros((self._nL, self._nL))
        for i in range(self._nL):
            for j in range(self._nL):
                explist = np.exp(1.j * self.p_list() * (self.x_list()[i] - self.x_list()[j]))
                # Should be times dx in the equation below!
                correlation[i,j] = np.real(np.sum(np.abs(omegalist) * explist) * self._dx / self._nL)
        return correlation

    def KitaevWebbDMDecomposition(self): #SFT
        '''
        :return: (Tuple(2D array, 2D array)) D and M from MDM decomposition, see https://arxiv.org/pdf/0801.0342.pdf.
        '''
        g = np.linalg.cholesky(self.Gij_matrix())
        d = np.real(np.diag(np.diag(g)**2))
        l = np.real(g @ np.diag(np.diag(g)**-1))
        m = np.linalg.inv(l.conj().T)
        return d, m
    def apply_double_phi(self, pos1, pos2, fact): #SFT
        """
        Applies the exponential of Exp[-i fact phi_pos1 phi_pos2]
        :param pos1: Lattice position of first phi
        :param pos2: Lattice position of second phi
        :param fact: The pre-factor 
        :return: The quantum circuit
        """
        #This uses the fact that the phi operator can be written as
        #phi^(n) ~ pre * Sum_i 2^i sigma^(n)_i
        #phi^(n) phi^(m) ~ pre^2 * Sum_{i>j} 2^{i+j+1} sigma^(n)_i sigma^(n)_j
        #with pre = phi_max / (2^nQ - 1)
        qc = QuantumCircuit(self._q_register)
        #Deal with the wrapping and boundary condition
        wrap = 0
        point1 = [0] * self._dimension
        point2 = [0] * self._dimension
        for i in range(self._dimension):
            (tmpwrap,point1[i]) = divmod(pos1[i], self._nL)
            wrap += tmpwrap
            (tmpwrap,point2[i]) = divmod(pos2[i], self._nL)
            wrap += tmpwrap

        if self._twisted:
            fact *= (-1)**wrap
            
        qbit_map = self._subregister_map(point1) + self._subregister_map(point2)
        qreg = QuantumRegister(2*self._nQ)
        prefact = fact * (self._phiMax / (2**self._nQ - 1))**2 

        for i in range(self._nQ):
            for j in range(self._nQ):
                fact = prefact * 2**(i+j)
                j += self._nQ #Different lattice site
                qc_sub = basic_circuits.exp_pauli_product(qreg, fact, [['Z',i], ['Z',j]]) #minus sign because we are doing exp[-I t]
                qc.compose(qc_sub, qbit_map, inplace=True)
        return qc      

class ground_state(latt_sft): #Specific
    """
    Class hanlding the preparation of the ground state for the given scalar lattice.
    """

    def __init__(self, lattice, full_correlation=False, shear= True):
        '''
        :lattice:          (Lattice) Lattice on which the ground state is prepared
        :full_correlation: (bool)    If True, prepare a fully correlated state (exponential cost)
                                     If False, prepare a product state of uncorrelated 1D Gaussians,
                                        with widths determined by MDM decomposition
        :shear:            (bool)    If True, apply the shearing operation to a product of uncorrelated 1D Gaussians.
                                        Note: If full_correlation == True, then self._shear is set to False by default.
        '''
        self._lattice =  lattice
        self._latt_sft = latt_sft
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

        Gij= self._latt_sft.Gij_matrix(latt_sft) # inverse of the correlation matrix
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


