# General Imports
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg
from itertools import product
import gmpy2

# Local Imports
import sys
#sys.path.append('../arithmetic/')
sys.path.append('arithmetic_pkg')
import testing_utilities as tu

########################################################################################################################################
# Lattice class
########################################################################################################################################

class Lattice:
    """ Lattice class. Holds relevant information about the qubit lattice
        we want to use. Certain helper quantites are initialized to useful
        default values but can be reset if desired.
        
        In particular, field digitization is set for optimum performance
        assuming a free scalar field theory, which
    """
    
    def __init__(self, nL, nQ, dx=1, twist=0):
        """ Construct a physical lattice with nL sites of nQ qubits each.
            Lattice spacing and twist (in units of pi) can be provided.
        """
        self.dx, self.twist = dx, twist
        self.nL, self.nQ = nL, nQ
        
        # self.phiMax = np.sqrt(1/self.AvgEModes() * np.pi/2 * (self.nPhi() - 1)**2/self.nPhi())
        # Not needed since nQ setter now handles this every time nQ is changed
    
    @property
    def nQ(self):
        return self._nQ
    
    @nQ.setter
    def nQ(self, value):
        self._nQ = value
        self.phiMax = np.sqrt(1/self.AvgEModes() * np.pi/2 * (self.nPhi() - 1)**2/self.nPhi())
    
    def __repr__(self):
        ''' Print out constructor for state. '''
        return 'Lattice(nL={}, nQ={}, dx={}, twist={})'.format(self.nL, self.nQ, self.dx, self.twist)
        
    def __str__(self):
        """ Print out a basic description of the lattice. """
        return "{}-site lattice with {} qubits/site. \nlattice spacing: a = {}, boundary phase shift: theta = {:.2f}".format(nL, nQ, dx, self.theta())

    def theta(self):
        return np.pi*self.twist

    def nPhi(self):
        return 2**self.nQ
    
    def EModes(self):
        """ A helper function to compute the mode energies assuming a free
            field theory. Used to initialize the field spacing for the digitized
            field representation. """
        return 2 * np.sqrt(np.sin(self.dx * self.pLattice()/2)**2)
    
    def AvgEModes(self):
        """ A helper function to compute average mode energy assuming a free
            theory. Used to initialize the field spacing for the digitized
            field representation.
        """
        return np.average( self.EModes() )
    
    def xLattice(self, offset=0):
        """ Return physical lattice.
            First site is at 0 by default, but can be adjusted by changing offset.
            (This has no physical effect, but might be useful for plot formatting.)
        """
        return np.linspace(0 + offset, self.dx*self.nL + offset, self.nL, endpoint=False, dtype=float)
    
    def pLattice(self):
        """ Return momentum (reciprocal) lattice. 
            The momentum lattice is determined by the choice of twist variable,
            which is useful to avoid dealing with the p = 0 mode, but whose
            physical effect should vanish at nL increases.
        """
        dp = 2*np.pi/(self.nL*self.dx)
        p_lat = np.linspace(dp*self.twist/2, dp*(self.nL+self.twist/2), self.nL, endpoint=False)
#        p_max = np.pi/self.dx if self.nL%2 == 0 else np.pi/self.dx * (self.nL-1)/self.nL # Technically only maximum momentum if twist = 0
#        p_lat = np.linspace(-p_max - self.theta()/(self.dx*self.nL),
#                            -p_max + 2*np.pi/self.dx - self.theta()/(self.dx*self.nL), self.nL, endpoint=False)
        return p_lat
    
    def phiLattice(self):
        """ Return field value lattice. """
        return np.linspace(-self.phiMax, self.phiMax, self.nPhi())
    
    def piLattice(self):
        """ Return conjugate field (field momentum) lattice. """
        piMax = np.pi/(2*self.phiMax) * (self.nPhi() -1)**2/self.nPhi()
        return np.linspace(-piMax, piMax, self.nPhi())

########################################################################################################################################
# Fourier Transforms
########################################################################################################################################

def DFT_phi(N):
    i, j = np.meshgrid(np.linspace(-(N-1)/2, (N-1)/2, N), np.linspace(-(N-1)/2, (N-1)/2, N))
    omega = np.exp( 2 * np.pi * 1j / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W

def iDFT_phi(N):
    W = DFT_phi(N).conj().T
    return W

########################################################################################################################################
# Single site operators
########################################################################################################################################

def idOp(lattice):
    return np.identity(lattice.nPhi())

def phiOp(lattice):
    return np.diag(lattice.phiLattice())

def phi2Op(lattice):
    return np.diag(lattice.phiLattice()**2)

def phi4Op(lattice):
    return np.diag(lattice.phiLattice()**4)

def ipiOp(lattice, tol=13):
    ipi_diagOp = np.diag(-1j*lattice.piLattice())
    return (iDFT_phi(lattice.nPhi()) @ ipi_diagOp @ DFT_phi(lattice.nPhi())).round(tol)

def ipidagOp(lattice, tol=13):
    ipidag_diagOp = np.diag(1j*lattice.piLattice())
    return (iDFT_phi(lattice.nPhi()) @ ipidag_diagOp @ DFT_phi(lattice.nPhi())).round(tol)

def pi2Op(lattice, tol=13):
    pi2_diagOp = np.diag(lattice.piLattice()**2)
    return (iDFT_phi(lattice.nPhi()) @ pi2_diagOp @ DFT_phi(lattice.nPhi())).round(tol)

########################################################################################################################################
# SHO
########################################################################################################################################

def SHOOp(lattice):
    maxE = lattice.EModes()[math.floor(lattice.nL/2)]
    return 0.5*(pi2Op(lattice) + maxE**2*phi2Op(lattice))

########################################################################################################################################
# Multi-site operators via Kronecker products
########################################################################################################################################

def operatorSite(opTup, posTup, lattice, swapsites=False):
    """ Takes a tuple of operators and positions, and returns the Kronecker
        product of these on the specified lattice. Applies appropriate twist
        if operator position is outside the first Brouillin zone (0, nL-1).
        
        op is a tuple of functions of the form op(lattice) that are assumed to
        have already been defined. If this is not the case, the function will
        fail unexpectedly.
        
        pos is a tuple of positions, none of which can appear twice when
        projected to the first Bruillin zone, otherwise an exception is raised.
        
        !!! Has a known bug if the operator is itself a function of other
            operators, e.g., a Wilson line. The operators inside the function
            don't pick up required phases when moved back to the first BZ.
    """
    posList, phaseList = [], []
    for pos in posTup:
        wrapnum, posnum = divmod(pos, lattice.nL)
        phase = np.exp(1j*wrapnum*lattice.theta()).round(13)
        posList.append(posnum)
        phaseList.append(phase)
        
    if max([posList.count(x) for x in set(posList)]) > 1:
        raise Exception("More than one operator per lattice site!")
    
    opList = [idOp(lattice)] * lattice.nL
    #print('Before: ' + str(opList))
    for op, pos, phase in zip(opTup, posList, phaseList):
        opList[pos] = phase*op
    #print('After: ' + str(opList) + '\n')
    fullOp = np.array([1])
    for op in opList:
        if swapsites: fullOp = np.kron(op, fullOp) # Reverse the order of the Kronecker product, i.e. site 0 is the rightmost qubits
        else: fullOp = np.kron(fullOp, op)
        
    return fullOp

########################################################################################################################################
# Hamiltonian
########################################################################################################################################

def HamiltonianPi(lattice):
    return lattice.dx * 0.5 * np.sum([operatorSite([pi2Op(lattice)], [i], lattice) for i in range(lattice.nL)], axis=0)

def HamiltonianPhi_inefficient(lattice, lam=0):
    kinterm = -1./lattice.dx**2 * np.sum([operatorSite([phiOp(lattice),phiOp(lattice)],[i,i-1],lattice)
                                         + operatorSite([phiOp(lattice),phiOp(lattice)],[i,i+1],lattice)
                                         - 2*operatorSite([phi2Op(lattice)],[i],lattice) for i in range(lattice.nL)], axis=0)
    intterm = 0. if lam==0 else lam * np.sum([operatorSite([phi4Op(lattice)], [i], lattice) for i in range(lattice.nL)], axis=0)
    return lattice.dx * (0.5 * kinterm + (1/math.factorial(4)) * intterm)


def HamiltonianPhi(lattice, lam=0):
    kinterm = 2./lattice.dx**2 *  ( np.sum([operatorSite([phi2Op(lattice)],[i],lattice)
                                            - operatorSite([phiOp(lattice),phiOp(lattice)],[i,i-1],lattice)
                                            for i in range(1, lattice.nL)], axis=0)
                                   + operatorSite([phi2Op(lattice)],[0],lattice)
                                   + operatorSite([phiOp(lattice),phiOp(lattice)],[0,lattice.nL-1],lattice)
                                   )
    intterm = 0. if lam==0 else lam * np.sum([operatorSite([phi4Op(lattice)], [i], lattice) for i in range(lattice.nL)], axis=0)
    return lattice.dx * (0.5 * kinterm + (1/math.factorial(4)) * intterm)

def Hamiltonian(lattice, lam=0):
    ''' Create Hamiltonian for a massless scalar field theory.
        Optional quartic coupling can be set'''
    return HamiltonianPi(lattice) + HamiltonianPhi(lattice, lam)

########################################################################################################################################
# Time evolution
########################################################################################################################################

def expHamiltonianPi(t, lattice):
    return scipy.linalg.expm(-1.j * t * HamiltonianPi(lattice))

def expHamiltonianPhi(t, lattice, lam=0):
    return scipy.linalg.expm(-1.j * t * HamiltonianPhi(lattice, lam))

def evolveH(t, lattice, lam=0):
    return scipy.linalg.expm(-1.j * t * Hamiltonian(lattice, lam))

def evolveHTrotter(t, n, lattice, lam=0):
    if n == 0:
        return np.identity(2**(lattice.nL * lattice.nQ))
    else:
        return np.linalg.matrix_power(expHamiltonianPi(t/n, lattice) @ expHamiltonianPhi(t/n, lattice, lam),n)

########################################################################################################################################
# State preparation
########################################################################################################################################

def GPhi(i, j, lattice, tol=13):
    x = lattice.xLattice()
    modeweight = lattice.EModes() * np.exp(1j*lattice.pLattice()*(x[i] - x[j]))
    return (1/lattice.nL)*np.sum(modeweight).round(tol)

def GPhiMatrix(lattice, tol=13):
    return [[GPhi(i,j, lattice) for j in range(lattice.nL)] for i in range(lattice.nL)]

def KitaevWebbDMDecomposition(input_mat):
    #https://arxiv.org/abs/0801.0342
    #g = np.linalg.cholesky(input_mat)
    #d = np.diag(np.diag(g)**2)
    #m = np.linalg.inv(np.diag(np.diag(g)**-1) @ g.T)
    #return d, m, g
    g = np.linalg.cholesky(input_mat)
    d = np.real(np.diag(np.diag(g)**2))
    l = np.real(g @ np.diag(np.diag(g)**-1))
    m = np.linalg.inv(l.conj().T)
    return d,m


def toPhiList(pos, lattice):
    # This one doesn't use gmpy2, which can only handle bases up to 62.
    philist= np.zeros(lattice.nL)
    nPhi= lattice.nPhi()

    j= 0
    nBase= []
    while pos > 0:
        nRem= (pos % nPhi**(j+1)) / nPhi**j
        nBase.append(nRem)
        pos-= nRem * nPhi**j
        j+= 1

    while len(nBase) < lattice.nL:
        nBase.append(0)
    
    for i in range(lattice.nL):
        philist[i]= lattice.phiLattice()[int(nBase[lattice.nL-i-1])]

    philist= np.flip(philist, 0)  # NEW
    return philist


def toPos(philist, lattice):
    # Inverse function of toPhiList
    
    philist= np.flip(philist, 0)                    # NEW
    
    i= 0
    for j in range(lattice.nL):
        index= np.where(abs(lattice.phiLattice() - philist[j]) < 1e-14)
        i+= lattice.nPhi()**(lattice.nL-j-1) * index[0][0]
    return i


def createEigenstate(exlist, lattice):
    ''' Creates the normalized wavefunction for a digitized approximation
        of an energy eigenstate for a 1D lattice. 
          exlist: occupation numbers for momentum modes in the order given
                  by lattice.EModes(). [0, ..., 0] is the ground state
    '''
    if len(exlist) != lattice.nL:
        raise TypeError('Mode list incompatible with lattice size!')
    wavefun = np.zeros(lattice.nPhi()**lattice.nL, dtype=complex)
    x = lattice.xLattice()
    p = lattice.pLattice()
    phicov = GPhiMatrix(lattice)

    for i in range(lattice.nPhi()**lattice.nL):
        philist = toPhiList(i, lattice)
        expfactor = np.exp(-0.5 * philist @ phicov @ philist)
        #expfactor = np.exp(-0.25 * philist @ phicov @ philist)

        terms = np.zeros(lattice.nL, dtype=complex)
        for n, nk in zip(range(len(exlist)), exlist):
            Hcoeff = np.zeros(nk+1)
            Hcoeff[nk] = 1
            field = np.sqrt(lattice.EModes()[n]) * np.sqrt(1/lattice.nL)*np.sum([philist[i]*np.exp(-1j * p[n] * x[i]) for i in range(lattice.nL)])
            term = np.polynomial.hermite.hermval([field], Hcoeff).round(13)
            terms[n] = term
        wavefun[i] = expfactor * np.prod(terms)
    wavefun = wavefun/np.sqrt(np.sum(np.abs(wavefun)**2))
    return wavefun


def createKWstate(exlist, lattice):
    ''' Creates the normalized wavefunction for the approximation
        of an energy eigenstate for a 1D lattice using the correlation
        of the Kitaev-Webb procedure.
        NB: Currently does not actually construct the KW state, since
        the shear matrix is not applied.
          exlist: occupation numbers for momentum modes in the order given
                  by lattice.EModes(). [0, ..., 0] is the ground state
    '''
    if len(exlist) != lattice.nL:
        raise TypeError('Mode list incompatible with lattice size!')
    wavefun = np.zeros(lattice.nPhi()**lattice.nL, dtype=complex)
    x = lattice.xLattice()
    p = lattice.pLattice()
    phicov = KitaevWebbDMDecomposition(GPhiMatrix(lattice))[0]

    for i in range(lattice.nPhi()**lattice.nL):
        philist = toPhiList(i, lattice)

        #Compute rotated field values after rounding to nearest digitized value
        #dphi = 2*lattice.phiMax/(lattice.nPhi()-1)
        #philist = philist/dphi

        expfactor = np.exp(-0.5 * philist @ phicov @ philist)
        #expfactor = np.exp(-0.25 * philist @ phicov @ philist)

        terms = np.zeros(lattice.nL, dtype=complex)
        for n, nk in zip(range(len(exlist)), exlist):
            Hcoeff = np.zeros(nk+1)
            Hcoeff[nk] = 1
            field = np.sqrt(lattice.EModes()[n]) * np.sqrt(1/lattice.nL)*np.sum([philist[i]*np.exp(-1j * p[n] * x[i]) for i in range(lattice.nL)])
            term = np.polynomial.hermite.hermval([field], Hcoeff).round(13)
            terms[n] = term
        wavefun[i] = expfactor * np.prod(terms)

    wavefun = wavefun/np.sqrt(np.sum(np.abs(wavefun)**2))
    return wavefun


def createKWground(lattice):
    ''' Creates the KW approximationg of the scalar ground state for a given
        lattice, including proper treatment of shear matrices, as implemented in the circuit
                     (shear.py)
    '''
    dphi = 2*lattice.phiMax/(lattice.nPhi()-1)
    ground = np.zeros(lattice.nPhi()**lattice.nL, dtype=complex)

    dcov,mshear = KitaevWebbDMDecomposition(GPhiMatrix(lattice))
    invmshear = np.linalg.inv(mshear)
    
    # Scale D
    dcov*= dphi**2

    for i in range(lattice.nPhi()**lattice.nL):
        philist = toPhiList(i, lattice)

        # Scale to half integer values
        philist = philist / dphi

        nL= lattice.nL
        nQ= lattice.nQ

        if nL <= 2: # Never the case in practice, but this is a precondition that ensures r >= k.
            r= nQ
        else:
            r= int(nQ - 1 + np.ceil(np.log2(nL-1)))

        philistSheared= np.array(philist)
        for j in range(nL):
            for k in range(nL-j-1):

                Mij_bin= tu.decimal_to_binary2(mshear[j][k+j+1], nQ + r, r)
                Mij= tu.binary_to_decimal(Mij_bin, r)
                nj_bin= tu.decimal_to_binary2(philist[k+j+1], nQ + r, r)
                nj= tu.binary_to_decimal(nj_bin, r)

                # Have to subtract the part that's not computed on the circuit --
                # This is the 1/2 place of nj times the least significant bit of Mij

                if Mij_bin[-1] == 1:
                    philistSheared[j]-= 2**(-r-1)

                Mij_nj_bin= tu.decimal_to_binary(Mij * nj, nQ + r, r)
                Mij_nj= tu.binary_to_decimal(Mij_nj_bin, r)

                philistSheared[j]+= Mij_nj
        

        # Correct for machine errors...
        for j in range(lattice.nL):
            if 1 - (philistSheared[j] % 1) < 1e-14:
                philistSheared[j]= np.ceil(philistSheared[j])

        # Rounding
        philistSheared= np.floor(philistSheared.astype(float))
        philistSheared+= 0.5

        # Wrap around
        philistSheared = np.mod(philistSheared.astype(float), lattice.nPhi())
        philistSheared[np.where(philistSheared >= lattice.nPhi()/2)]-= lattice.nPhi() # where over the largest positive

        #expfactor = np.exp(-0.25 * philist @ dcov @ philist)
        expfactor = np.exp(-0.5 * philist @ dcov @ philist)
        
        philistSheared= philistSheared * dphi
        pos= toPos(philistSheared, lattice)
        amp= ground[pos]
        ground[pos] = np.sqrt(expfactor**2 + amp**2)
        
    ground = ground/np.sqrt(np.sum(np.abs(ground)**2))

    return ground


def createKWground_ext(lattice, ool_keep=True, decompose_shear=False, binary_approx=True, testmode= True, twosite=False, new=False):
    ''' Extended version of createKWground, with several utilities.
    
            ool_keep: keep out-of-digitization-lattice entries by wrapping
                      to other side of lattice (dropping these would require
                      gabage collection on the quantum circuit) [True]
            decompose_shear: original KW procedure where every individual shift
                             is digitized before application [False]
            binary_approx: approximate arithmetic, that's equivalent to the arithmetic implemented in 
                           the circuit from shear.construct_shear_circuit_theoretical
            testmode: if True, also returns a list of the sheared states, and prints helpful things.
            twosite: if True, constructs an arbitrary 2x2 shearing matrix, rather than a physical one
    '''
    dphi = 2*lattice.phiMax/(lattice.nPhi()-1)
    ground = np.zeros(lattice.nPhi()**lattice.nL, dtype=complex)
    
    if testmode:
        shearList= []

    if twosite:
        M01= 0.35
        corr= np.array([[1,M01],[M01,1]], dtype=float)
        invcorr= inv(corr)
        dcov,mshear,_ = KitaevWebbDMDecomposition3(invcorr)

    else:
        dcov,mshear = KitaevWebbDMDecomposition(GPhiMatrix(lattice))
        invmshear = np.linalg.inv(mshear)
    
    # Scale D
    dcov*= dphi**2
    
    if testmode:
        print(mshear)

    for i in range(lattice.nPhi()**lattice.nL):
        philist = toPhiList(i, lattice)

        # Scale to half integer values
        philist = philist / dphi
        
        if decompose_shear: # TODO
            for j in range(len(mshear)):
                shifts = np.rint((mshear[j] * philist)[j+1:])
                philist[j] += np.sum(shifts)
            philist = np.rint(philist + lattice.phiMax/dphi)

        else:
            if binary_approx == False:
                philistSheared= (mshear @ philist).astype(float)
            else:
                nL= lattice.nL
                nQ= lattice.nQ
                
                if nL <= 2: # Never the case in practice, but this is a precondition that ensures r >= k.
                    r= nQ
                else:
                    if new == True:
                        r= int(nQ - 1 + nL**2)
                    else:
                        r= int(nQ - 1 + np.ceil(np.log2(nL-1))) # THIS RIGHT HERE

                philistSheared= np.array(philist)
                for j in range(nL):
                    for k in range(nL-j-1):
                        
                        Mij_bin= tu.decimal_to_binary2(mshear[j][k+j+1], nQ + r, r)
                        Mij= tu.binary_to_decimal(Mij_bin, r)
                        nj_bin= tu.decimal_to_binary2(philist[k+j+1], nQ + r, r)
                        nj= tu.binary_to_decimal(nj_bin, r)
                        
                        # Have to subtract the part that's not computed on the circuit --
                        # This is the 1/2 place of nj times the least significant bit of Mij
                        
                        if Mij_bin[-1] == 1:
                            philistSheared[j]-= 2**(-r-1)
                        
                        Mij_nj_bin= tu.decimal_to_binary(Mij * nj, nQ + r, r)
                        Mij_nj= tu.binary_to_decimal(Mij_nj_bin, r)
                        
                        if testmode:
                            print(j, k)
                            print(str(Mij_bin) + ' --> ' + str(Mij))
                            print(str(nj_bin) + ' --> ' + str(nj))
                            print(str(Mij_nj_bin) + ' --> ' + str(Mij_nj) + '\n')

                        philistSheared[j]+= Mij_nj

            # Correct for machine errors...
            for j in range(lattice.nL):
                if 1 - (philistSheared[j] % 1) < 1e-14:
                    philistSheared[j]= np.ceil(philistSheared[j])

            # Rounding
            philistSheared= np.floor(philistSheared.astype(float))
            philistSheared+= 0.5

        #Two options for how to deal with out-of-lattice points
        if ool_keep:
            #Option A: wrap out-of-lattice point back to original lattce
            philistSheared = np.mod(philistSheared.astype(float), lattice.nPhi())
            philistSheared[np.where(philistSheared >= lattice.nPhi()/2)]-= lattice.nPhi() # where over the largest positive
            
            if testmode:
                shearList.append(philistSheared)
                print(philistSheared)

            #expfactor = np.exp(-0.25 * philist @ dcov @ philist)
            expfactor = np.exp(-0.5 * philist @ dcov @ philist)
            
        if not ool_keep:
            #Option B: just drop them (requires garbage collection in qcircuit)
            ool = any(id < -0.1 or id > lattice.nPhi() - 0.1 for id in philistSheared)
            philistSheared = philistSheared*dphi - lattice.phiMax
            #expfactor = np.exp(-0.25 * philist @ dcov @ philist) if not ool else 0.
            expfactor = np.exp(-0.5 * philist @ dcov @ philist) if not ool else 0.
        
        philistSheared= philistSheared * dphi
        pos= toPos(philistSheared, lattice)
        amp= ground[pos]
        ground[pos] = np.sqrt(expfactor**2 + amp**2)
        
    ground = ground/np.sqrt(np.sum(np.abs(ground)**2))

    if testmode:
        return ground, shearList
    else:
        return ground




########################################################################################################################################
# Ladder operators
########################################################################################################################################

def aop(k, lattice):
    ''' Lowering operator for mode k of given lattice. '''
    x = lattice.xLattice()
    p = lattice.pLattice()
    # Hack to fix reversed indexing behavior relative to the direct state creation code that I don't understand
    k = lattice.nL - 1 - k
    
    pre = np.sqrt(lattice.EModes()[k]/(2*lattice.nL))
    phipiece = np.sum([operatorSite([phiOp(lattice)], [i], Lat)*np.exp(-1j * p[k] * x[i]) for i in range(lattice.nL)], axis=0)
    pipiece = np.sum([operatorSite([ipiOp(lattice)], [i], Lat)*np.exp(-1j * p[k] * x[i]) for i in range(lattice.nL)], axis=0)
    return pre * (phipiece + lattice.dx/lattice.EModes()[k] * pipiece)

def adagop(k, lattice):
    ''' Raising operator of mode k of given lattice. '''
    x = lattice.xLattice()
    p = lattice.pLattice()
    # Hack to fix reversed indexing behavior relative to the direct state creation code that I don't understand
    k = lattice.nL - 1 - k    
    
    pre = np.sqrt(lattice.EModes()[k]/(2*lattice.nL))
    phipiece = np.sum([operatorSite([phiOp(lattice)], [i], Lat)*np.exp(-1j * p[k] * x[i]) for i in range(lattice.nL)], axis=0)
    pipiece = np.sum([operatorSite([ipidagOp(lattice)], [i], Lat)*np.exp(-1j * p[k] * x[i]) for i in range(lattice.nL)], axis=0)
    return pre * (phipiece + lattice.dx/lattice.EModes()[k] * pipiece)

########################################################################################################################################
# Plotting
########################################################################################################################################

def plot_persite(input_vals_list,nQ,nL,labs):
    f = plt.figure(figsize=(5,5))
    
    forplot = np.zeros([nL*2**nQ,len(input_vals_list)])
    for k in range(len(input_vals_list)):
        for i in range(nL):
            for j in range(2**(nQ*nL)):
                qval = int(gmpy2.digits(j,2**nQ).zfill(nL)[i])
                forplot[qval+i*2**nQ,k]+=input_vals_list[k][j]
                pass
            pass
        pass
    
    xlabs = []
    for i in range(nL):
        plt.text(2**nQ*i+2**nQ/2-0.5,-0.4,"site "+str(i),horizontalalignment='center')
        if (i > 0):
            plt.axvline(2**nQ*i-0.5,color='grey',ls=":")
        for j in range(2**nQ):
            xlabs += [r'$|'+bin(j)[2:].zfill(nQ)+r'\rangle$']
            pass
        pass
    
    for k in range(len(input_vals_list)):
        plt.plot(forplot[:,k],label=labs[k])
        pass
    plt.ylabel("Pr(site)")
    plt.legend()
    
    plt.xticks(range(nL*2**nQ),xlabs,rotation='vertical',fontsize=10,horizontalalignment='center')

########################################################################################################################################
# Wilson Lines operators
########################################################################################################################################

def WLop(g, lattice):
    return scipy.linalg.expm(1j * g * lattice.dx * phiOp(lattice))

def WLdagop(g, lattice):
    return scipy.linalg.expm(-1j * g * lattice.dx * phiOp(lattice))

def WilsonTrotter(g, n, lattice): # Evolve signs are reversed, compared to PaperPlotandCrossCheck.ipynb
    Uop = np.identity(lattice.nPhi()**lattice.nL)
    midp = math.floor(lattice.nL/2)
    for step in range(1,midp+1):
#        print("Now on step {}".format(step+1))
        stepU = operatorSite([WLop(g, lattice), WLdagop(g, lattice)], [midp+step, midp-step], lattice, swapsites= True) @ evolveHTrotter(-lattice.dx, n, lattice)
        Uop = stepU @ Uop

    Uop= evolveHTrotter(lattice.dx * midp, n * midp, lattice) @ Uop
    return Uop

def WilsonEvolve(g, lattice): # Evolve signs are reversed, compared to PaperPlotandCrossCheck.ipynb
    Uop = np.identity(lattice.nPhi()**lattice.nL)
    midp = math.floor(lattice.nL/2)
    for step in range(1,midp+1):
#        print("Now on step {}".format(step+1))
        stepU = operatorSite([WLop(g, lattice), WLdagop(g, lattice)], [midp+step, midp-step], lattice, swapsites= True) @ evolveH(-lattice.dx, lattice)
        Uop = stepU @ Uop

    Uop= evolveH(lattice.dx * midp, lattice) @ Uop
    return Uop

def WilsonNoEvolve(g, lattice): # Evolve signs are reversed, compared to PaperPlotandCrossCheck.ipynb
    Uop = np.identity(lattice.nPhi()**lattice.nL)
    midp = math.floor(lattice.nL/2)
    for step in range(1,midp+1):
#        print("Now on step {}".format(step+1))
        stepU = operatorSite([WLop(g, lattice), WLdagop(g, lattice)], [midp+step, midp-step], lattice, swapsites= True)
        Uop = stepU @ Uop
    return Uop

########################################################################################################################################
# Wilson Lines analytic comparison
########################################################################################################################################

def omega(klist, nL, delta = 0.5):
    ''' Frequency of kth mode (assumes twisted BCs at the moment). '''
    return 2*np.sqrt(np.sum([np.sin(np.pi*(k + delta)/nL)**2 for k in klist]))

def alpha(klist, nL, delta = 0.5):
    ''' Displacement operator shift of resulting coherent state
        after applying Wilson line along first axis.
    '''
    prefactor = 2/np.sqrt(nL) * 1/np.sqrt(2*omega(klist, nL, delta))
    sum = np.sum([np.exp(1j*(s+1)*omega(klist, nL, delta))*np.sin(2*np.pi*(s+1)*(klist[0] + delta)/nL) for s in range(math.floor(nL/2))])
    return prefactor * sum

def transrate(g, nL, exlist):
    ''' Transition rate from ground state to state given by exlist.
          g: coupling to Wilson line, in units of the lattice spacing
          nL: lattice size
          exlist: list of excitation numbers for each mode on the lattice.
                  [0, 0, ..., 0] correponds to the ground state and frequencies
                  increase toward the middle of the list. Throws an error if
                  length of list is incorrect for lattice.
                  !!! Currently only implemented for 1D lattices. !!!
    '''
    if len(exlist) != nL:
        raise TypeError('Mode list incompatible with lattice size!')
    prefactor = np.exp(-g**2*np.sum([np.abs(alpha([k],nL))**2 for k in range(nL)]))
    terms = [(1 if mk == 0 else (g**2*np.abs(alpha([m],nL))**2)**mk)/math.factorial(mk) for m, mk in zip(range(len(exlist)), exlist)]
    prod = np.prod(terms)
    return prefactor * prod

# Define wavefuctions for eigenstates here? e.g. wavefunc([phi_1, phi_2, ... phi_nL], [n_1, n_2, ..., n_nL]])