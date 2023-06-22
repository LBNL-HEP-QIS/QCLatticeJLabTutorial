# Imports
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit import visualization
import qiskit


from lattice_qft.core import basic_circuits
# Global Variables:
_infty= 100


# General and Kitaev-Webb Functions

def f(sigma, mu, infty):
    '''
    Computes f(sigma, mu), as defined by Kitaev-Webb.

    :param sigma: (float) std dev
    :param mu:    (float) mean
    :param infty: (int)   summation limit, used in place of infinity
    :return:      (int)   f(sigma, mu)
    '''
    s= 0
    for j in range(infty):
        s+= np.exp(-(1/2)*((j+1-mu)/sigma)**2)
        s+= np.exp(-(1/2)*((-j-1-mu)/sigma)**2)
    s+= np.exp((-1/2)*((mu)/sigma)**2)
    return s


def gaussian_wavefunction(sigma, mu, x):
    '''
    Returns the square root of a normal distribution at x, where x is an integer or array of integers

    :param sigma:  (float)      std dev
    :param mu:     (float)      mean
    :param x:      (Array[int]) x-axis, an integer array
    :return:       (Array)      Gaussian(sigma, mu) at integer values
    '''
    #psiArray= np.exp(-1*(x-mu)**2 / (2*(sigma**2)))
    psiArray= np.exp(-1*(x-mu)**2 / (4*(sigma**2)))
    return psiArray / np.sqrt(f(sigma, mu, _infty))


def analytic_gaussian_wavefunction(sigma, mu, x):
    '''
    Returns the square root of a normal distribution with mean mu and std dev sigma at x
    
    :param sigma:  (float)        std dev
    :param mu:     (float)        mean
    :param x:      (Array[float]) x-axis
    :return:       (Array)        sqrtGaussian[sigma, mu](x)
    '''
    return np.exp(-1*(x-mu)**2 / (4*(sigma**2))) / np.sqrt(sigma*np.sqrt(2*np.pi))
    #return np.exp(-1*(x-mu)**2 / (2*(sigma**2))) / np.sqrt(sigma*np.sqrt(2*np.pi))


def xi(sigma, mu, N, i):
    '''
    Computes Xi_{sigma, mu, N}(i), as defined by K-W
    
    :param sigma: (float) std dev
    :param mu:    (float) mean
    :param N:     (int)   number of qubits
    :param i:     (int)   integer at which Xi is evaluated
    :return:      (float) Xi_{sigma, mu, N}(i), as defined in KW
    '''
    s= 0
    for j in range(_infty):
        s+= gaussian_wavefunction(sigma, mu, i+((j+1)*2**N))**2
        s+= gaussian_wavefunction(sigma, mu, i-((j+1)*2**N))**2
    s+= gaussian_wavefunction(sigma, mu, i)**2
    return np.sqrt(s)


def alpha(sigma, mu):
    '''
    Computes alpha(sigma, mu), as defined by KW
    
    :param sigma: (float) std dev
    :param mu:    (float) mean
    :return:      (float) alpha(sigma, mu)
    '''
    num= f(sigma/2, mu/2, _infty)
    den= f(sigma, mu, _infty)
    #print(num)
    #print(den)
    x= np.sqrt(f(sigma/2, mu/2, _infty)/f(sigma, mu, _infty))
    #print(x)
    if np.isnan(x):
        if num == 0 and den != 0:
            return 0
        else:
            return np.pi/2
    elif x > 1:
        return 0
    elif x < 0:
        return np.pi/2
    else:
        return np.arccos(x)


# Functions to compute sigmas, mus, and alphas

def generate_sigmas(sigma0, N):
    '''
    Generates a list of std devs, where the (n)th value corresponds to the
    std dev used in the (n)th iteration, which is defined as
    
    σ(n)= σ(n-1)/2
    
    :param sigma0: (float) std dev
    :param N:      (int)   number of iterations/target qubits
    :return:       (List)  List (length N) of std dev values
    '''
    sigmaArray= [sigma0]
    sigma= sigma0
    for j in range(N-1):
        sigmaArray.append(sigma/2)
        sigma= sigma/2
    
    return sigmaArray


def generate_mus(mu0, N):
    '''
    Generates an array (List of Lists) of means, where the (n)th entry of the outer list is a
    length 2^n list containing the possible means used in the (n)th iteration, which are
    
    μ(n)= μ(n-1)/2  OR  [μ(n-1)-1]/2

    :param mu0: (float)             mean
    :param N:   (int)               number of iterations/target qubits
    :return:    (List[List[float]]) Binary tree of mean values
    '''
    muArray= [[mu0]]
    for j in range(N-1):
        muArray.append([])
        for n in range(2**j):
            muPrev= muArray[j][n]
            muArray[j+1].append(muPrev / 2)
            muArray[j+1].append((muPrev - 1) / 2)
    
    return muArray


def generate_alphas(N, sigmaArray, muArray):
    '''
    Generates an array (List of Lists) of alphas, where the (n)th entry of the outer list is a
    length 2^n list containing the possible alphas used in the (n)th iteration,
    
    α(n)= α(σ(n), μ(n))
    
    At each iteration, two values of alpha are possible, thus the 2^n scaling.

    :param N:          (int)               number of iterations/target qubits
    :param sigmaArray: (list)              contains sigma values
    :param muArray:    (List[List[float]]) Binary tree of mean values
    :return:           (List[List[float]]) Binary tree of alpha values
    '''
    # Check that inputs are correct
    if N != len(sigmaArray):
        print('sigmaArray given does not have length N=%d. Exiting...' %(N))
        return
    if N != len(muArray):
        print('muArray given does not have length N=%d. Exiting...' %(N))
        return
    
    alphaArray= []
    for j in range(N): # O(2**N) total
        # Check muArray
        if len(muArray[j]) != 2**j:
            print('muArray[%d] given does not have length 2^%d=%d. Exiting...' %(j, j, 2**j))
            return
        alphaArray.append([])
        for n in muArray[j]:
            alphaArray[j].append(alpha(sigmaArray[j], n))
    return alphaArray


def decimal_to_binary(n, K):
    '''
    Returns the binary string representation 'b1b2...bK' to K bits of a decimal n.
    The leftmost bit is most significant, i.e. n= (1/2)^(b1) + ... + (1/2^K)^(bK)
    
    :param n: (float) a base 10 decimal between 0 and 1
    :param K: (int)   binary precision (number of bits)
    :return:  (str)   bit-string representation of n
    
    Note: if n is not between 0 and 1, only the decimal part of n is converted to binary.
    '''
    bitstring= ''
    for k in range(K):
        if n >= 1/(2**(k+1)):
            bitstring+= '1'
            n-= 1/(2**(k+1))
        else:
            bitstring+= '0'
    return bitstring


def scale_alphas(N, alphaArray, scaleVal=2/np.pi):
    '''
    Scales every alpha in alphaArray by the scaleVal (in place).
    
    :param N:          (int)        number of iterations/target qubits
    :param alphaArray: (List[List]) contains alpha values
    :param scaleVal:   (float)      satisfies 0 < scaleVal*alpha < 1 for all alphas
    '''
    # Maybe: add check for correct dimensinos of alphaArray
    for n in range(N):
        for j in range(2**n):
            alphaArray[n][j]*= scaleVal


def binary_alphas(N, K, alphaArray):
    '''
    Converts an array of alphas to binary string format.

    :param N:          (int)               number of iterations/target qubits
    :param K:          (int)               binary precision
    :param alphaArray: (List[List[float]]) Binary tree of alpha values
    :return:           (List[List[str]])   Binary tree of alpha values in bit-string format
    '''
    alphaBinary= []
    for l in range(N):
        alphaBinary.append([])
        for n in range(2**l):
            alphaBinary[l].append(decimal_to_binary(alphaArray[l][n], K))
    return alphaBinary


# Circuit Functions

def setup_qc_compact(N, K):
    '''
    Create a new QuantumCircuit, add target register, alpha register, classical register,
    and initialize all qubits to 0.
    
    :param N:         (int)            - number of iterations/target qubits
    :param K:         (int)            - binary precision
    :return:          (Tuple(QuantumCircuit, QuantumRegister, QuantumRegister))
                            (new QC        , target reg.    , alpha reg.      )
    '''
    qc= QuantumCircuit()
    
    aReg= QuantumRegister(K)
    tReg= QuantumRegister(N)
    
    qc.add_register(aReg)
    qc.add_register(tReg)

    return (qc, tReg, aReg)


def create_Rgate(K, scaleVal=2/np.pi):
    '''
    Creates the R(α) gate, as described by K-W.
    
    :param K:        (int)   binary precision
    :param scaleVal: (float) value that satisfies 0 < scaleVal*alpha < 1 for all alphas
    :return:         (Gate)  R(α) gate
    '''
    R= QuantumCircuit(K+1)
    for k in range(K):
        R.cry(1/(2**k * scaleVal), k, K)

    return R.to_gate(label='R(α)')


def rotate(qc, n, Rgate, aReg, tReg):
    '''
    Applies the rotation gate, controlled on alpha (aReg), to target qubit n (tReg(n)).
    
    :param qc: (QuantumCircuit)  Quantum Circuit containing tReg, aReg
    :param n:  (int)             iteration number
    :Rgate:    (Gate)            R(α) Gate (output of create_Rgate)
    :tReg:     (QuantumRegister) target register
    :aReg:     (QuantumRegister) alpha register
    '''
    if not qc.has_register(tReg):
        print('Warning: QuantumCircuit ' + str(qc) + ' does not contain ' + str(tReg))
    if not qc.has_register(aReg):
        print('Warning: QuantumCircuit ' + str(qc) + ' does not contain ' + str(aReg))
    if Rgate.num_qubits-1 != aReg.size:
        print('Precision of Rgate differs from the number of qubits in QuantumRegister aReg. Exiting...')
    return qc.append(Rgate, list(aReg) + [tReg[n]])


def construct_circuit_simple(N, sigma, mu, verbose=False):
    '''
    Constructs a new QuantumCircuit, and evaluates the simplified KW algorithm
    to prepare a Gaussian(std dev= sigma, mu= mean) distribution in the 
    computational basis states, by pre-computing all alpha values, and applying
    controlled α-rotations.
    
    :param N:         (int)            number of iterations/target qubits
    :param sigma:     (float)          std dev
    :param mu:        (float)          mean
    :param verbose:   (Boolean)        If true, executes helpful print statements
    :return:          (QuantumCircuit) with target register qubits Gaussian-distributed
    '''

    # Generate sigmas, mus, alphas
    if verbose: 
        print('Computing σs, μs, αs...')
    sigmaArray= generate_sigmas(sigma, N)
    muArray= generate_mus(mu, N)
    alphaArray= generate_alphas(N, sigmaArray, muArray)
    
    if verbose:
        print('muArray: ' + str(muArray) + '\n')
        print('alphaArray:  ' + str(alphaArray) + '\n')
        print('Setting up Quantum Circuit and Quantum Registers...')

    qc= QuantumCircuit()

    tReg= QuantumRegister(N)
    qc.add_register(tReg)
    
    if verbose:
        print('Applying first rotation...')
    qc.ry(2*alphaArray[0][0], tReg[0])

    if verbose:
        print('Adding remaining rotations...')
    for n in range(1,N):
        for j in range(2**n):
            alpha= alphaArray[n][j]

            # Get the control state corresponding to alpha
            jB= format(j,'b').zfill(n)
            Rgate= qiskit.circuit.library.RYGate(2*alpha).control(n, ctrl_state= jB)
            ctrl_qb= tReg[:n] # the ctrl state comes out backwards
            ctrl_qb.reverse()
            qc.append(Rgate, ctrl_qb + [tReg[n]])

    if verbose:
        print('\nDone!')
    return qc


# Phase Estimation Method
class DiagonalUnitary():
    '''
    A class that provides a build function to construct a diagonal unitary matrix.
    '''
    def __init__(self, phases):
        '''
        :param _phases: (List[float]) Phases of the diagonal entries of a matrix, e.g. exp(i2π * _phases[0]) is the [0,0] entry
        '''
        self._phases= phases
        
    def build(self, qc, q, q_ancillas=None, params=None):
        '''
        Build function -- builds a unitary matrix on a QuantumCircuit.

        :param self:       (DiagonalUnitary) 
        :param qc:         (QuantumCircuit)  on which to build
        :param q:          (QuantumRegister) on which to build
        :param q_ancillas: (None)            Required by parent class
        :param params:     (None)            Required by parent class
        '''

        Uj= np.exp(1j*2*np.pi*self._phases)
        qc.diagonal(list(Uj), q)

    #Add Build Controlled here


def PhaseEstimation(T, Nu, diag, verbose=False, circuit=False):
    '''
    Constructs a QuantumCircuit, that evaluates phase estimation, where the oracle U
    is a diagonal matrix.
    
    :param T:         (int)            number of target qubits
    :param Nu:        (int)            number of qubits in the oracle space
    :param diag:      (List[float])    Phases of the diagonal entries of the oracle U, e.g. exp(i2π * diag[0]) is the [0,0] entry
    :param verbose:   (bool)           If true, executes helpful print statements
    :param circuit:   (bool)           If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate
    :return:          (QuantumCircuit) Phase Estimation algorithm
                      (Gate)
    '''

    if verbose:
        print('T= %d\n' %(T))
        print('Nu= %d\n' %(Nu))
        print('Phases: ' + str(diag))
    
    t= QuantumRegister(T, 't')
    u= QuantumRegister(Nu, 'u')
    
    qc= QuantumCircuit(t, u) # note the order

    # Divide the phases by 2, so they're in [0, 1/2]. Then will have to rotate by twice the result
    diag= diag / 2.

    # Generate Superposition
    qc.h(t)
    # Apply Controlled-U^j gates
    for i in range(T):
        j= 2**i
        phases= (j * diag) % 1
        
        Uj= DiagonalUnitary(phases)
        
        Uj.build(qc, u, t[i])

        # CircuitLibrary applies a global phase during `build_controlled`, so that the
        # first entry of phases goes to 1. So undo that global phase:
        qc.p(2*np.pi*phases[0], t[i])

    # Inverse Fourier Transform
    qc.compose(basic_circuits.fourier_transform(t, inv=True, swap=False), inplace=True)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='PhaseEstimation')


def construct_circuit_PE(N, K, sigma, mu, verbose=False, scaleVal=2/np.pi):
    '''
    Constructs a new QuantumCircuit, and evaluates the simplified KW algorithm
    to prepare a Gaussian(std dev= sigma, mu= mean) distribution in the 
    computational basis states, pre-computing all alpha values and encoding them
    using phase estimation.
    
    :param N:         (int)            number of iterations/target qubits
    :param K:         (int)            binary precision of alphas
    :param sigma:     (float)          std dev
    :param mu:        (float)          mean
    :param verbose:   (Boolean)        If true, executes helpful print statements
    :param scaleVal:  (float)          value that satisfies 0 < scaleVal*alpha < 1 for all alphas
    :return:          (QuantumCircuit) with target register qubits Gaussian-distributed
    '''

    # Generate sigmas, mus, alphas
    if verbose: 
        print('Computing σs, μs, αs...')
    sigmaArray= generate_sigmas(sigma, N)
    muArray= generate_mus(mu, N)
    alphaArray= generate_alphas(N, sigmaArray, muArray)
    
    # Scale alphas and convert to binary
    if verbose:
        print('Scaling αs and converting to binary...\n')
    scale_alphas(N, alphaArray)
    alphaBinary= binary_alphas(N, K, alphaArray)
    if verbose:
        print('muArray: ' + str(muArray) + '\n')
        print('alphaBinary:  ' + str(alphaBinary) + '\n')
        print('Setting up Quantum Circuit and Quantum Registers...')
    qc, tReg, aReg= setup_qc_compact(N, K)

    if verbose:
        print('Creating rotation gate...')
    Rgate= create_Rgate(K, scaleVal)
    
    if verbose:
        print('Setting alpha0, applying first rotation, and uncomputing alpha0...')

    set_alpha_compact(qc, 0, alphaBinary, tReg, aReg)
    rotate(qc, 0, Rgate, aReg, tReg)
    set_alpha_compact(qc, 0, alphaBinary, tReg, aReg)

    if verbose:
        print('Adding remaining rotations: set alpha(n), rotate target(n), uncompute alpha(n)...')

    # Also need an R gate that rotates by twice alpha; scaleval is in the denominator of the rotation angle
    Rgate= create_Rgate(K, scaleVal/2.)

    for n in range(1,N):
        # Replacing set_aplha_compact with PhaseEsimation
        diag= np.array(alphaArray[n])

        # Get the right ordering of tReg, because qiskit is weird
        tRegRev= tReg[:n]
        tRegRev.reverse()

        qc.compose(PhaseEstimation(K, n, diag, circuit=False), aReg[:] + tRegRev, inplace=True)
        rotate(qc, n, Rgate, aReg, tReg)
        qc.compose(PhaseEstimation(K, n, diag, circuit=False).reverse_ops(), aReg[:] + tRegRev, inplace=True)

    if verbose:
        print('\nDone!')
    return qc


# Plotting functions

def simulate(qc, shotNum, backend):
    '''
    :param qc:      (QuantumCircuit)      to simulate
    :param shotNum: (int)                 number of simulations
    :param backend: (QasmSimulator)       
    :return:        (dictionary{str:int}) counts dictionary
    '''
    job_sim = execute(qc, backend, shots=shotNum)
    result_sim = job_sim.result()
    print('Simulation Done. Executed %d shots.' %(shotNum))
    return result_sim.get_counts(qc)


def plot1(counts, N, K, mu, sigma):
    '''
    Plot and save a histogram of counts.
    
    :param counts:    (dictionary{str:int}) counts dictionary, i.e. simulation result
    
    To label plot:
    :param N:         (int)                 number of iterations/target qubits
    :param K:         (int)                 binary precision of alphas
    :param sigma:     (float)               std dev
    :param mu:        (float)               mean
    '''
    # Convert counts to 2s complement integers
    counts_2s= {}
    for key in counts:
        numb= int(key, 2)
        if numb >= 2**(N-1):
            numb-= 2**N
        counts_2s[numb]= counts[key]

    # Plot the counts in 2s complement 
    sub= plt.axes([0,0,1.2,1.2])
    sub.set_xlabel("Basis State (Two's Complement)")
    sub.text(1.00,0.94,'μ = %.2f\nσ = %.2f' %(mu, sigma), fontsize=12, transform=plt.gcf().transFigure)
    sub.text(1.00,0.84,'N = %d\nK = %d' %(N, K), fontsize=12, transform=plt.gcf().transFigure)
    visualization.plot_histogram(counts_2s, title='Histogram: Probability vs. Basis State', ax= sub, bar_labels=True)
    plt.savefig('hi2', bbox_inches='tight')






# Functions that are useful for the acutal KW, but aren't related to the two exponential versions

def setup_qc(N, K, cl= False):
    '''
    Create a new QuantumCircuit, add target register, alpha register, classical register,
    and initialize all qubits to 0.
    
    :param N:         (int)            - number of iterations/target qubits
    :param K:         (int)            - binary precision
    :return:          (Tuple(QuantumCircuit, QuantumRegister, QuantumRegister, ClassicalRegister))
                            (new QC        , target reg.    , alpha reg.     ,  measurement reg.)
    '''
    qc= QuantumCircuit()
    
    aReg= QuantumRegister(K)
    tReg= QuantumRegister(N)
    
    qc.add_register(aReg)
    qc.add_register(tReg)

    if cl == True:
        clReg= ClassicalRegister(N)
        qc.add_register(clReg)
    
        for j in range(qc.num_qubits):
            qc.initialize([1,0], j)

    if cl == True:
        return (qc, tReg, aReg, clReg)
    else:
        return (qc, tReg, aReg)


def setup_alpha0(qc, K, aReg, alphaBinary):
    '''
    Configure the alpha register to the binary representation of alpha0 (in place).

    :param qc:         (QuantumCircuit)  QuantumCircuit that contains aReg
    :param K:          (int)             binary precision
    :param aReg:       (QuantumRegister) alpha register
    :param alphaArray: (List[List[str]]) Binary tree of alpha values in bit-string format
    '''
    if not qc.has_register(aReg):
        print('Warning: QuantumCircuit ' + str(qc) + ' does not contain ' + str(aReg))
    
    for b in range(K):
        if alphaBinary[0][0][b] == '1':
            qc.x(aReg[b])


def set_alpha(qc, n, alphaBinary, tReg, aReg):
    '''
    Encodes the value of alpha(n) into the alpha register (aReg), conditioned 
    on the (n-1)st target qubit.
    
    :param qc:          (QuantumCircuit)  Quantum Circuit that contains tReg, aReg
    :param n:           (int)             iteration number
    :param alphaBinary: (list[list[str]]) contains bit-string representations of alphas
    :param tReg:        (QuantumRegister) target register
    :param aReg:        (QuantumRegister) alpha register
    
    '''
    if not qc.has_register(tReg):
        print('Warning: QuantumCircuit ' + str(qc) + ' does not contain ' + str(tReg))
    if not qc.has_register(aReg):
        print('Warning: QuantumCircuit ' + str(qc) + ' does not contain ' + str(aReg))
        
    N= tReg.size
    K= aReg.size
    if n <= 0 or n>= N:
        print('Index n is not within range 1..%d. Exiting...' %(N-1))
        return
    for j in range(2**n):
        for k in range(K):
            if alphaBinary[n][j][k] == '1':
                jB= format(j,'b')
                jBlen= len(jB)
                if jBlen < n:
                    jB= '0'*(n-jBlen) + jB
                ctrl_gate= qiskit.circuit.library.XGate().control(n, ctrl_state=jB)
                ctrl_qb= tReg[0:n]
                ctrl_qb.reverse()   # for some reason the ctrl_state is interpreted backwards
                qc.append(ctrl_gate, ctrl_qb + [aReg[k]])


def construct_circuit(N, K, sigma, mu, scaleVal=2/np.pi, verbose=False):
    '''
    Constructs a new QuantumCircuit, and evaluates the simplified K-W's algorithm
    to prepare a Gaussian(std dev= sigma, mu= mean) distribution in the 
    computational basis states, by pre-computing all alpha values.
    
    :param N:         (int)            number of iterations/target qubits
    :param K:         (int)            binary precision of alphas
    :param sigma:     (float)          std dev
    :param mu:        (float)          mean
    :param scaleVal:  (float)          value that satisfies 0 < scaleVal*alpha < 1 for all alphas
    :return:          (QuantumCircuit) with target register qubits Gaussian-distributed
    '''
    # Generate sigmas, mus, alphas
    if verbose:
        print('Computing σs, μs, αs...')
    sigmaArray= generate_sigmas(sigma, N)
    muArray= generate_mus(mu, N)
    alphaArray= generate_alphas(N, sigmaArray, muArray)
    
    # Scale alphas and convert to binary
    if verbose:
        print('Scaling αs and converting to binary...\n')
    scale_alphas(N, alphaArray)
    alphaBinary= binary_alphas(N, K, alphaArray)
    if verbose:
        print('muArray: ' + str(muArray) + '\n')
        print('alphaBinary:  ' + str(alphaBinary) + '\n')
    
        print('Initializing Quantum Circuit...')
    qc, tReg, aReg= setup_qc(N, K, cl= False)
    
    if verbose:
        print('Creating rotation gate...')
    Rgate= create_Rgate(K, scaleVal)
    
    if verbose:
        print('Setting alpha0, applying first rotation, and uncomputing alpha0...')
    setup_alpha0(qc, K, aReg, alphaBinary)
    rotate(qc, 0, Rgate, aReg, tReg)
    setup_alpha0(qc, K, aReg, alphaBinary)
    
    if verbose:
        print('Adding remaining rotations: set alpha(n), rotate target(n), uncompute alpha(n)...')
    for n in range(1,N):
        set_alpha(qc, n, alphaBinary, tReg, aReg)
        rotate(qc, n, Rgate, aReg, tReg)
        set_alpha(qc, n, alphaBinary, tReg, aReg)
        
    #print('Measuring target...')
    #qc.measure(tReg, range(qc.num_clbits))
    
    if verbose:
        print('\nDone!')
    return qc


# NEW FUCTIONS
def create_AlphaGate(K, n, alphaBinary):
    '''
    Returns a gate that sets the value of α(n), controlled on the (n-1) less significant target qubits.
    The indexing of this gate is : 
        [0..K-1) <--> [a(0)..a(K-1)), where   α = sum(i=0..K-1) {a(i) * 2^(i-1)}
        [K..K+n) <--> [t(0)..t(n))

    :param K:           (int)             binary precision
    :param n:           (int)             iteration number
    :param alphaBinary: (list[list[str]]) contains bit-string representations of alphas
    :return:            (Gate)            Set α(n) gate
    '''
    AlphaGate= QuantumCircuit(K+n)
    if n == 0: # there is no control
        for k in range(K):
            if alphaBinary[0][0][k] == '1':
                AlphaGate.x(k)
    else:
        for j in range(2**n):
            for k in range(K):
                if alphaBinary[n][j][k] == '1':
                    jB= format(j,'b').zfill(n)
                    ctrl_gate= qiskit.circuit.library.XGate().control(n, ctrl_state=jB)
                    ctrl_qb= AlphaGate.qubits[K:]
                    ctrl_qb.reverse()
                    AlphaGate.append(ctrl_gate, ctrl_qb + [AlphaGate.qubits[k]])
    
    return AlphaGate.to_gate(label='Set α(%d)' %(n))


def set_alpha_compact(qc, n, alphaBinary, tReg, aReg):
    '''
    Exactly the same as 'set_alpha', except the gate applications are bundled up using
    'create_AlphaGate'.
    '''
    if not qc.has_register(tReg):
        print('Warning: QuantumCircuit ' + str(qc) + ' does not contain ' + str(tReg))
    if not qc.has_register(aReg):
        print('Warning: QuantumCircuit ' + str(qc) + ' does not contain ' + str(aReg))
    N= tReg.size
    K= aReg.size
    if n < 0 or n>= N:
        print('Index n is not within range 1..%d. Exiting...' %(N-1))
        return
    qc.append(create_AlphaGate(K, n, alphaBinary), list(aReg) + list(tReg)[0:n])


def construct_circuit_compact(N, K, sigma, mu, verbose=False, scaleVal=2/np.pi):
    '''
    Exactly the same as 'construct_circuit', except setting the alphas is bundled up using
    'create_AlphaGate' and 'set_alpha_compact', so using QuantumCircuit.draw() on the return
    value is less verbose and easier to interpet.

    Also has an additional mode: verbose
    :param verbose: (Boolean) If true, executes helpful print statements.

    There is no classical register included.
    There is no initialization.
    '''

    # Generate sigmas, mus, alphas
    if verbose: 
        print('Computing σs, μs, αs...')
    sigmaArray= generate_sigmas(sigma, N)
    muArray= generate_mus(mu, N)
    alphaArray= generate_alphas(N, sigmaArray, muArray)
    
    # Scale alphas and convert to binary
    if verbose:
        print('Scaling αs and converting to binary...\n')
    scale_alphas(N, alphaArray)
    alphaBinary= binary_alphas(N, K, alphaArray)
    if verbose:
        print('muArray: ' + str(muArray) + '\n')
        print('alphaBinary:  ' + str(alphaBinary) + '\n')
        print('Setting up Quantum Circuit and Quantum Registers...')
    qc, tReg, aReg= setup_qc_compact(N, K)
 
    if verbose:
        print('Creating rotation gate...')
    Rgate= create_Rgate(K, scaleVal)
    
    if verbose:
        print('Setting alpha0, applying first rotation, and uncomputing alpha0...')
    set_alpha_compact(qc, 0, alphaBinary, tReg, aReg)
    rotate(qc, 0, Rgate, aReg, tReg)
    set_alpha_compact(qc, 0, alphaBinary, tReg, aReg)

    if verbose:
        print('Adding remaining rotations: set alpha(n), rotate target(n), uncompute alpha(n)...')
    for n in range(1,N):
        set_alpha_compact(qc, n, alphaBinary, tReg, aReg)
        rotate(qc, n, Rgate, aReg, tReg)
        set_alpha_compact(qc, n, alphaBinary, tReg, aReg)

    if verbose:
        print('\nDone!')
    return qc