'''
    Arithmetic functions included:

        Absolute (modular=False):
            RippleAdder:      (n, n)     --> (n, n+1)    ; (a, b)    --> (a, a+b)
            Subtractor:       (n, n)     --> (n, n+1)    ; (a, b)    --> (a, a-b)
            Subtractor2:      (n, n)     --> (n+1, n)    ; (a, b)    --> (a-b, b)
            Ctrl-Adder:       (1, n, n)  --> (1, n, n+1) ; (c, a, b) --> (c, a, (a+b)*c)
            Ctrl-Subtractor:  (1, n, n)  --> (1, n, n+1) ; (c, a, b) --> (c, a, (b-a)*c)
            Multiplier:       (n, n, 2n) --> (n, n, 2n)  ; (a, b, 0) --> (a, b, a*b)
            Add-Sub           (1, n, n)  --> (1, n+1, n) ; (c, a, b) --> (c, (a+b) - (2b)*c, b)

        Modular (modular= True):
            RippleAdder:      (n, n)    --> (n, n)    ; (a, b)    --> (a, a+b (mod 2^n))
            Subtractor:       (n, n)    --> (n, n)    ; (a, b)    --> (a, a-b (mod 2^n))
            Subtractor2:      (n, n)    --> (n, n)    ; (a, b)    --> (a-b (mod 2^n), b)
            Ctrl-Adder:       (1, n, n) --> (1, n, n) ; (c, a, b) --> (c, a, (a+b)*c (mod 2^n))
            Ctrl-Subtractor:  (1, n, n) --> (1, n, n) ; (c, a, b) --> (c, a, (b-a)*c (mod 2^n))
            Add-Sub           (1, n, n) --> (1, n, n) ; (c, a, b) --> (c, (a+b) - (2b)*c, b)
            ModularMultiplier:  (p, 1, p, p+m) --> (p, 1, p, p+m) ; (a, b[-1], b, 0) --> (a, b[-1], b, a*b (mod 2^k))
            ModularMultiplier2: (p, 1, k, p)   --> (p, 1, k, p)   ; (a, b[-1], b, 0) --> (a, b[-1], b, a*b (mod 2^k))
            ModularMultiplier3: (p, 1, p, p)   --> (p, 1, p, p)   ; (a, b[-1], b, 0) --> (a, b[-1], b, a*b (mod 2^k))
            ModularSquarer:     (p, 1, 1, p)   --> (p, 1, 1, p)   ; (a, a[-1], 0, 0) --> (a, a[-1], 0, a^2 (mod 2^k))

        Division:
            Divider:            See function
            Divider2:           See function

        Note 0: 
            p = k + m
        Note 1:
            For arithmetic circuits that use ancillary qubits, those ancillaries 
            must be initialized to zero before applying an arithmetic circuit.
        Note 2:
            All arithmetic is in Two's Complement notation, unless specified otherwise.
'''


####################################################################################################
# General Imports                                                                                  #
####################################################################################################
import numpy as np
import sys
import matplotlib.pyplot as plt
from qiskit import(
  QuantumCircuit,
  QuantumRegister,
  ClassicalRegister,
  execute,
  Aer,
  circuit)
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import MCMT
import qiskit.circuit.library.arithmetic.adders as test_add

####################################################################################################
# Local Imports                                                                                    #
####################################################################################################

from lattice_qft.core.arithmetic_pkg.testing_utilities import *


####################################################################################################
# Arithmetic Functions                                                                             #
####################################################################################################
def MAJ(qc, c, b, a):
    '''
    Adds MAJ gate in place to a QuantumCircuit. See https://arxiv.org/abs/quant-ph/0410184.

    :param qc: (QuantumCircuit)
    :param a:  (QuantumRegister) 
    :param b:  (QuantumRegister)
    :param c:  (QuantumRegister)
    '''
    qc.cx(a, b)
    qc.cx(a, c)
    qc.ccx(c, b, a)
    return

####################################################################################################
def UMA(qc, c, b, a):
    '''
    Adds UMA gate in place to a QuantumCircuit. See https://arxiv.org/abs/quant-ph/0410184.
    '''
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.cx(c, b)
    return

####################################################################################################
def ctrl_MAJ(qc, ctrl, c, b, a):
    '''
    Adds a controlled-MAJ gate in place to a QuantumCircuit. See https://arxiv.org/abs/quant-ph/0410184.

    :param qc:   (QuantumCircuit)
    :param a:    (QuantumRegister) 
    :param b:    (QuantumRegister)
    :param c:    (QuantumRegister)
    :param ctrl: (QuantumRegister)
    '''
    qc.ccx(a, ctrl, b)
    qc.ccx(a, ctrl, c)
    qc.mcx([c, b, ctrl], a, mode='noancilla')
    return

####################################################################################################
def ctrl_UMA(qc, ctrl, c, b, a):
    '''
    Adds a controlled-UMA gate in place to a QuantumCircuit. See https://arxiv.org/abs/quant-ph/0410184.
    '''
    qc.mcx([c, b, ctrl], a)
    qc.ccx(a, ctrl, c)
    qc.ccx(c, ctrl, b)
    return

####################################################################################################
def RippleAdder(K, modular=False, circuit=True, twos_comp=True):
    '''
    Constructs a new QuantumCircuit that implements the Ripple-Carry adder given in
    https://arxiv.org/abs/quant-ph/0410184.

    :param K:       (int)                   Binary precision of inputs
    :param modular: (bool)                  If true, performs modular addition. If false, outputs the addition result on K+1 qubits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |anc> |a> |b> --> |anc> |a> |a+b>
    '''

    a= QuantumRegister(K)
    anc= QuantumRegister(1)

    if not modular:
        b= QuantumRegister(K+1)
    else:
        b= QuantumRegister(K)

    # Note the order of the Registers
    qc= QuantumCircuit(anc, a, b)

    # This fixes two's complement
    if twos_comp:
        if not modular:
            qc.cx(a[K-1], b[K])
            qc.cx(b[K-1], b[K])

    # Add MAJ gates
    MAJ(qc, anc, b[0], a[0])
    for n in (np.arange(K-1)+1):
        MAJ(qc, a[n-1], b[n], a[n])
    
    if not modular:
        qc.cx(a[K-1], b[K])
    
    # Add UMA gates
    for n in (np.arange(K-1)+1):
        UMA(qc, a[K-n-1], b[K-n], a[K-n])
    UMA(qc, anc, b[0], a[0])
    
    if circuit:
        return qc
    else:
        return qc.to_gate(label='ripple_add(%d)' %(K))
test = RippleAdder(1)
print(test.draw())
test2 = test_add.cdkm_ripple_carry_adder.CDKMRippleCarryAdder(1)
print(test2.decompose().decompose().draw())
qc = QuantumCircuit(4)
qc.compose(test, inplace=True)
qc.measure_all()
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator, shots=1000).result()
counts = result.get_counts()
print(counts)
####################################################################################################
def Subtractor(K, modular=False, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements a Subtractor, using the RippleAdder
    plus some additional NOTs.

    :param K:       (int)                   Binary precision of inputs
    :param modular: (bool)                  If true, performs modular addition. If false, outputs the addition result on K+1 qubits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |anc> |a> |b> --> |anc> |a> |a-b>
    '''

    a= QuantumRegister(K)
    anc= QuantumRegister(1)

    if modular == False:
        b= QuantumRegister(K+1)
    else:
        b= QuantumRegister(K)

    qc= QuantumCircuit(anc, a, b)
    qc.x(a)

    if modular == False:
        qc.compose(RippleAdder(K, modular=False, circuit=True), inplace=True)
    else:
        qc.compose(RippleAdder(K, modular=True, circuit=True), inplace=True)

    qc.x(b)
    qc.x(a)
    
    if circuit:
        return qc
    else:
        return qc.to_gate(label='subtractor')

####################################################################################################
def Subtractor2(K, modular=False, circuit=True):
    '''
    Same as Subtractor, except the register organization is different.

    Register Organization:
        |anc> |a> |b> --> |anc> |a-b> |b>
    '''

    b= QuantumRegister(K)
    anc= QuantumRegister(1)

    if modular == False:
        a= QuantumRegister(K+1)
    else:
        a= QuantumRegister(K)

    qc= QuantumCircuit(anc, a, b)

    qc.x(a[:K])

    if modular == False:
        qc.compose(RippleAdder(K, modular=False, circuit=True), anc[:] + b[:] + a[:], inplace=True)
    else:
        qc.compose(RippleAdder(K, modular=True, circuit=True), anc[:] + b[:] + a[:], inplace=True)

    qc.x(a)
    
    if circuit:
        return qc
    else:
        return qc.to_gate(label='subtractor2')

####################################################################################################
def CAdder(K, modular=False, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements the Controlled-Adder given in
    https://arxiv.org/pdf/1706.05113.

    :param K:       (int)                   Binary precision of inputs
    :param modular: (bool)                  If true, performs modular addition. If false, outputs the addition result on K+1 qubits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |c> |a> |b> |z> --> |c> |a> |(a+b) x c> |(a+b)[output carry]>     if not modular
        |c> |a> |b>     --> |c> |a> |(a+b) x c>                           if modular
    '''

    a= QuantumRegister(K) # Summand 1
    b= QuantumRegister(K) # Summand 2: result stored here
    c= QuantumRegister(1) # Control qubit

    if not modular:
        z= QuantumRegister(2) # Output carry in z[0]
        qc= QuantumCircuit(c, a, b, z)
    else:
        qc= QuantumCircuit(c, a, b)

    # Fix for two's complement in the case that there is no addition
    if not modular:
       qc.ccx(c, a[K-1], z[0])
       qc.cx(b[K-1], z[0])

    # Step 1
    for n in range(K-1):
        qc.cx(a[n+1], b[n+1])

    # Step 2.1
    if not modular:
        qc.ccx(c, a[K-1], z[0])

    # Step 2.2
    for n in reversed(range(K-2)):
        qc.cx(a[n+1], a[n+2])

    # Step 3
    for n in range(K-1):
        qc.ccx(a[n], b[n], a[n+1])

    if not modular:
        # Step 4.1
        qc.ccx(a[K-1], b[K-1], z[1])
        # Step 4.2    
        qc.ccx(c, z[1], z[0])    
        # Step 4.3
        qc.ccx(a[K-1], b[K-1], z[1])

    # Step 4.4
    qc.ccx(c, a[K-1], b[K-1])    
    
    # Step 5
    for n in reversed(range(K-1)):
        # range(3) in this case: 2, 1, 0
        qc.ccx(a[n], b[n], a[n+1])
        qc.ccx(c, a[n], b[n])
        
    # Step 6
    for n in range(K-2):
        qc.cx(a[n+1], a[n+2])
    
    # Step 7
    for n in range(K-1):
        qc.cx(a[n+1], b[n+1])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='ctrl_adder')

####################################################################################################
def CSubtractor(K, modular=False, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements a Controlled-Subtractor 
    that uses the Controlled-Adder plus some NOTs.

    :param K:       (int)                   Binary precision of inputs
    :param modular: (bool)                  If true, performs modular addition. If false, outputs the addition result on K+1 qubits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |c> |a> |b> |z> --> |c> |a> |(b-a) x c> |(b-a)[output carry]>     if not modular
        |c> |a> |b>     --> |c> |a> |(b-a) x c>                           if modular
    '''

    a= QuantumRegister(K) # Summand 1
    b= QuantumRegister(K) # Summand 2: result stored here
    c= QuantumRegister(1) # Control qubit

    if not modular:
        z= QuantumRegister(2) # Output carry in z[0]
        qc= QuantumCircuit(c, a, b, z)
    else:
        qc= QuantumCircuit(c, a, b)

    # Fix for two's complement in the case that there is no addition
    if not modular:
       qc.ccx(c, a[K-1], z[0])
       qc.cx(b[K-1], z[0])

    # Invert b
    qc.x(b)

    # Step 1
    for n in range(K-1):
        qc.cx(a[n+1], b[n+1])

    # Step 2.1
    if not modular:
        qc.ccx(c, a[K-1], z[0])

    # Step 2.2
    for n in reversed(range(K-2)):
        qc.cx(a[n+1], a[n+2])

    # Step 3
    for n in range(K-1):
        qc.ccx(a[n], b[n], a[n+1])

    if not modular:
        # Step 4.1
        qc.ccx(a[K-1], b[K-1], z[1])
        # Step 4.2    
        qc.ccx(c, z[1], z[0])    
        # Step 4.3
        qc.ccx(a[K-1], b[K-1], z[1])

    # Step 4.4
    qc.ccx(c, a[K-1], b[K-1])    
    
    # Step 5
    for n in reversed(range(K-1)):
        # range(3) in this case: 2, 1, 0
        qc.ccx(a[n], b[n], a[n+1])
        qc.ccx(c, a[n], b[n])
        
    # Step 6
    for n in range(K-2):
        qc.cx(a[n+1], a[n+2])
    
    # Step 7
    for n in range(K-1):
        qc.cx(a[n+1], b[n+1])

    # Invert to the result
    qc.x(b)

    #if not modular:
    #    qc.x(z[0])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='ctrl_subtractor')

####################################################################################################
def Multiplier(K, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements a Multiplier, using the standard
    partial sum formula (long multiplication), with the Controlled-Adder.

    :param K:       (int)                   Binary precision of inputs
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |a> |b> |p> --> |a> |b> |a*b>

    Note: |b> is extended by an additional qubit to provide an ancilla for the Controlled-Adder.
    Note: Both inputs |a>, |b> can be either positive or negative numbers.
    '''

    a= QuantumRegister(K, 'a')
    b= QuantumRegister(K+1, 'b') # b is a K-bit number, that is extended by 1 bit
    p= QuantumRegister(2*K, 'p')

    qc= QuantumCircuit(a, b, p)

    # First partial product
    for n in range(K+1):
        qc.ccx(a[0], b[n], p[n])

    # Extend the sign (two's complement)
    qc.cx(p[K], p[K+1])

    # Order of CAdder is (c, a, b, z)
    for n in range(K-2):
        qc.compose(CAdder(K+1, modular=True, circuit=False), [a[n+1]] + b[:] + p[n+1:n+K+2], inplace=True)
        qc.cx(p[K+n+1], p[K+n+2]) # extend the sign

    # Finally, subtract the last partial product (two's complement)
    qc.compose(CSubtractor(K+1, modular=True, circuit=False), [a[K-1]] + b[:] + p[K-1:2*K], inplace=True)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='multiplier')

####################################################################################################
def AddSub(K, modular=True, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements an Adder-Subtractor,
    where the circuit implements either addition (Ripple Adder) or 
    subtraction (Subtractor2) based on the value of a control qubit, using 
        a - b = a + not(b) + 1

    :param K:       (int)                   Binary precision of inputs
    :param modular: (bool)                  If true, performs modular addition. If false, outputs the addition result on K+1 qubits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |ctrl> |a> |b>  --> |ctrl> |a+b> |b>        if ctrl= 0
        |ctrl> |a> |b>  --> |ctrl> |a-b> |b>        if ctrl= 1
    '''

    ctrl= QuantumRegister(1)
    if modular:
        a= QuantumRegister(K)
    else:
        a= QuantumRegister(K+1)
    b= QuantumRegister(K)

    qc= QuantumCircuit(ctrl, a, b)

    # Sign extension (two's complement)
    if not modular:
        qc.cx(a[K-1], a[K])

    for k in range(K):
        qc.cx(ctrl, b[k])

    qc.compose(RippleAdder(K, modular, circuit=False), ctrl[:] + b[:] + a[:], inplace=True)

    for k in range(K):
        qc.cx(ctrl, b[k])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='AddSub')

####################################################################################################
def Divider(K, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements integer division, 
    according to the non-restoring algorithm from https://arxiv.org/pdf/1809.09732.pdf,
    which produces a quotient and remainder.

    Note: (Precondition) Both inputs must be postive numbers in Two's Complement. 
                         The behavior of the circuit is strange if not.

    :param K:       (int)                   Binary precision of inputs
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization: (a bit confusing)
        |a[:K-1], a[K-1]> |B> |q> |anc> --> |(a%B), (a/b)[0]> |B> |(a/b)[1:]> |anc>
    '''

    # R= a[0..K-2]
    # Q= a[n-1] + [0]*(K-1)
    a= QuantumRegister(K)
    B= QuantumRegister(K)
    q= QuantumRegister(K-1)
    anc= QuantumRegister(1)

    qc= QuantumCircuit(a, q, B, anc)

    # Step 1 -- Q = Q - b
    Q= [a[K-1]] + q[:] # size K
    R= a[:K-1] # size K-1

    qc.compose(Subtractor2(K, modular=True, circuit=False), anc[:] + Q[:] + B[:], inplace=True)

    # Step 2 --
    for k in range(1, K):
        Y= R[K-1-k:] + Q[:K-k]
        qc.x(Q[K-k])
        qc.compose(AddSub(K, modular=True, circuit=False), [Q[K-k]] + Y[:] + B[:], inplace=True)

    # Step 3 --
    qc.compose(CAdder(K-1, modular=True, circuit=False), [Q[0]] + B[:K-1] + R[:], inplace=True)
    qc.x(Q[0])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='Divider')

####################################################################################################
def Divider2(K, circuit=True):
    '''
    Same as Divider, but has an alternate Register Organizatoin. Referencing 
    the Register Organization of Divider, denote
        R= a[:K-1]
        Q= a[K-1] + q
    '''

    R= QuantumRegister(K-1)
    B= QuantumRegister(K)
    Q= QuantumRegister(K)
    anc= QuantumRegister(1)

    qc= QuantumCircuit(R, Q, B, anc)

    qc.compose(Subtractor2(K, modular=True, circuit=False), anc[:] + Q[:] + B[:], inplace=True)

    # Step 2 --
    for k in range(1, K):
        Y= R[K-1-k:] + Q[:K-k]
        qc.x(Q[K-k])
        qc.compose(AddSub(K, modular=True, circuit=False), [Q[K-k]] + Y[:] + B[:], inplace=True)

    # Step 3 --
    qc.compose(CAdder(K-1, modular=True, circuit=False), [Q[0]] + B[:K-1] + R[:], inplace=True)
    qc.x(Q[0])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='Divider2')

####################################################################################################
def ModularMultiplier(k, m, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements a Modular Multiplier, using the standard
    partial sum formula (long multiplication), with the Controlled-Adder. The result is a product (mod 2^k).

    :param k:       (int)                   Number of integral binary digits
    :param m:       (int)                   Number of decimal binary digits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |a> |b> |p> --> |a> |b> |a*b (mod 2^k)>

    Note: |b> is extended by an additional qubit to provide an ancilla for the Controlled-Adder.
    Note: Both inputs |a>, |b> can be either positive or negative numbers.

    Note: Does NOT accumulate. |p> must be initialized to |0>
    '''
    p= k + m
    a= QuantumRegister(p, 'a')
    b= QuantumRegister(p+1, 'b') # b is a K-bit number, that is extended by 1 bit
    prod= QuantumRegister(p + m, 'p') # result is (mod 2^k)

    qc= QuantumCircuit(a, b, prod)

    # First partial product
    for n in range(p+1):
        qc.ccx(a[0], b[n], prod[n])

    # Extend the sign (two's complement)
    if m > 1:
        qc.cx(prod[p], prod[p+1])

    # Order of CAdder is (c, a, b, z)
    for n in range(m-1):
        qc.compose(CAdder(p+1, modular=True, circuit=False), [a[n+1]] + b[:] + prod[n+1:n+p+2], inplace=True)
        if n+2 < m:
            qc.cx(prod[p+n+1], prod[p+n+2]) # extend the sign

    for n in range(k-1):
        qc.compose(CAdder(p-n, modular=True, circuit=False), [a[n+m]] + b[:p-n] + prod[n+m:], inplace=True)
    # Finally, subtract the last partial product (two's complement)

    qc.compose(CSubtractor(m+1, modular=True, circuit=False), [a[p-1]] + b[:m+1] + prod[p-1:], inplace=True)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='modular_multiplier')

####################################################################################################
def ModularMultiplier2(k, m, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements a Modular Multiplier, using the standard
    partial sum formula (long multiplication), with the Controlled-Adder. The result is a product (mod 2^k), 
    with m decimal digits.

    :param k:       (int)                   Number of integral binary digits
    :param m:       (int)                   Number of decimal binary digits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |a> |b> |p> --> |a> |b> |a*b (mod 2^k)>

    Note: |a> is a register with k+r qubits (decimal with r digits to the right of the decimal)
    Note: |b> is a register with k qubits (integer)
    Note: Qubits connected to the |b> register must represent an integer, for the truncation to be exact
    Note: |b> is extended by an additional qubit to provide an ancilla for the Controlled-Adder.
    Note: Both inputs |a>, |b> can be either positive or negative numbers.

    Note: Does NOT accumulate. |p> must be initialized to |0>
    '''
    p= k + m
    a= QuantumRegister(p, 'a')
    b= QuantumRegister(k+1, 'b') # b is a K-bit number, that is extended by 1 bit
    prod= QuantumRegister(p, 'p') # result is (mod 2^k)

    qc= QuantumCircuit(a, b, prod)

    # First partial product
    for n in range(k+1):
        qc.ccx(a[0], b[n], prod[n])

    # Extend the sign (two's complement)
    if m > 1:
        qc.cx(prod[k], prod[k+1])

    # Order of CAdder is (c, a, b, z)
    for n in range(m-1):
        qc.compose(CAdder(k+1, modular=True, circuit=False), [a[n+1]] + b[:] + prod[n+1:n+k+2], inplace=True)
        if n+2 < m:
            qc.cx(prod[k+n+1], prod[k+n+2]) # extend the sign

    for n in range(k-1):
        qc.compose(CAdder(k-n, modular=True, circuit=False), [a[n+m]] + b[:k-n] + prod[n+m:], inplace=True)

    # Finally, subtract the last partial product (two's complement)
    #qc.compose(CSubtractor(1, modular=True, circuit=False), [a[p-1]] + b[:1] + prod[p-1:], inplace=True)
    qc.ccx(a[p-1], b[0], prod[p-1])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='modular_multiplier2(%d, %d)' %(k, m))

####################################################################################################
def ModularMultiplier3(p, r, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements a Modular Multiplier, using the standard
    partial sum formula (long multiplication), with the Controlled-Adder. The result is a product (mod 2^k),
    with m decimal digits.

    :param k:       (int)                   Number of integral binary digits
    :param r:       (int)                   Number of decimal binary digits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |a> |b> |p> --> |a> |b> |a*b (mod 2^k)>

    Note: |a> is a register with k+m qubits (decimal with m digits to the right of the decimal)
    Note: |b> is a register with k+m qubits (decimal with m digits to the right of the decimal), and 1 padding qubit
    Note: |b> is extended by an additional qubit to provide an ancilla for the Controlled-Adder.
    Note: Both inputs |a>, |b> can be either positive or negative numbers.

    Note: Does NOT accumulate. |p> must be initialized to |0>
    '''
    k= p - r
    # r= m
    # p= b
    a= QuantumRegister(p, 'a')
    b= QuantumRegister(p+1, 'b') # b is a K-bit number, that is extended by 1 bit
    prod= QuantumRegister(p, 'p') # result is (mod 2^k)

    qc= QuantumCircuit(a, b, prod)

    # First partial product
    for n in range(k+1):
        qc.ccx(a[0], b[r+n], prod[n])

    # Extend the sign (two's complement)
    if r > 1:
        qc.cx(prod[k], prod[k+1])

    # Order of CAdder is (c, a, b, z)
    for n in range(r-1):
        qc.compose(CAdder(k+n+2, modular=True, circuit=False), [a[n+1]] + b[r-n-1:] + prod[:n+k+2], inplace=True)
        if n != r - 2:
            qc.cx(prod[k+n+1], prod[k+n+2]) # extend the sign

    for n in range(k-1):
        qc.compose(CAdder(p-n, modular=True, circuit=False), [a[n+r]] + b[:p-n] + prod[n:], inplace=True)

    # Finally, subtract the last partial product (two's complement)
    qc.compose(CSubtractor(r+1, modular=True, circuit=False), [a[p-1]] + b[:r+1] + prod[k-1:], inplace=True)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='modular_multiplier3(%d, %d)' %(p, r))

####################################################################################################
def ModularSquarer(p, r, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements a Modular Squarer, using the standard
    partial sum formula (long multiplication), with the Controlled-Adder. The result is a product (mod 2^k),
    with m decimal digits.

    :param p:       (int)                   Number of total binary digits
    :param r:       (int)                   Number of fractional binary digits
    :param circuit: (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:        (QuantumCircuit / Gate)

    Register Organization:
        |a> |anc> |p> --> |a> |anc> |a*a (mod 2^k)>

    Note: |a> is a register with p qubits (decimal with r digits to the right of the decimal)
    Note: |a> is extended by an additional qubit to .. ????
    Note: |a> can be either positive or negative numbers.

    Note: Does NOT accumulate. |p> must be initialized to |0>
    '''
    k= p - r
    a= QuantumRegister(p+1, 'a')
    anc= QuantumRegister(1, 'anc')
    prod= QuantumRegister(p, 'p') # result is (mod 2^k)

    qc= QuantumCircuit(a, anc, prod)

    # First partial product
    qc.cx(a[0], anc)
    for n in range(k+1):
        qc.ccx(anc, a[r+n], prod[n])
    qc.cx(a[0], anc)

    # Extend the sign (two's complement)
    if r > 1:
        qc.cx(prod[k], prod[k+1])

    # Order of CAdder is (c, a, b, z)
    for n in range(r-1):
        qc.cx(a[n+1], anc)
        qc.compose(CAdder(k+n+2, modular=True, circuit=False), anc[:] + a[r-n-1:] + prod[:n+k+2], inplace=True)
        qc.cx(a[n+1], anc)
        if n != r - 2:
            qc.cx(prod[k+n+1], prod[k+n+2]) # extend the sign

    for n in range(k-1):
        qc.cx(a[n+r], anc)
        qc.compose(CAdder(p-n, modular=True, circuit=False), anc[:] + a[:p-n] + prod[n:], inplace=True)
        qc.cx(a[n+r], anc)

    # Finally, subtract the last partial product (two's complement)
    qc.cx(a[p-1], anc)
    qc.compose(CSubtractor(r+1, modular=True, circuit=False), anc[:] + a[:r+1] + prod[k-1:], inplace=True)
    qc.cx(a[p-1], anc)

    if circuit:
        return qc
    else:
        return qc.to_gate(label='modular_squarer(%d, %d)' %(p, r))

####################################################################################################
def CCM_general(a, p, r, int_input=False, circuit=True):
    '''
    Constructs a new QuantumCircuit that implements CCMA, which accumulates the product
    of a classical number 'a' and a quantum number, using the standard shift-and-add multiplication
    algorithm.

    :param a:         (str)                   p-bit multiplier, a binary string
    :param p:         (int)                   Number of total binary digits
    :param r:         (int)                   Number of fractional binary digits
    :param int_input: (bool)                  True if the (quantum) multiplicand is an integer. In that case, |b> uses p-r qubits.
    :param circuit:   (bool)                  If true, returns the circuit as a QuantumCircuit; if false, returns a the circuit as a Gate

    :return:          (QuantumCircuit / Gate)

    Register Organization:
        {a} |b> |p> --> {a} |b> |p + a*b (mod 2^k)>

    Note: |a> is a classical register with p bits, i.e. a binary string, (decimal with r digits to the right of the decimal)
    Note: |b> is a register with k qubits (integer)
    Note: |b> is extended by an additional qubit to provide an ancilla for the Controlled-Adder.
    Note: Both inputs {a}, |b> can be either positive or negative numbers.
    Note: The general circuit (int_input=False) is not exact, but the circuit with integer input is exact.

    Note: DOES accumulate. |p> can be initialized to any number.
    '''
    if len(a) != p:
        print('Error: Multiplier is not a p-bit binary string. Exiting...')
        return
    k= p - r
    if int_input:
        b= QuantumRegister(k, 'b')
    else:
        b= QuantumRegister(p, 'b') # b is a p-bit number
    ext= QuantumRegister(r, 'r') # extension register
    anc= QuantumRegister(1, 'anc') # Ancillary for RippleAdder
    prod= QuantumRegister(p, 'p') # result is (mod 2^k)

    qc= QuantumCircuit(b, ext, anc, prod)

    # Extend b to |ext>
    for j in range(r):
        qc.cx(b[-1], ext[j])

    # Accumulate partial products

    for n in range(p):
        if a[p-n-1] == '1': # Check if the nth bit of a is one
            if n <= r:
                if int_input:
                    qc.compose(RippleAdder(p-n, modular=True, circuit=False), anc[:] + b[:] + ext[:r-n] + prod[n:], inplace=True) # p - n
                else:
                    qc.compose(RippleAdder(p, modular=True, circuit=False), anc[:] + b[r-n:] + ext[:r-n] + prod[:], inplace=True) # p - n
            else:
                if int_input:
                    qc.compose(RippleAdder(p-n, modular=True, circuit=False), anc[:] + b[:p-n] + prod[n:], inplace=True) # p - n
                else:
                    if n < p-1:
                        qc.compose(RippleAdder(p-n+r, modular=True, circuit=False), anc[:] + b[:p-n+r] + prod[n-r:], inplace=True) # p - n
                    else: # last one is subtraction
                        qc.compose(Subtractor2(p-n+r, modular=True, circuit=False), anc[:] + prod[n-r:] + b[:p-n+r], inplace=True) # p - n

    # Un-Extend b to |ext>
    for j in range(r):
        qc.cx(b[-1], ext[j])

    if circuit:
        return qc
    else:
        return qc.to_gate(label='CCM_general(%d, %d)' %(p, r))