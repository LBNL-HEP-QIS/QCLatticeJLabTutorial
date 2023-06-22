from lattice_qft.Scalar_Field_Theory.stateprep_1D import *
import matplotlib, numpy as np, matplotlib.pyplot as plt
from qiskit import Aer, execute
from qiskit.quantum_info import Statevector, state_fidelity

#Load in the different backends
simulator = Aer.get_backend('unitary_simulator')
simulator_state = Aer.get_backend('statevector_simulator')

sigma= 1.
mu= -0.5


#for b in range(2, bMax):
bMax= 10
kList= [4, 6, 8]

fidels= np.zeros((len(kList), bMax - 2))

for b in range(2, bMax):
    for k in range(len(kList)):
        print(b, k)
        print('simple circuit')
        qc1= construct_circuit_PE(kList[k], sigma, mu, verbose=False)
        result1 = execute(qc1, simulator_state).result()
        sv1= result1.get_statevector().data
        
        print('approx circuit')
        qc2= construct_circuit(kList[k], b, sigma, mu)
        result2 = execute(qc2, simulator_state).result()
        #print(qc2.draw())
        sv2_nonzero= result2.get_statevector().data[np.where(abs(result1.get_statevector().data) != 0)]
        #print(sv2_nonzero)
        #print(sv1)
        
        fidels[k][b-2]= qiskit.quantum_info.state_fidelity(sv1, sv2_nonzero)
        
print("Done.")

fig, ax= plt.subplots(figsize=(8,6))
ax.yaxis.get_offset_text().set_font('times new roman')
ax.yaxis.get_offset_text().set_fontsize(24)
ax.xaxis.get_offset_text().set_font('times new roman')
ax.xaxis.get_offset_text().set_fontsize(24)
      
for k in range(len(kList)):
    plt.plot(np.arange(2, bMax-1), np.log2(1-fidels[k][:-1]), label= r'$k=%d$' %(kList[k]), linewidth= 3)


#plt.title('1D Gaussian State Prep. Gate Count')
plt.xlabel(r'$\alpha$ register size $(b)$', fontname='times new roman', size=26)
plt.ylabel(r'$\log_2(1 - $Fidelity)', fontname='times new roman', size=26)

plt.xticks(np.arange(2, bMax), fontsize=20, fontname= 'times new roman')
plt.yticks(np.arange(-17.5, 2.5, 2.5), fontsize=20, fontname= 'times new roman')

plt.legend(prop=matplotlib.font_manager.FontProperties(family='times new roman', size= 17), loc=1)
plt.show()