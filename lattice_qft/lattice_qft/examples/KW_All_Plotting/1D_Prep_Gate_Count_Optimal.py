#General Imports
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#Local Imports
from gate_counting_functions import *


#
# Plot sigma, as a function of k
#
plt.rcParams['mathtext.fontset'] = 'cm'

kMax= 19 # number of qubits per lattice site.
d= 2

kArray= np.arange(2, kMax)
countsArray= np.zeros(kMax-2, dtype=int)

transition= {} # dictionary that stores where sigma transitions between regimes.

for k in range(2, kMax):
    sigma= 0.5 * np.sqrt(2**k / np.pi)
    counts= 0
    for j in range(1, k):
        r= j + 1     # a minimal assumption
        b= r + 1
        sigma/= 2.
    
        if sigma > 0.65:
            counts+= large_sigma_alpha_count(j, d, r)
                
        elif sigma > 0.05:
            counts+= int_sigma_alpha_count(j, d, r, b)
            
            if '0.65' not in transition:
                transition['0.65']= j

        else:
            counts+= 2
            
            if '0.05' not in transition:
                transition['0.05']= j
    
        counts+= 2 * r # rotating alpha
    countsArray[k-2]= counts
    
    print('Counts= %d, k= %d, sigma= %.3f' %(counts, k, sigma))

fig, ax= plt.subplots()
ax.yaxis.get_offset_text().set_font('times new roman')
ax.yaxis.get_offset_text().set_fontsize(20)

plt.plot(kArray, countsArray/1000, label=r'$\sigma=\frac{1}{2}\sqrt{\frac{2^k}{\pi}}$', linewidth=3)
#plt.plot(kArray, 2**(kArray - 1.) + kArray - 3, label='Generic Symmetric', linewidth=3, color='C3', linestyle='--')
plt.plot(kArray, (2**(kArray - 1.) + kArray - 3)/1000, label='Generic Symmetric', linewidth=3, color='C3', linestyle='--')

#plt.title('1D Gaussian State Prep. Gate Count')
plt.xlabel('State Qubits ($k$)', fontname='times new roman', size=26)
plt.ylabel(r'CNOTs $\left[\times 10^3\right]$', fontname='times new roman', size=26)

plt.xticks(fontsize=20, fontname= 'times new roman')
plt.yticks(fontsize=20, fontname= 'times new roman')
#plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#plt.yscale('log')

plt.legend(prop=matplotlib.font_manager.FontProperties(family='times new roman', size= 18))
plt.show()
#plt.savefig('1DPrep_CNOTs_scalingsigma_k<=%d.pdf' %(kMax), bbox_inches='tight')
#plt.savefig('1D_CNOTs_k<=%d.png' %(kMax))