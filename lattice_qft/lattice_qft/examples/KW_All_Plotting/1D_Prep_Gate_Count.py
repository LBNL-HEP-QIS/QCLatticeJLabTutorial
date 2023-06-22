#General Imports
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#Local Imports
from gate_counting_functions import *

#
# Plot some arbitrary values of sigma
#
plt.rcParams['mathtext.fontset'] = 'cm'

kMax= 18 # number of qubits per lattice site.
d= 2
sigma0_list= [2, 8, 32]
countsArray_list= []

kArray= np.arange(2, kMax)

transition= {} # dictionary that stores where sigma transitions between regimes.

for sigma0 in sigma0_list:
    countsArray= np.zeros(kMax-2, dtype=int)

    for k in range(2, kMax):
        # j = k - 1
        r= k
        b= r + 1
    
        if k == 2:
            sigma= sigma0/2.
    
            if sigma > 0.65:
                countsArray[k-2]= large_sigma_alpha_count(k-1, d, r)
                
            elif sigma > 0.05:
                countsArray[k-2]= int_sigma_alpha_count(k-1, d, r, b)
            
                if '0.65' not in transition:
                    transition['0.65']= k

            else:
                countsArray[k-2]= 2
            
                if '0.05' not in transition:
                    transition['0.05']= k

        else:
            sigma/= 2.
            countsArray[k-2]= countsArray[k-3]
        
            if sigma > 0.65:
                countsArray[k-2]+= large_sigma_alpha_count(k-1, d, r)
                
            elif sigma > 0.05:
                countsArray[k-2]+= int_sigma_alpha_count(k-1, d, r, b)
            
                if '0.65' not in transition:
                    transition['0.65']= k

            else:
                countsArray[k-2]+= 2
            
                if '0.05' not in transition:
                    transition['0.05']= k
        
        countsArray[k-2]+= 2 * r
        #print('Counts= %d, k= %d, sigma= %.3f' %(countsArray[k-2], k, sigma0))
  
    countsArray_list.append(countsArray)

fig, ax= plt.subplots()
ax.yaxis.get_offset_text().set_font('times new roman')
ax.yaxis.get_offset_text().set_fontsize(20)

for j in range(len(sigma0_list)):
    plt.plot(kArray, countsArray_list[j]/1000, label='$\sigma=%.1f$' %(sigma0_list[j]), linewidth=3)

#plt.plot(kArray, 2**(kArray - 1.), label='Generic Symmetric', linewidth=3, linestyle='--')
plt.plot(kArray, (2**(kArray - 1.))/1000, label='Generic Symmetric', linewidth=3, linestyle='--')

#plt.title('1D Gaussian State Prep. Gate Count')
plt.xlabel('State Qubits ($k$)', fontname='times new roman', size=26)
plt.ylabel(r'CNOTs $\left[\times 10^3\right]$', fontname='times new roman', size=26)

plt.xticks(fontsize=20, fontname= 'times new roman')
plt.yticks(fontsize=20, fontname= 'times new roman')
#plt.ticklabel_format(axis="y", style="sci", scilimits=(0,3))

plt.legend(prop=matplotlib.font_manager.FontProperties(family='times new roman', size= 16))
plt.show()
#plt.savefig('1DPrep_CNOTs_fixedsigma_k<=%d.pdf' %(kMax), bbox_inches='tight')