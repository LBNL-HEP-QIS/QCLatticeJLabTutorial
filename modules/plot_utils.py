import matplotlib.pyplot as plt
import numpy as np
import gmpy2

def plot_persite_combined(result,nQ,nL,mylabels,doreverse):
    '''
    result is a list of lists of probabilities for the 2^(nQ*nL) states.  For example, if have only one prediction, then result will be [[Pr(000..000),Pr(000..001),...]].
    #nQ is the number of qubits per lattice site
    #nL is the number of lattice sites
    #mylabels is an array of labels for each element in result.
    #doreverse is an array of booleans for each element in result.  Qiskit results should have True while everything else should have False (Qiskit labels states "backwards")

    '''

    f = plt.figure(figsize=(5,5))
    plt.ylabel("Pr(site)")

    xlabs = []
    for i in range(nL):
        plt.text(2**nQ*i+2**nQ/2-0.5,-0.1,"site "+str(i),horizontalalignment='center')
        if (i > 0):
            plt.axvline(2**nQ*i-0.5,color='grey',ls=":")
        for j in range(2**nQ):
            xlabs += [r'$|'+bin(j)[2:].zfill(nQ)+r'\rangle$']
            pass
        pass

    forplot = np.zeros([nL*2**nQ,len(result)])
    for k in range(len(result)):
        for i in range(nL):
            for j in range(2**(nQ*nL)):
                qval = int(gmpy2.digits(j,2**nQ).zfill(nL)[i])
                forplot[qval+i*2**nQ,k]+=result[k][j]
            pass
        pass
    plt.xticks(range(nL*2**nQ),xlabs,rotation='vertical',fontsize=10,horizontalalignment='center')

    mycolors = ["grey","orange","black"]
    mylinestyles = ["-","-",":"]
    myalphas = [0.5,1,1]
    for k in range(3,len(result)):
        mycolors+=["black"]
        mylinestyles+=[":"]
        myalphas+=[1.]
        pass
    
    for k in range(len(result)):
        if (doreverse[k]):
            plt.plot(forplot[:,k][::-1],label=mylabels[k],color=mycolors[k],ls=mylinestyles[k],alpha=myalphas[k]) #annoyingly, Qiskit counts backwards
        else:
            plt.plot(forplot[:,k],label=mylabels[k],color=mycolors[k],ls=mylinestyles[k],alpha=myalphas[k])
            pass
        pass
    plt.legend()
