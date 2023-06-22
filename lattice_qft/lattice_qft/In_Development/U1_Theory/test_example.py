import numpy as np
import matplotlib.pyplot as plt 


dx = 1
nL = 4
dimension = 2
x,y = np.arange(0,4,1),np.arange(0,4,1)
print(x)
grid = np.meshgrid(x,y)
def link(coord,direction):
    new_coord = np.array(coord)%nL
    new_coord[direction] = new_coord[direction]+dx
    return np.array([coord,new_coord])


X,Y = grid

link_test =link([X[0][0],Y[0][0]],1)
print(link_test)

plt.plot(link_test[0],link_test[1],color='cyan',marker='.')
plt.scatter(X,Y,color='r', marker='x')
plt.grid(True)
plt.show()




