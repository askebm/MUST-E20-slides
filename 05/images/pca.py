%reset -f
##

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
##

r=0.25

X = np.random.normal(size=1000)
Y = 3*X*r + np.random.normal(size=1000)
D = np.array([X,Y]).T


##

S = np.cov(D.T)
R = np.corrcoef(D.T)

y,E = np.linalg.eig(S)

Ei = E*y

ei0 = Ei[:,0]
ei1 = Ei[:,1]

#angle
a1 = np.arctan2(E[0,0],E[1,0])
a2 = np.arctan2(E[0,1],E[1,1])
print("Angle with eigen values: ",a1, " and ", a2, " ++ ",np.abs(a1)+np.abs(a2) )


a1 = np.arctan2(ei0[0],ei0[1])
a2 = np.arctan2(ei1[0],ei1[1])
print("Angle with eigen values: ",a1, " and ", a2 , " ++ ",np.abs(a1)+np.abs(a2) )

##
s = 1

plt.scatter(X,Y,s=1.8)
plt.axis('off')
plt.savefig('scatter_eigen.pgf')
plt.show()

plt.scatter(X,Y,s=1.8)
plt.arrow(0,0,s*ei0[0],s*ei0[1],color='red')
plt.arrow(0,0,s*ei1[0],s*ei1[1],color='red')
plt.gca().set_aspect(1)
plt.axis('off')
plt.savefig('scatter_eigen_arrow.pgf')
plt.show()

## Normalised


Xn = ( X -X.mean()) / np.sqrt(X.var())
Yn = ( Y -Y.mean()) / np.sqrt(Y.var())
Dn = np.array([Xn,Yn]).T
Sn = np.cov(Dn.T)
Rn = np.corrcoef(Dn.T)

yn,En = np.linalg.eig(Sn)

Ein = En*yn

#angle
a1 = np.arctan2(En[0,0],En[1,0])
a2 = np.arctan2(En[0,1],En[1,1])
print("Angle with eigen values: ",a1, " and ", a2 , " ++ ",np.abs(a1)+np.abs(a2) )


a1 = np.arctan2(Ein[0,0],Ein[1,0])
a2 = np.arctan2(Ein[0,1],Ein[1,1])
print("Angle with eigen values: ",a1, " and ", a2 , " ++ ",np.abs(a1)+np.abs(a2) )


##
plt.scatter(Xn,Yn,s=1.8)
plt.arrow(0,0,Ein[0,0],Ein[1,0],color='red')
plt.arrow(0,0,Ein[0,1],Ein[1,1],color='red')
plt.gca().set_aspect(1)
plt.axis('off')
plt.savefig('scatter_eigen_norm.pgf')
plt.show()

## PCA


plt.scatter(X,Y,s=1.8)
plt.arrow(0,0,Ein[0,0],Ein[1,0],color='lime')
plt.arrow(0,0,Ein[0,1],Ein[1,1],color='lime')
plt.arrow(0,0,s*ei0[0],s*ei0[1],color='red')
plt.arrow(0,0,s*ei1[0],s*ei1[1],color='red')
plt.gca().set_aspect(1)
plt.axis('off')
plt.savefig('scatter_eigen_pitfall1.pgf')
plt.show()


plt.scatter(Xn,Yn,s=1.8)
plt.arrow(0,0,Ein[0,0],Ein[1,0],color='lime')
plt.arrow(0,0,Ein[0,1],Ein[1,1],color='lime')
plt.arrow(0,0,ei0[0],ei0[1],color='red')
plt.arrow(0,0,ei1[0],ei1[1],color='red')
plt.gca().set_aspect(1)
plt.axis('off')
plt.savefig('scatter_eigen_pitfall2.pgf')
plt.show()

