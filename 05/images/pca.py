import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
##

r=0.25

X = np.random.normal(size=1000,scale=100)
Y = X*r + np.random.normal(size=1000,scale=100)*(1-r)
D = np.array([X,Y]).T


##

S = np.cov(D.T)
R = np.corrcoef(D.T)

y,E = np.linalg.eig(R)


#angle
np.arctan2(E[0,0]-E[0,1],E[1,0]-E[1,1])

##

plt.scatter(X,Y)
#plt.plot(np.array([[0,0],y[0]*E[0]]))
#plt.plot(v1.T,color='red')
plt.arrow(0,0,y[0]*100*E[0,0],y[0]*100*E[1,0])
plt.arrow(0,0,y[1]*100*E[0,1],y[1]*100*E[1,1])
plt.show()

## Normalised


X = ( X -X.mean()) / np.sqrt(X.var())
Y = ( Y -Y.mean()) / np.sqrt(Y.var())
d = np.array([X,Y]).T


S = np.cov(D.T)
R = np.corrcoef(D.T)

y,E = np.linalg.eig(R)


#angle
np.arctan2(E[0,0]-E[0,1],E[1,0]-E[1,1])

##

plt.scatter(X*100,Y*100)
#plt.plot(np.array([[0,0],y[0]*E[0]]))
#plt.plot(v1.T,color='red')
plt.arrow(0,0,y[0]*100*E[0,0],y[0]*100*E[1,0])
plt.arrow(0,0,y[1]*100*E[0,1],y[1]*100*E[1,1])
