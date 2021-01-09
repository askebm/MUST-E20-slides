import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
##

N=200

X = np.random.normal(size=N)
Y = X*3 + np.random.normal(size=N)*1.5

D = np.array([X,Y]).T

##

df = pd.DataFrame(data=D)

## Plotting
#scatter
pd.plotting.scatter_matrix(df)
plt.savefig('scatter_matrix.pgf')
plt.show()

#Probability
ss.probplot(X,plot=plt)
ss.probplot(Y,plot=plt)
plt.xlabel("")
plt.savefig('qq.pgf')
plt.show()


