import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
##

p = 4
n = 30

x = np.linspace(0,5,200)

y = np.array([ ss.f.pdf(i,p,n-p) for i in x])

##

plt.plot(x,y)
plt.grid()
plt.savefig('f.pgf')

