import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
##

N = 30


for mean,var,c in [ \
        ( [2,2], 0.7,'red' ),\
        ( [0,0], 0.7 ,'green'),\
        ( [-2,3], 0.9 ,'blue') ]:
    points =  np.random.normal(size=(N,2))*var + mean 
    plt.scatter(points[:,0],points[:,1],c=c)
    
plt.axis('off')
plt.savefig('clusters.pgf')


    
## Plot



