import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
##

D = [ [ np.random.normal() , np.random.normal()*3 + 1] for i in range(100) ]
df = pd.DataFrame(data=D)

pd.plotting.scatter_matrix(df)


