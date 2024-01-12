import pandas as pd
import numpy as np
data=pd.read_csv("iris.csv")


y=data.values[:,-1]
x=data.values[:,:-1]



print(data.head())

cov=np.cov(x.transpose())
print(cov)

from numpy import linalg as la
eigenvalue,ev=la.eig(cov)
print(eigenvalue)
print(ev)

sorted_index=np.argsort(eigenvalue)[::-1]
sorted_index

eigenvalue=eigenvalue[sorted_index]
print(eigenvalue)
ev=ev[sorted_index]
print(ev)


ev_subset=ev[:,0:2]
print(ev_subset)

pca=np.dot(ev_subset.transpose(),x.transpose()).transpose()
pca

import matplotlib.pyplot as plt
plt.scatter(pca[:,0:1],pca[:,-1],c=y)
plt.xlabel("pca1")
plt.ylabel("pca2")
