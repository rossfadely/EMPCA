"""
Dumbass demo of EMPCA...
"""

import numpy as np
import matplotlib.pyplot as pl

from empca import *

# 2D, zero mean, with simple covariance
M = np.zeros(2)
C = np.eye(2)
C[0] = 100 
d = np.random.multivariate_normal(M,C,1000)

# one latent dimension
model1 = EMPCA(d.T,1)
# projection
x1 = np.dot(model1.lam,model1.lat).T

# two (full) latent dimensions
model2 = EMPCA(d.T,2)
# projection
x2 = np.dot(model2.lam,model2.lat).T

xs, xe = 0.05, 0.975
xb = 0.025
xext = (xe-xs-3*xb)/3
ys = 0.1
ax1 = [xs,ys,xext,xext*3]
ax2 = [xs+xext+xb,ys,xext,xext*3]
ax3 = [xs+2*xext+2*xb,ys,xext,xext*3]


fig = pl.figure(figsize=(12,4))
ax = pl.axes(ax1)
ax.plot(d[:,0],d[:,1],'bo',alpha=0.3,label='Data')
ax.legend(loc=2)
ax = pl.axes(ax2)
ax.plot(d[:,0],d[:,1],'bo',alpha=0.3,label='Data')
ax.plot(x1[:,0],x1[:,1],'ro',alpha=0.3,label='1D Proj.')
ax.set_yticks([])
ax.legend(loc=2)
ax = pl.axes(ax3)
ax.plot(d[:,0],d[:,1],'bo',alpha=0.3,label='Data')
ax.plot(x2[:,0],x2[:,1],'ro',alpha=0.3,label='2D Proj.')
ax.set_yticks([])
ax.legend(loc=2)
fig.savefig('demo.png')
