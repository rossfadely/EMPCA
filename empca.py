# Author - Ross Fadely 

import numpy as np

class EMPCA(object):
    """
    PCA using Expectation-Maximization.

    http://www.cs.nyu.edu/~roweis/papers/empca.pdf

    Parameters
    ----------

    data : n_feature by n_sample array of the data
    M    : the number of latent dimensions.
    var  : Optional array of variances associated with
           the data.  Only used for mean estimation
           currently.
    max_iter : Number of iterations to run EM.

    To Do:
    ------
    - Automatic latent dimensionality determination.
    - Robust stopping criteria.
    
    """
    def __init__(self,data,M,max_iter=10,var=None):
        """
        `D` : Feature dimensionality
        `N` : Number of samples
        `M` : Latent dimensionality
        `lam` : Latent transformation matrix, shape D x M
        `lat` : Projection of latents, shape
        """
        self.D = data.shape[0]
        self.N = data.shape[1]
        self.M = M 
        self.data = np.atleast_2d(data)
        self.var = var
        self.lam = np.random.randn(self.D,self.M)

        # zero mean the data
        self.zero_mean()

        # run EM
        self.run_EM()

    def zero_mean(self):

        if self.var!=None:
            self.var = np.atleast_2d(self.var)
            mean = np.sum(self.data / self.var, axis=0) / \
                np.sum(1.0 / self.var, axis=0)
        else:
            mean = np.mean(self.data, axis=0)

    def run_EM(self, maxiter=10):

        i = 0
        while (i < maxiter) :
            
            self.run_E_step()
            self.run_M_step()
            i += 1

    def run_E_step(self):

        lamTdata = np.dot(self.lam.T,self.data)
        lamTlamI = np.linalg.inv(np.dot(self.lam.T,self.lam))
        self.lat = np.dot(lamTlamI,lamTdata)

    def run_M_step(self):

        latlatTI = np.linalg.inv(np.dot(self.lat,self.lat.T))
        datalatT = np.dot(self.data,self.lat.T)
        self.lam = np.dot(datalatT,latlatTI)
