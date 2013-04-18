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
    tol  : Tolerance criterion for delta squared error.
    max_iter   : Maximum number of iterations to run EM.
    check_iter : Interval number of EM iterations betwee
                 convergence checks.

    To Do:
    ------
    - Automatic latent dimensionality determination.
    
    """
    def __init__(self,data,M,var=None,
                 max_iter=1000,check_iter=10,
                 tol=1.e-4):
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
        self.lam = np.random.randn(self.D,self.M)

        # zero mean the data
        self.zero_mean(var)

        # run EM
        self.run_EM(max_iter,tol,check_iter)

    def zero_mean(self,var):
        """
        Subtract the mean, use weighted mean if variance 
        is given.
        """
        if var!=None:
            var  = np.atleast_2d(var)
            mean = np.sum(self.data / var, axis=0) / \
                np.sum(1.0 / var, axis=0)
        else:
            mean = np.mean(self.data, axis=0)

    def run_EM(self,max_iter,tol,check_iter):
        """
        Run Expectation-Maximization
        """
        i = 0
        sq_error = np.Inf
        dt_error = np.Inf
        while ((i < max_iter) & (dt_error>tol)):
            
            self.run_E_step()
            self.run_M_step()
            if np.mod(i,check_iter)==0:
                sq_error_new = self.total_squared_error()
                dt_error = sq_error - sq_error_new
                sq_error = sq_error_new
            i += 1

    def run_E_step(self):
        """
        Single Expectation step
        """
        lamTdata = np.dot(self.lam.T,self.data)
        lamTlamI = np.linalg.inv(np.dot(self.lam.T,self.lam))
        self.lat = np.dot(lamTlamI,lamTdata)

    def run_M_step(self):
        """
        Single Maximization step
        """
        latlatTI = np.linalg.inv(np.dot(self.lat,self.lat.T))
        datalatT = np.dot(self.data,self.lat.T)
        self.lam = np.dot(datalatT,latlatTI)

    def project(self):
        """
        Project the latents into the data space.
        """
        self.projections = np.dot(self.lam,self.lat)

    def total_squared_error(self):
        """
        Calculate the total squared error of the model.
        """
        self.project()
        return np.sum((self.data - self.projections)**2.)
