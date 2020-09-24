#
# Bayesian variable selection
# (c) N. Packham, 2019
#

#
# Bayesian variable selection for linear regression model with normal inverse Gaussian prior
#

import numpy as np
import scipy as sp
import scipy.stats as scs
import matplotlib.pyplot as plt
import math
from timeit import default_timer as timer

#
# Normal inverse Gaussian PDF
#

def nig_pdf(x, sigsq, mu, Minv, a, b):
    p = len(mu)
    x0 = np.array(x)
    mu0 = np.array(mu)
    return np.linalg.det(Minv)**(0.5) * (2*np.pi)**(-p/2) * b**a / sp.special.gamma(a) * (1/sigsq)**(a + 1 + p/2) \
      * np.exp(-(2*b + (x0-mu0).transpose().dot(Minv.dot(x0-mu0)))/(2*sigsq))


class BayesModel(object):
    def __init__(self, Y, X, m, M, a, b):
        self.__Y=Y # expects a one-dim. array
        self.__X=X # expects a two-dim. array
        self.__m=m # one-dim. array
        self.__M=M # two-dim array
        self.__a=a
        self.__b=b
        self.__n = max(self.__X.shape)
        self.__p = min(self.__X.shape)
        self.__Minv = sp.linalg.inv(self.__M)
        self.__tM = sp.linalg.inv((self.__X.transpose()).dot(self.__X) + self.__Minv)
        self.__tMinv = (self.__X.transpose().dot(self.__X) + self.__Minv)
        self.__tm = self.__tM.dot(self.__Minv.dot(self.__m) + (self.__X.transpose()).dot(self.__Y))
        self.__ta = self.__a + self.__n/2
        self.__tb = self.__b + 0.5 * ((self.__Y.transpose()).dot(self.__Y) \
                + (self.__m.transpose()).dot(self.__Minv.dot(self.__m)) \
                - (self.__tm.transpose()).dot(self.__tMinv.dot(self.__tm)))

    def get_tM(self):
        return self.__tM

    def get_ta(self):
        return self.__ta

    def get_tb(self):
        return self.__tb
                

    def prior_pdf(self, beta, sigsq):
        if(len(beta) != self.__p):
            return np.nan
        return nig_pdf(beta, sigsq, self.__m, self.__Minv, self.__a, self.__b)

    
    def prior_marginal_pdf(self, x, k):
        if(k>self.__p):
            return np.nan
        if(k==self.__p):
            return scs.invgauss.pdf(x, self.__a, self.__b)
        return scs.t.pdf(x, 2*self.__a, loc=self.__m[k], scale=np.sqrt(((self.__b/self.__a) * self.__M)[k,k]))

    
    def posterior_pdf(self, beta, sigsq):
        if(len(beta) != self.__p):
            return np.nan
        return nig_pdf(beta, sigsq, self.__tm, self.__tMinv, self.__tb, self.__tb)

    
    def posterior_marginal_pdf(self, x, k):
        if(k>self.__p):
            return np.nan
        if(k==self.__p):
            return scs.invgauss.pdf(x, self.__ta, self.__tb)
        return scs.t.pdf(x, 2*self.__ta, loc=self.__tm[k], scale=np.sqrt(((self.__tb/self.__ta) * self.__tM)[k,k]))

    
    def posterior_mean(self):
        return self.__tm

    
    def posterior_hpd(self):
        result = np.zeros((p+1, 2))
        for k in range(p):
            result[0,k] = scs.t.ppf(0.025, 2*self.__ta, loc = self.__tm[k], \
                                        scale = np.sqrt(((self.__tb/self.__ta) * self.__tM)[k,k]))
            result[1,k] = scs.t.ppf(0.975, 2*self.__ta, loc = self.__tm[k], \
                                        scale = np.sqrt(((self.__tb/self.__ta) * self.__tM)[k,k]))
        result[0,p] = scs.invgauss.ppf(0.025, self.__ta, self.__tb)
        result[1,p] = scs.invgauss.ppf(0.975, self.__ta, self.__tb)
        return result

    def marginal_likelihood(self):
        return 1/(2*np.pi)**(self.__n/2)  * np.sqrt(np.linalg.det(self.__tM) / np.linalg.det(self.__M)) \
          * self.__b**self.__a / self.__tb**self.__ta * math.gamma(self.__ta) / math.gamma(self.__a)


    # avoids math overflow du to large constants that cancel out in MCMC anyway
    def marginal_likelihood_wo_constants(self):
        q = 0.5 * (np.log(np.linalg.det(self.__tM)) - np.log(np.linalg.det(self.__M)) \
                          + self.__a * np.log(self.__b) - self.__ta* np.log(self.__tb))
        return np.exp(q)
#        return np.sqrt(np.linalg.det(self.__tM) / np.linalg.det(self.__M)) \
#                * self.__b**self.__a / self.__tb**self.__ta

    def log_marginal_likelihood_wo_constants(self):
        q = 0.5 * (np.log(np.linalg.det(self.__tM)) - np.log(np.linalg.det(self.__M)) \
                          + self.__a * np.log(self.__b) - self.__ta* np.log(self.__tb))
        return q

    def model_prior_constant(self, sc, theta=0.5): # pi denotes the number of independent variables in the model (i.e., the number of coefficients to be estimated
#        if(np.isscalar(theta)):
#            s = np.where(sc[0]==1)[0]
#            pi = s.size            
#            return theta**(pi) * (1-theta)**(self.__p - pi)
        return np.prod(theta**sc * (1-theta)**(1-sc))


    def model_prior_beta(self, pi, a=1, b=1): # expectation of beta distribution: a/(a+b) = 1/2 in default case
        return sp.special.gamma(a+b) / (sp.special.gamma(a)*sp.special.gamma(b)) \
          * (sp.special.gamma(a+pi) * sp.special.gamma(b + self.__p - pi)) / sp.special.gamma(a + b + self.__p)
        

    def simulate_posterior_probability(self, nsimulations=500, rel=False, theta=0.5):
        m = nsimulations
        x = np.zeros(m)
        sc = np.zeros((m,self.__p))
        sc[0] = sp.stats.bernoulli.rvs(0.5, size=self.__p)
        z = scs.randint.rvs(0, self.__p, size=m) # indices to flip
        u = scs.uniform.rvs(size=m)
        s = np.where(sc[0]==1)[0]
        X2 = self.__X[:, s]
                  
        factor = self.log_marginal_likelihood_wo_constants() if rel else 0

        if X2.shape[1]>0:
            bm = BayesModel(self.__Y, X2, self.__m[s], self.__M[s,:][:,s], self.__a, self.__b)
            x[0] = np.exp(bm.log_marginal_likelihood_wo_constants()-factor) * self.model_prior_constant(sc[0], theta)
        else:
            x[0]=0
        ml_old = x[0]
        for i in range(1,m):
            sc[i] = sc[i-1]
            sc[i,z[i]] = 1 - sc[i-1,z[i]] # flip gamma at random position
            s = np.where(sc[i]==1)[0]
            X2 = self.__X[:, s]
            if X2.shape[1]>0:
                bm = BayesModel(self.__Y, X2, self.__m[s], self.__M[s,:][:,s], self.__a, self.__b)
                ml = np.exp(bm.log_marginal_likelihood_wo_constants()-factor)  * self.model_prior_constant(sc[i], theta)
            else:
                ml = 0
            alpha = ml / ml_old if ml_old > 0 else 1
            x[i] = ml if u[i] <= alpha else x[i-1]
            sc[i] = sc[i] if u[i] <= alpha else sc[i-1]
            ml_old = x[i]
        return x, sc


    def simulate_posterior_probability_beta(self, nsimulations=500, rel=False, a=1, b=1):
        m = nsimulations
        x = np.zeros(m)
        sc = np.zeros((m,self.__p))
        sc[0] = sp.stats.bernoulli.rvs(0.5, size=self.__p)
        z = scs.randint.rvs(0, self.__p, size=m) # indices to flip
        u = scs.uniform.rvs(size=m)
        s = np.where(sc[0]==1)[0]
        X2 = self.__X[:, s]
                  
        factor = self.log_marginal_likelihood_wo_constants() if rel else 0

        if X2.shape[1]>0:
            bm = BayesModel(self.__Y, X2, self.__m[s], self.__M[s,:][:,s], self.__a, self.__b)
            x[0] = np.exp(bm.log_marginal_likelihood_wo_constants()-factor) * self.model_prior_beta(s.size, a, b)
        else:
            x[0]=0
        for i in range(1,m):
            sc[i] = sc[i-1]
            sc[i,z[i]] = 1 - sc[i-1,z[i]] # flip gamma at random position
            s = np.where(sc[i]==1)[0]
            X2 = self.__X[:, s]
            if X2.shape[1]>0:
                bm = BayesModel(self.__Y, X2, self.__m[s], self.__M[s,:][:,s], self.__a, self.__b)
                ml = np.exp(bm.log_marginal_likelihood_wo_constants()-factor)  * self.model_prior_beta(s.size, a, b)
            else:
                ml = 0
            alpha = ml / x[i-1] if x[i-1] > 0 else 1
            x[i] = ml if u[i] <= alpha else x[i-1]
            sc[i] = sc[i] if u[i] <= alpha else sc[i-1]
        return x, sc

    
    def log_gamma_posterior(self, nu, lam, tY, D, R, v0, v1, sc):
        n = self.__n
        p = self.__p
        for k in range(p):
            D[k,k] = v1[k] * sc[k] + v0[k] * (1-sc[k]) 
        # start = timer()
        # D2 = sp.linalg.sqrtm(np.linalg.inv(D.dot(R).dot(D)))
        D2 = sp.linalg.fractional_matrix_power(D.dot(R).dot(D), -0.5)
        # print ("Simulated in in %f s" % (timer() - start))
        tX = np.concatenate((self.__X, D2))
        Ssq = tY.transpose().dot(tY) - \
          tY.transpose().dot(tX).dot(np.linalg.inv(tX.transpose().dot(tX))).dot(tX.transpose().dot(tY))
        log_pi = (-0.5) * np.log(np.linalg.det(tX.transpose().dot(tX))) \
          + (-0.5) * np.log(np.linalg.det(D.dot(R.dot(D)))) \
         -(n+nu)/2 * np.log(nu * lam + Ssq) + p * np.log(0.5)
        return log_pi

    
    def gamma_prior(self, w, sc):
        return ((w**sc) * (1-w)**(1-sc)).prod()

    
    def simulate_gamma_posterior(self, nu, lam, v0, v1, theta=0.5, nsimulations=500, rel=False):
        m = nsimulations
        x = np.zeros(m)
        sc = np.zeros((m,self.__p))
        sc[0] = sp.stats.bernoulli.rvs(0.5, size=self.__p)
        sc[0] = np.ones(self.__p)
        z = scs.randint.rvs(0, self.__p, size=m) # indices to flip
        u = scs.uniform.rvs(size=m)

        v0 = v0 * np.ones(self.__p) if np.isscalar(v0) else v0
        v1 = v1 * np.ones(self.__p) if np.isscalar(v1) else v1
        
        tY = np.concatenate((self.__Y, np.zeros(self.__p)))
        R = np.identity(self.__p)
        D = np.identity(self.__p)
        factor = self.log_gamma_posterior(nu, lam, tY, D, R, v0, v1, np.ones(self.__p)) if rel else 0
        log_pi = self.log_gamma_posterior(nu, lam, tY, D, R, v0, v1, sc[0])

        x[0] = 0 if np.all(sc[0]==0) else np.exp(log_pi-factor) * self.gamma_prior(theta, sc[0])
        for i in range(1,m):
            sc[i] = sc[i-1]
            sc[i,z[i]] = 1 - sc[i-1,z[i]] # flip gamma at random position
            log_pi = self.log_gamma_posterior(nu, lam, tY, D, R, v0, v1, sc[i])
            ml = 0 if np.all(sc[i]==0) else np.exp(log_pi-factor) * self.gamma_prior(theta, sc[i])
            alpha = ml / x[i-1] if x[i-1] > 0 else 1
            x[i] = ml if u[i] <= alpha else x[i-1]
            sc[i] = sc[i] if u[i] <= alpha else sc[i-1]
        return x, sc
        
