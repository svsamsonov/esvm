import numpy as np
import scipy.optimize as opt
from multiprocessing import Pool
import multiprocessing
import copy
from scipy.stats import rv_continuous
import scipy.stats as spstats

class poly_dens(rv_continuous):
    def _pdf(self,x):
        return np.sqrt(2)/(np.pi*(1+x**4))

def compute_sqrt(Sigma):
    S,V = np.linalg.eig(Sigma)
    Sigma_half = V.dot(np.diag(np.sqrt(S)).dot(V.T))
    return Sigma_half
    
"""
here density is assumed to be e^{-U(x)},
U(x) = 0.5*<\Sigma^{-1}(x-\mu),x-\mu> + |x - \mu|^{\alpha}
"""
class IndependentPotential:
    """
    Used to get classical Monte-Carlo estimates for given density type
    """
    def __init__(self,typ,params):
        self.params = params
        self.typ = typ
        self.d = params["d"]
        
    def sample(self,rand_seed,N):
        """
        Samples N observations with given random seed
        """
        np.random.seed(rand_seed)
        if self.typ == "Normal":
            mu = self.params["mu"]
            Sigma = self.params["Sigma"]
            if self.d > 1:
                Sigma_half = compute_sqrt(Sigma)
                traj = np.random.randn(N,self.d)
                traj = traj.dot(Sigma_half)
            else: #1-dimensional example
                sigma_half = np.sqrt(Sigma)
                traj = sigma_half*np.random.randn(N,1)
            traj += mu.reshape((1,self.d))
            traj_grad = self.gradpotential(traj)
        elif self.typ == "Laplace":
            mu = self.params["mu"]
            l = self.params["lambda"]
            traj = np.random.laplace(loc = mu, scale = l, size = (N,self.d))
            traj_grad = self.gradpotential(traj)
        elif self.typ == "Cauchy":
            traj = np.random.standard_cauchy((N,self.d))
            traj_grad = self.gradpotential(traj)
        elif self.typ == "Pareto":
            b = self.params["b"]
            rv = spstats.pareto(b)
            traj = rv.rvs(size = (N,self.d))
            traj_grad = self.gradpotential(traj)
        elif self.typ == "3rd_poly":
            #here we will use implicitly the generation by inverse cdf
            traj = np.random.rand(N,self.d)
            traj = np.sqrt(np.abs(np.tan(np.pi*(traj-0.5))))*np.sign(traj-0.5)
            traj_grad = self.gradpotential(traj)
        elif self.typ == "Poly":
            sample_class = poly_dens()
            traj = sample_class.rvs(size = (N,self.d))
            traj_grad =self.gradpotential(traj)
        else:
            raise "Not implemented error in IndependentPotential::sample"
        return traj,traj_grad
    
    def gradpotential(self,X):
        """
        Evaluates gradients of log-density at points X
        Args:
            X - np.array of shape (N,d)
        Outputs:
            X_grad - np.array of shape (N,d)
        """
        if self.typ == "Normal":
            mu = self.params["mu"]
            Sigma = self.params["Sigma"]
            if self.d > 1:
                Sigma_inv = np.linalg.inv(Sigma)
                return -(X - mu.reshape((1,self.d))).dot(Sigma_inv)
            else: #1d case
                return -(X - mu.reshape((1,self.d)))/Sigma
        elif self.typ == "Laplace":
            mu = self.params["mu"]
            l = self.params["lambda"]
            return -np.sign(X - mu)/l
        elif self.typ == "3rd_poly":
            return 1.0/X - (4*X**3)/(1+X**4) 
        elif self.typ == "Pareto":
            b = self.params["b"]
            return -(b+1)/X
        elif self.typ == "Cauchy":
            return -2*X/(1+X**2)
        elif self.typ == "Poly":
            return -4*X**3/(1 + X**4)
        else:
            raise "Not implemented error in IndependentPotential::gradpotential"
        
    
    
class GaussPotential:
    """
    implements pseudo-gaussian potential function
    arguments:
    mu - mode;
    Sigma - covariates;
    alpha - degree of summand (note that it should be between 1 and 2 if we are willing to justify it theoretically)
    typ - type of density:
        "g" - pure gaussian - without second summand
        "m" - mixed - geenral case
    """
    def __init__(self,Sigma,mu,alpha,typ):
        self.mu = mu
        self.S_inv = np.linalg.inv(Sigma)
        self.alpha = alpha
        self.typ = typ
        self.dim = self.S_inv.shape[0]
        
    def potential(self,x):
        """
        returns log-density at point x
        """
        if self.typ == "g":
            return -0.5*np.dot(self.S_inv @ (x-self.mu), x-self.mu)
        else:
            return -0.5*np.dot(self.S_inv @ (x-self.mu), x-self.mu) - np.power(np.linalg.norm(x-self.mu),self.alpha)
    def gradpotential(self,x):
        """
        returns gradient of log-density at point x
        """
        if self.typ == "g":
            return -self.S_inv @ (x-self.mu) 
        else:
            return -self.S_inv @ (x-self.mu) - \
                self.alpha*np.power(np.linalg.norm(x-self.mu),self.alpha-1)*(x-self.mu)/np.linalg.norm(x-self.mu)    
    def vec_grad(self,X):
        """
        returns vector of gradients at point x 
        """
        return -np.dot(X - self.mu.reshape(1,self.dim),self.S_inv)
        
#################################################################################################################################
class GaussMixture:
    """
    implements gaussian mixture potential function
    arguments:
        Sigma_1,Sigma_2 - covariates;
        mu_1,mu_2 - means;
        p - probability of getting into cluster 1;
    """
    def __init__(self,Sigma_1,Sigma_2,mu_1,mu_2,p):
        self.p = p
        self.S1 = np.linalg.inv(Sigma_1)
        self.det_s1 = np.sqrt(np.linalg.det(Sigma_1))
        self.dim = self.S1.shape[0]
        self.mu_1 = mu_1
        self.mu_1_vec = mu_1.reshape(1,self.dim)
        self.mu_2 = mu_2
        self.mu_2_vec = mu_2.reshape(1,self.dim)
        self.S2 = np.linalg.inv(Sigma_2)
        self.det_s2 = np.sqrt(np.linalg.det(Sigma_2))
        self.eps = 1e-10
        
    def gradpotential(self,x):
        """
        returns gradient of log-density at point x
        """
        #mu_1 = self.mu_1.ravel()
        #mu_2 = self.mu_2.ravel()
        numer = -self.p*self.S1 @ (x-self.mu_1)*np.exp(-np.dot(self.S1 @ (x - self.mu_1),x-self.mu_1)/2)/self.det_s1 -\
                (1-self.p)*self.S2 @ (x-self.mu_2)*np.exp(-np.dot(self.S2 @ (x - self.mu_2),x-self.mu_2)/2)/self.det_s2
        denom = self.eps + self.p*np.exp(-np.dot(self.S1 @ (x - self.mu_1),x-self.mu_1)/2)/self.det_s1 +\
                (1-self.p)*np.exp(-np.dot(self.S2 @ (x - self.mu_2),x-self.mu_2)/2)/self.det_s2
        return numer/denom 
    
    def potential(self,x):
        """
        returns log-density
        """
        return np.log(self.p*np.exp(-np.dot(self.S1 @ (x - self.mu_1),x-self.mu_1)/2)/self.det_s1 + (1-self.p)*np.exp(-np.dot(self.S2 @ (x - self.mu_2),x-self.mu_2)/2))
    
    def vec_val(self,X):
        """
        returns vector of density values at each point X[i,:]
        Arguments:
            X - np.array of shape (n,d)
        returns:
            np.array of shape (n)
        """
        clust_1 = self.p*np.exp(-qform_q(self.S1,X-self.mu_1,X-self.mu_1)/2)/self.det_s1
        clust_2 = (1-self.p)*np.exp(-qform_q(self.S2,X-self.mu_2,X-self.mu_2)/2)/self.det_s2
        return clust_1 + clust_2
        
    def vec_grad(self,X):
        """
        returns vector of gradients at each point X[i,:]
        Arguments:
            X - np.array of shape (n,d)
        returns:
            X_grad - np.array of shape (n,d), gradients
        """
        S1 = self.S1
        mu_1 = self.mu_1_vec
        S2 = self.S2
        mu_2 = self.mu_2_vec
        p = self.p
        n = X.shape[0]
        #clusters with same covariance structure
        if self.homogen:
            numer = -p*np.exp(-np.dot(mu_1.ravel(),S1 @ mu_1.ravel())/2)*(X-mu_1).dot(S1)*np.exp(X@(S1 @ mu_1.ravel())).reshape((n,1)) -\
                (1-p)*np.exp(-np.dot(mu_2.ravel(),S2 @ mu_2.ravel())/2)*(X-mu_2).dot(S2)*np.exp(X @ (S2 @ mu_2.ravel())).reshape((n,1))
            denom = self.lin_vec_val(X) + self.eps
        #different covariance
        else:
            numer = -p*(X-mu_1).dot(S1)*np.exp(-qform_q(S1,X-mu_1,X-mu_1)).reshape((n,1))/self.det_s1 -\
                (1-p)*(X-mu_2).dot(S2)*np.exp(-qform_q(S2,X-mu_2,X-mu_2)).reshape((n,1))/self.det_s2
            denom = self.vec_val(X) + self.eps    
        return numer / denom.reshape((n,1))
#################################################################################################################################
class GausMixtureIdent:
    """
    Implements Gaussian Mixture potential function for identity covariance structure
    with probability p: 1st cluster (with mean \mu)
    with probability 1-p: 2nd cluster (with mean -\mu)
    """
    def __init__(self,mu,p):
        self.p = p
        self.mu = mu  
        
    def potential(self, x):
        """
        returns log-density 
        """
        d = len(x)
        return -np.log(2*np.pi)*d/2 -0.5*np.sum((x-self.mu)**2) + np.log(self.p + (1-self.p)*np.exp(-2*np.dot(self.mu,x)))
        
    def gradpotential(self,x):
        """
        returns gradient of log-density at point x
        """
        return self.mu - x - 2*(1-self.p)*self.mu/(1-self.p + self.p*np.exp(2*np.dot(self.mu,x)))

#################################################################################################################################
class GausMixtureSame:
    """
    Implements Gaussian Mixture potential function for equal covariance structure in both clusters
    with probability p: 1st cluster (with mean \mu)
    with probability 1-p: 2nd cluster (with mean -\mu)
    """
    def __init__(self,Sigma,mu,p):
        self.p = p
        self.mu = mu
        self.Sigma_inv = np.linalg.inv(Sigma)
        self.det_sigma = np.linalg.det(Sigma)
        
    def potential(self, x):
        """
        returns log-density at x
        """
        d = len(x)
        return -np.log(2*np.pi)*d/2 - np.log(self.det_sigma)/2 - 0.5*np.dot(x-self.mu, self.Sigma_inv @ (x-self.mu)) + np.log(self.p + (1-self.p)*np.exp(-2*np.dot(self.mu,self.Sigma_inv @ x)))
        
    def gradpotential(self,x):
        """
        returns gradient of log-density at point x
        """
        return self.Sigma_inv @ (self.mu - x) - 2*(1-self.p)*self.Sigma_inv @ self.mu/(1-self.p + self.p*np.exp(2*np.dot(self.mu,self.Sigma_inv @ x)))
#################################################################################################################################
    
class BananaShape:
    """
    Implements Banana-shaped density potential function in R^2 for density f(x,y) = \exp{-\frac{x^2}{2M} - \frac{1}{2}(y+Bx^2-100B)^2}
    """
    def __init__(self,B,M,d=2):
        self.B = B
        self.M = M
        self.d = d
        
    def potential(self,z):
        """
        returns log-density at z
        """
        x = z[0]
        y = z[1]
        M = self.M
        B = self.B
        exponent = -1./(2*M)*x**2 - 1./2*(y+B*x**2-M*B)**2
        for i in range(2,self.d):
            exponent -= 1./2*z[i]**2
        return exponent
    
    def gradpotential(self,z):
        """
        returns gradient of log-density at point z
        """
        x = z[0]
        y = z[1]
        M = self.M
        B = self.B
        grad = np.zeros(self.d, dtype = float)
        grad[0] = -x/M-(y+B*x**2-M*B)*2*B*x
        grad[1] = -y-B*x**2+M*B
        for i in range(2,self.d):
            grad[i] = -z[i]
        return grad
################################################################################################################################ #### 
class potentialRegression:
    """ implementing a potential U = logarithm of the posterior distribution
        given by a Bayesian regression
     - Linear
     - Logistic
     - Probit
    """
    varY = 1 # Variance of the linear likelihood
    varTheta = 100 # Variance of the prior Gaussian distribution
    def __init__(self,Y,X,typ, print_info = False):
        """ initialisation 
        Args:
            Y: observations
            X: covariates
            typ: type of the regression, Linear, Logistic or Probit
        """
        self.Y = Y
        self.X = X
        self.type = typ  
        self.p, self.d = X.shape
        self.dim = self.d
    
    def hess_potential_determ(self,theta):
        """Second-order optimization to accelerate optimization and (possibly) increase precision
        """
        XTheta = self.X @ theta
        term_exp = np.divide(np.exp(-XTheta/2),1 + np.exp(XTheta))
        X_add = self.X*term_exp.reshape((self.p,1))
        #second summand comes from prior
        return np.dot(X_add.T,X_add) + 1./self.varTheta
        
    def loglikelihood(self,theta):
        """loglikelihood of the Bayesian regression
        Args:
            theta: parameter of the state space R^d where the likelihood is
                evaluated
        Returns:
            real value of the likelihood evaluated at theta
        """
        if self.type == "linear": # Linear regression
            return -(1. / (2*self.varY))* np.linalg.norm(self.Y-np.dot(self.X,theta))**2 \
                        - (self.d/2.)*np.log(2*np.pi*self.varY)
        elif self.type == "logistic": # Logistic
            XTheta = np.dot(-self.X, theta)
            temp1 = np.dot(1.0-self.Y, XTheta)
            temp2 = -np.sum(np.log(1+np.exp(XTheta)))
            return temp1+temp2
        else: # Probit
            cdfXTheta = spstats.norm.cdf(np.dot(self.X, theta))
            cdfMXTheta = spstats.norm.cdf(-np.dot(self.X, theta))
            temp1 = np.dot(self.Y, np.log(cdfXTheta))
            temp2 = np.dot((1 - self.Y), np.log(cdfMXTheta))
            return temp1+temp2
    
    def gradloglikelihood_determ(self,theta):
        """Purely deterministic gradient of log-likelihood, used for theta^* search
        Returns:
            R^d vector of the (full and fair) gradient of log-likelihood, evaluated at theta^*
        """
        if self.type == "linear": # Linear
            temp1 = np.dot(np.dot(np.transpose(self.X), self.X), theta)
            temp2 = np.dot(np.transpose(self.X), self.Y)
            return (1. / self.varY)*(temp2 - temp1)
        elif self.type == "logistic": # Logistic
            temp1 = np.exp(np.dot(-self.X, theta))
            temp2 = np.dot(np.transpose(self.X), self.Y)
            temp3 = np.dot(np.transpose(self.X), np.divide(1, 1+temp1))
            return temp2 - temp3
        else: # Probit
            XTheta = np.dot(self.X, theta)
            logcdfXTheta = np.log(spstats.norm.cdf(XTheta))
            logcdfMXTheta = np.log(spstats.norm.cdf(-XTheta))
            temp1 = np.multiply(self.Y, np.exp(-0.5*(np.square(XTheta)+np.log(2*np.pi)) \
                                               -logcdfXTheta))
            temp2 = np.multiply((1 - self.Y), np.exp(-0.5*(np.square(XTheta)+np.log(2*np.pi)) \
                                               -logcdfMXTheta))
            return np.dot(np.transpose(self.X), temp1-temp2)
            
        
    def logprior(self, theta):
        """ logarithm of the prior distribution, which is a Gaussian distribution
            of variance varTheta
        Args:
            theta: parameter of R^d where the log prior is evaluated
        Returns:
            real value of the log prior evaluated at theta
        """
        return -(1. / (2*self.varTheta))* np.linalg.norm(theta)**2  \
                - (self.d/2.)*np.log(2*np.pi*self.varTheta)
    
    def gradlogprior(self, theta):
        """ gradient of the logarithm of the prior distribution, which is 
            a Gaussian distribution of variance varTheta
        Args:
            theta: parameter of R^d where the gradient log prior is evaluated
        Returns:
            R^d vector of the gradient of the log prior evaluated at theta
        """
        return -(1. / self.varTheta)*theta
    
    def potential(self, theta):
        """ logarithm of the posterior distribution
        Args:
            theta: parameter of R^d where the log posterior is evaluated
        Returns:
            real value of the log posterior evaluated at theta
        """
        return self.loglikelihood(theta)+self.logprior(theta)
    
    def minus_potential(self,theta):
        """Actually, a very silly function. Will re-write it later
        """
        return -self.potential(theta)
    
    def gradpotential(self,theta):
        """full gradient of posterior
        """
        return self.gradloglikelihood_determ(theta) + self.gradlogprior(theta)
    
    def gradpotential_deterministic(self,theta):
        """
        A bit strange implementation of always deterministic gradient, this one is needed for fixed point search
        """
        return -self.gradloglikelihood_determ(theta) - self.gradlogprior(theta)
         