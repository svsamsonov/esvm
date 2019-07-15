import numpy as np
import scipy.optimize as opt
from multiprocessing import Pool
import multiprocessing
import copy
    
"""
here density is assumed to be e^{-U(x)},
U(x) = 0.5*<\Sigma^{-1}(x-\mu),x-\mu> + |x - \mu|^{\alpha}
"""
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
        if np.linalg.norm(Sigma_1 - Sigma_2) < 1e-5:
            self.homogen = True
        else:
            self.homogen = False
        self.eps = 1e-10
        
    def gradpotential(self,x):
        """
        returns gradient of log-density at point x
        """
        mu_1 = self.mu_1.ravel()
        mu_2 = self.mu_2.ravel()
        numer = -self.p*self.S1 @ (x-mu_1)*np.exp(-np.dot(self.S1 @ (x - mu_1),x-mu_1)/2)/self.det_s1 -\
                (1-self.p)*self.S2 @ (x-mu_2)*np.exp(-np.dot(self.S2 @ (x - mu_2),x-mu_2)/2)/self.det_s2
        denom = self.eps + self.p*np.exp(-np.dot(self.S1 @ (x - mu_1),x-mu_1)/2)/self.det_s1 +\
                (1-self.p)*np.exp(-np.dot(self.S2 @ (x - mu_2),x-mu_2)/2)/self.det_s2
        return numer/denom 
               
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
    
    def lin_vec_val(self,X):
        """
        same without quadratic part, which vanishes in case of same covariance structure
        """
        mu_1 = self.mu_1
        mu_2 = self.mu_2
        clust_1 = self.p*np.exp(-np.dot(mu_1,self.S1 @ mu_1)/2)*np.exp(X @ (self.S1 @ mu_1))
        clust_2 = (1-self.p)*np.exp(-np.dot(mu_2,self.S2 @ mu_2)/2)*np.exp(X @ (self.S2 @ mu_2))
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

#######################################################################################################################################
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
    def __init__(self,Y,X,typ,optim_params,\
                 batch_size = 50, print_info = False):
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
        if not optim_params["GD"]:#deterministic optimization
            self.theta_star = self.compute_MAP_determ(print_info,optim_params)
        else:#stochastic
            self.theta_star = self.compute_MAP_gd(print_info,optim_params)
        #Ignored if deterministic gradients are calculated
        self.batch_size = batch_size
        #Flag whether to use fixed point adaptation based on previously obtained theta^* or not
        self.ratio = self.p/self.batch_size
        
    def compute_MAP_determ(self,print_info,optim_params):
        """Compute MAP estimation either by stochastic gradient or by deterministic gradient descent
        """
        #by default
        if optim_params["compute_fp"] == False:
            return np.zeros(self.d, dtype = np.float64)
        n_restarts = optim_params["n_restarts"]
        tol = optim_params["gtol"]
        sigma = optim_params["sigma"]
        order = optim_params["order"]
        converged = False
        cur_f = 1e100
        cur_x = np.zeros(self.d,dtype = np.float64)
        best_jac = None
        for n_iters in range(n_restarts):
            if order == 2:#Newton-CG, 2nd order
                vspom = opt.minimize(self.minus_potential,sigma*np.random.randn(self.d),method = "Newton-CG", jac = self.gradpotential_deterministic, hess = self.hess_potential_determ, tol=tol)
            elif order == 1:#BFGS, quasi-Newtion, almost 1st order
                vspom = opt.minimize(self.minus_potential,sigma*np.random.randn(self.d), jac = self.gradpotential_deterministic, tol=tol)
            else:
                raise "not implemented error: order of optimization method should be 1 or 2"
            converged = converged or vspom.success
            if print_info:#show some useless information
                print("optimization iteration ended")
                print("success = ",vspom.success)
                print("func value = ",vspom.fun)
                print("jacobian value = ",vspom.jac)
                print("number of function evaluation = ",vspom.nfev)
                print("number of jacobian evaluation = ",vspom.njev)
                print("number of optimizer steps = ",vspom.nit)
            if vspom.fun < cur_f:
                cur_f = vspom.fun
                cur_x = vspom.x
                best_jac = vspom.jac
        theta_star = cur_x
        if converged:
            print("theta^* found succesfully")
        else:
            print("requested precision not necesserily achieved during searching for theta^*, try to increase error tolerance")
        print("final jacobian at termination: ")
        print(best_jac)
        return theta_star
    
    def grad_descent(self,rand_seed,print_info,optim_params,typ):
        """repeats gradient descent until convergence for one starting point
        """
        stochastic = optim_params["stochastic"]
        batch_size = optim_params["batch_size"]
        sigma = optim_params["sigma"]
        gtol = optim_params["gtol"]
        gamma = optim_params["gamma"]
        weight_decay = optim_params["weight_decay"]
        N_iters = optim_params["loop_length"]
        Max_loops = optim_params["n_loops"]
        cur_jac_norm = 1e100
        loops_counter = 0
        np.random.seed(rand_seed)
        x = sigma*np.random.randn(self.d)
        while ((cur_jac_norm > gtol) and (loops_counter < Max_loops)):
            #if true gradient still didn't converge => do SGD
            print("jacobian norm = %f, step size = %f, loop number = %d" % (cur_jac_norm,gamma,loops_counter))
            for i in range(N_iters):
                if stochastic:#SGD
                    batch_inds = np.random.choice(self.p,batch_size)
                    grad = (self.p/batch_size)*self.gradloglikelihood_stochastic(x,batch_inds) + self.gradlogprior(x)
                else:#deterministic GD
                    grad = self.gradloglikelihood_determ(x) + self.gradlogprior(x)
                x = x + gamma*grad
            gamma = gamma*weight_decay
            cur_jac_norm = np.linalg.norm(self.gradpotential_deterministic(x))
            loops_counter += 1
        res = {"value":self.minus_potential(x),"x":x,"jac_norm":cur_jac_norm}
        return res
    
    def compute_MAP_gd(self,print_info,optim_params):
        """Compute MAP estimation by stochastic gradient ascent
        """
        if optim_params["compute_fp"] == False:
            return np.zeros(self.d, dtype = np.float64)
        cur_f = 1e100
        cur_x = np.zeros(self.d,dtype = np.float64)
        best_jac = None
        n_restarts = optim_params["n_restarts"]
        nbcores = multiprocessing.cpu_count()
        trav = Pool(nbcores)
        res = trav.starmap(self.grad_descent, [(777+i,print_info,optim_params) for i in range (n_restarts)])
        for ind in range(len(res)):
            if res[ind]["value"] < cur_f:
                cur_f = res[ind]["value"]
                cur_x = res[ind]["x"]
                best_jac = res[ind]["jac_norm"]
        theta_star = cur_x
        print("best jacobian norm = ",best_jac)
        return theta_star
    
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
        if self.type == "g": # Linear regression
            return -(1. / (2*self.varY))* np.linalg.norm(self.Y-np.dot(self.X,theta))**2 \
                        - (self.d/2.)*np.log(2*np.pi*self.varY)
        elif self.type == "l": # Logistic
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
        if self.type == "g": # Linear
            temp1 = np.dot(np.dot(np.transpose(self.X), self.X), theta)
            temp2 = np.dot(np.transpose(self.X), self.Y)
            return (1. / self.varY)*(temp2 - temp1)
        elif self.type == "l": # Logistic
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
        
    def gradloglikelihood_stochastic(self,theta,batch_inds):
        """returns stochastic gradient estimation over batch_inds observations
        Args:
            ...
        Returns:
            ...
        """
        data = self.X[batch_inds,:]
        y_data = self.Y[batch_inds]
        if self.type == "g":#Linear
            raise "Not implemented error in gradloglikelihood stochastic"
        elif self.type == "l":#Logistic
            temp1 = np.exp(-np.dot(data, theta))
            temp2 = np.dot(np.transpose(data), y_data)
            temp3 = np.dot(np.transpose(data), np.divide(1, 1+temp1))
            return temp2 - temp3
        else:#Probit
            raise "Not implemented error in gradloglikelihood stochastic"
            
        
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
    
    def stoch_grad(self,theta):
        """compute gradient estimate as in SGLD
        """
        batch_inds = np.random.choice(self.p,self.batch_size)
        return self.ratio*self.gradloglikelihood_stochastic(theta,batch_inds) + self.gradlogprior(theta)
    
    def stoch_grad_fixed_point(self,theta):
        """compute gradient estimate as in SGLD with fixed-point regularization
        """ 
        batch_inds = np.random.choice(self.p,self.batch_size)
        prior_part = self.gradlogprior(theta)-self.gradlogprior(self.theta_star)
        like_part = self.ratio*(self.gradloglikelihood_stochastic(theta,batch_inds) - self.gradloglikelihood_stochastic(self.theta_star,batch_inds))
        return prior_part + like_part
    
    def stoch_grad_SAGA(self,theta):
        """compute gradient estimate in SGLD with SAGA variance reduction procedure
        """
        batch_inds = np.random.choice(self.p,self.batch_size)
        #update gradient at batch points
        vec_g_upd = self.update_gradients(batch_inds,theta)
        #update difference
        delta_g = np.sum(vec_g_upd,axis=0) - np.sum(self.grads_SAGA[batch_inds,:],axis=0)
        grad = self.gradlogprior(theta) + self.ratio * delta_g + self.g_sum
        self.g_sum += delta_g
        self.grads_SAGA[batch_inds,:] = copy.deepcopy(vec_g_upd)
        return grad
    
    def gradpotential_deterministic(self,theta):
        """
        A bit strange implementation of always deterministic gradient, this one is needed for fixed point search
        """
        return -self.gradloglikelihood_determ(theta) - self.gradlogprior(theta)
    
    #Methids for SAGA
    def init_grads_SAGA(self):
        """Function to initialize SAGA gradients
        """
        self.grads_SAGA = np.zeros((self.p,self.d),dtype = float)
        self.g_sum = np.zeros(self.d,dtype = float)
        return 
    
    def update_gradients(self,inds_arr,theta):
        """Updates gradients at batch_inds by values in point theta
        """
        X_cur = self.X[inds_arr,:]
        Y_cur = self.Y[inds_arr]
        temp = np.exp(-np.dot(X_cur, theta))
        return X_cur*(Y_cur-np.divide(1., 1 + temp)).reshape((len(inds_arr),1))
         