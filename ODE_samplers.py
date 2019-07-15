import numpy as np
from scipy.integrate import RK45,solve_ivp
from multiprocessing import Pool
import multiprocessing
from optimize import eval_samples
from VR_methods import set_Y_k_deg
import copy 

def grad_ascent_ODE(r_seed,Potential,step,params,N, d, typ, t = 1.0):
    """MCMC ULA for ODE
    """
    np.random.seed(r_seed)
    U = Potential.log_potential
    grad = Potential.grad_log_potential
    traj = np.zeros((N, d))
    traj_grad = np.zeros((N, d))
    if typ == "VdP":
        #van-der-Pole dynamics
        sigma = params["sigma"]
        x = np.exp(sigma*np.random.normal(scale=1.0, size=d))
    elif typ == "LV":
        #Lotki-Volterra system
        mu = params["mu"]
        sigma = params["sigma"]
        x = np.zeros(d,dtype = float)
        step = step*np.array([10.0,0.1,10.0,0.1])
        for i in range(len(x)):
            #sample from prior the initial parameter values
            x[i] = mu[i]*np.exp(sigma[i]*np.random.randn())
    else:
        #smthing strange
        raise "wrong type in grad_ascent function"
    for k in np.arange(N):
        #print(x)
        #update parameter value
        Potential.update_theta(x)
        #re-solve ODE system
        Potential.update_system_solvers()
        #calculate potential and gradient
        U_x = U(x,t)
        grad_x = grad(x,t)
        traj[k,] = x
        traj_grad[k,] = grad_x
        x = x + step * grad_x
    return traj, traj_grad

def ULA_ODE(r_seed,Potential,step,params,N, n, d, typ, t = 1.0):
    """MCMC ULA for ODE
    """
    np.random.seed(r_seed)
    U = Potential.log_potential
    grad = Potential.grad_log_potential
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    if typ == "VdP":
        #van-der-Pole dynamics
        sigma = params["sigma"]
        x = np.exp(sigma*np.random.normal(scale=1.0, size=d))
    elif typ == "LV":
        #Lotki-Volterra system
        mu = params["mu"]
        sigma = params["sigma"]
        #x = np.zeros(d,dtype = float)
        step = step*np.array([10.0,0.1,10.0,0.1])
        mu_init = Potential.theta_mle
        sigma_init = mu_init/10.0
        x = mu_init + sigma_init*np.random.randn(d)
        #for i in range(len(x)):
            #sample from prior the initial parameter values
            #x[i] = mu[i]*np.exp(sigma[i]*np.random.randn())
    else:
        #smthing strange
        raise "wrong type in ULA_ODE function"
    for k in np.arange(N):
        #print(x)
        #update parameter value
        Potential.update_theta(x)
        #re-solve ODE system
        Potential.update_system_solvers()
        #calculate potential and gradient
        U_x = U(x,t)
        grad_x = grad(x,t)
        x = x + step * grad_x + np.sqrt(2*step)*np.random.normal(size=d)
    for k in np.arange(n):
        #print(x)
        #update parameter value
        Potential.update_theta(x)
        #re-solve ODE system
        Potential.update_system_solvers()
        #calculate potential and gradient
        U_x = U(x,t)
        grad_x = grad(x,t)
        traj[k,] = x
        traj_grad[k,] = grad_x
        x = x + step * grad_x + np.sqrt(2*step)*np.random.normal(size=d)
    return traj, traj_grad

def MALA_ODE(r_seed,Potential,step,params,N, n, d, typ, t = 1.0):
    """ MCMC MALA for ODE
    Args:
        r_seed - random seed to be initialized with;
        step: stepsize of the algorithm;
        N: burn-in period;
        n: number of samples after the burn-in;
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored;
        traj_grad: numpy array of size (n, d), where the gradients of the 
            potential U along the trajectory are stored;
        n_accepted: number of accepted moves after burn-in period;
    """
    np.random.seed(r_seed)
    U = Potential.log_potential
    grad = Potential.grad_log_potential
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    ll_vals = np.zeros((n,1),dtype = float)
    if typ == "VdP":
        #van-der-Pole dynamics
        sigma = params["sigma"]
        x = np.exp(sigma*np.random.normal(scale=1.0, size=d))
    elif typ == "LV":
        #Lotki-Volterra system
        mu = params["mu"]
        sigma = params["sigma"]
        step = step*np.array([10.0,0.1,10.0,0.1])
        mu_init = Potential.theta_mle
        sigma_init = mu_init/20.0
        x = mu_init + sigma_init*np.random.randn(d)
        #x = np.zeros(d,dtype = float)
        #for i in range(len(x)):
            #sample from prior the initial parameter values
            #x[i] = mu[i]*np.exp(sigma[i]*np.random.randn())
    else:
        #smthing strange
        raise "wrong type in MALA_ODE function"
    for k in np.arange(N):
        #update parameter value
        #print(x)
        Potential.update_theta(x)
        #re-solve ODE system
        Potential.update_system_solvers()
        #calculate potential and gradient
        U_x = U(x,t)
        grad_x = grad(x,t)
        y = x + step * grad_x + np.sqrt(2*step)*np.random.normal(size=d)
        #update parameter value
        Potential.update_theta(y)
        #re-solve ODE system
        Potential.update_system_solvers()
        #calculate potential and gradient
        U_y = U(y,t)
        grad_y = grad(y,t)
        #if full gradient computed during burn-in, do acceptance-rejection step
        #logratio = U_y-U_x + (1./(4*step))*((np.linalg.norm(y-x-step*grad_x))**2 \
            #- (np.linalg.norm(x-y-step*grad_y))**2)
        logratio = U_y-U_x + (np.linalg.norm((y-x-step*grad_x)/(2*np.sqrt(step))))**2 \
            - (np.linalg.norm((x-y-step*grad_y)/(2*np.sqrt(step))))**2
        if np.log(np.random.uniform())<=logratio:
            x = y
    n_accepted = 0
    for k in np.arange(n):
        #print(x)
        #update parameter value
        Potential.update_theta(x)
        #re-solve ODE system
        Potential.update_system_solvers()
        #calculate potential and gradient
        U_x = U(x,t)
        grad_x = grad(x,t)
        traj[k,] = x
        traj_grad[k,] = grad_x
        #ll_vals[k,] = Potential.log_likelihood(x)
        y = x + step * grad_x + np.sqrt(2*step)*np.random.normal(size=d)
        #update parameter value
        Potential.update_theta(y)
        #re-solve ODE system
        Potential.update_system_solvers()
        #calculate potential and gradient
        U_y = U(y,t)
        grad_y = grad(y,t)
        #if full gradient computed during burn-in, do acceptance-rejection step
        #logratio = U_y-U_x + (1./(4*step))*((np.linalg.norm(y-x-step*grad_x))**2 \
            #- (np.linalg.norm(x-y-step*grad_y))**2)
        logratio = U_y-U_x + (np.linalg.norm((y-x-step*grad_x)/(2*np.sqrt(step))))**2 \
            - (np.linalg.norm((x-y-step*grad_y)/(2*np.sqrt(step))))**2
        if np.log(np.random.uniform())<=logratio:
            x = y
            n_accepted += 1
    return traj, traj_grad, n_accepted

def usual_evaluation(f_vals,traj,traj_grad,CV_dict,W_spec,n,d,vars_arr):
    """
    """
    ints_all = []
    vars_all = []
    integrals,vars_spec = eval_samples("Vanilla",f_vals,traj,traj_grad,1,W_spec,n,d,vars_arr) #usual samples, without variance reduction
    ints_all.append(integrals)
    vars_all.append(vars_spec)
    if CV_dict["ESVM"] != None:
        A_ESVM_1 = CV_dict["ESVM"][0]
        A_ESVM_2 = CV_dict["ESVM"][1]
        integrals,vars_spec = eval_samples("kth_order",f_vals,traj,traj_grad,A_ESVM_1,W_spec,n,d,vars_arr) #CV - polynomials of degree 1, ZAV estimator
        ints_all.append(integrals)
        vars_all.append(vars_spec)
        integrals,vars_spec = eval_samples("2nd_order",f_vals,traj,traj_grad,A_ESVM_2,W_spec,n,d,vars_arr) #CV - polynomials of degree 2, ZAV estimator
        ints_all.append(integrals)
        vars_all.append(vars_spec)
    if CV_dict["ZV"] != None:
        A_ZV_1 = CV_dict["ZV"][0]
        A_ZV_2 = CV_dict["ZV"][1]
        integrals,vars_spec = eval_samples("kth_order",f_vals,traj,traj_grad,A_ZV_1,W_spec,n,d,vars_arr) #CV - polynomials of degree 1, ZV estimator
        ints_all.append(integrals)
        vars_all.append(vars_spec)
        integrals,vars_spec = eval_samples("2nd_order",f_vals,traj,traj_grad,A_ZV_2,W_spec,n,d,vars_arr) #CV - polynomials of degree 2, ZV estimator
        ints_all.append(integrals)
        vars_all.append(vars_spec)
    if CV_dict["LS"] != None:
        A_LS_1 = CV_dict["LS"][0]
        A_LS_2 = CV_dict["LS"][1]
        integrals,vars_spec = eval_samples("kth_order",f_vals,traj,traj_grad,A_LS_1,W_spec,n,d,vars_arr) #CV - polynomials of degree 1, LS estimator
        ints_all.append(integrals)
        vars_all.append(vars_spec)
        integrals,vars_spec = eval_samples("2nd_order",f_vals,traj,traj_grad,A_LS_2,W_spec,n,d,vars_arr) #CV - polynomials of degree 2, LS estimator
        ints_all.append(integrals)
        vars_all.append(vars_spec) 
    ints_all = np.asarray(ints_all) 
    vars_all = np.asarray(vars_all)
    return ints_all,vars_all
    

def run_eval_test(intseed,method,vars_arr,Potential,W_spec,CV_dict,step,N,n,d,params_test,f_type,params_prior,s_type,t_moments):
    """ 
    generic function that runs a MCMC trajectory
    and computes means and variances for the ordinary samples, 
    CV1, ZV1, CV2 and ZV2 
    """
    if f_type == "posterior_mean":
        sampler_type = method["sampler"]
        if sampler_type == "ULA":
            traj,traj_grad = ULA_ODE(intseed,Potential,step,params_prior,N,n,d,s_type)
        elif sampler_type == "MALA":
            traj,traj_grad,n_accepted = MALA_ODE(intseed,Potential,step,params_prior,N,n,d,s_type)
        else:
            raise "Not implemented error when choosing sampler in run_eval_test"
        #lists to save the results of the trajectory
        ints_all = []
        vars_all = []
        #initialize function values
        f_vals = set_function(f_type,[traj],vars_arr,params_test)
        #kill dimension which is not needed
        f_vals = f_vals[0]
        ints_all,vars_all = usual_evaluation(f_vals,traj,traj_grad,CV_dict,W_spec,n,d,vars_arr)
        return ints_all,vars_all
    elif f_type == "evidence":
        ints_all = [[] for j in range(len(t_moments))]
        vars_all = [[] for j in range(len(t_moments))]
        f_vals = np.zeros((len(t_moments),n),dtype = float)
        traj = np.zeros((len(t_moments),n,d),dtype = float)
        traj_grad = np.zeros((len(t_moments),n,d),dtype = float)
        for i in range(len(t_moments)):
            if method["sampler"] == "ULA":
                f_vals[i],traj[i],traj_grad[i] = ULA_ODE(i+intseed*len(t_moments),Potential, step, params_prior, N, n, d, s_type,t_moments[i])
            elif method["sampler"] == "MALA":
                f_vals[i],traj[i],traj_grad[i],n_accepted = MALA_ODE(i+intseed*len(t_moments),Potential,step,params_prior,N,n,d,s_type,t_moments[i])
            ints_all[i],vars_all[i] = usual_evaluation(f_vals[i],traj[i],traj_grad[i],CV_dict[i],W_spec,n,d,vars_arr)
        #now calculate integrals based on new values
        evidence_est = np.zeros(len(ints_all[0]),dtype = float)
        for j in range(len(ints_all[0])):
            for i in range(len(f_vals)-1):
                evidence_est[j] += (ints_all[i+1][j] - inds_all[i][j])*(t_moments[i+1]-t_moments[i])/2
        return evidence_est

def set_function(f_type,traj,inds_arr,params):
    """Main function to be evaluated in case of logistic regression
    Args:
        f_type - one of "posterior_mean","posterior_ll_point","posterior_ll_mean"
        traj - list of trajectories
        inds_arr - reasonable in case of "posterior_mean", otherwise ignored
        params - dictionary with fields "X","Y"
    returns:
        array of function values of respective shapes
    """
    if f_type == "posterior_mean":#params is ignored in this case
        f_vals = np.zeros((len(traj),len(traj[0]),len(inds_arr)),dtype = float)
        for traj_ind in range(len(traj)):
            for point_ind in range(len(inds_arr)):
                f_vals[traj_ind,:,point_ind] = set_f(traj[traj_ind],inds_arr[point_ind])
    elif f_type == "power_posterior_integral":
        return 0
    else:#smthing strange
        raise "Not implemented error in set_function: check f_type value"
    return f_vals

def set_f(X,ind):
    """
    Element-wise function of observation, depending on ind, please, change it only here
    Arguments:
        X - np.array of shape (n,d);
        ind - int, 0 <= ind <= d 
    """
    return copy.deepcopy(X[:,ind]) 