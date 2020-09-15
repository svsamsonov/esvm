import numpy as np
import copy as copy
from baselines import Get_steps_SAGA
from multiprocessing import Pool
import multiprocessing
from potentials import potentialRegression

def MC_sampler(intseed,Potential,N_test,d):
    """
    Potential to sample iid observations for pure Monte-Carlo
    """
    traj,traj_grad = Potential.sample(intseed,N_test)
    return traj,traj_grad

def ULA(r_seed,Potential,step, N, n, d, burn_type = "SGLD",main_type = "SGLDFP"):
    """ MCMC ULA
    Args:
        Potential - one of objects from potentials.py
        step: stepsize of the algorithm
        N: burn-in period
        n: number of samples after the burn-in
        d: dimensionality of the problem
        burn_type: type of gradient updates during burn-in period;
                    allowed values: "full","SGLD","SGLDFP","SAGA"
        main_type: type of gradient updates during main loop;
                    allowed values: "full", "SGLD", "SGLDFP", "SAGA"    
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored
        traj_grad: numpy array of size (n, d), where the gradients of the 
            potential U along the trajectory are stored
    """
    np.random.seed(r_seed)
    #select method for gradient updates during burn-in
    if burn_type == "full":
        grad_burn = Potential.gradpotential
    elif burn_type == "SGLD":
        grad_burn = Potential.stoch_grad
    elif burn_type == "SGLDFP":
        grad_burn = Potential.stoch_grad_fixed_point
    #elif burn_type == "SAGA":
        #grad_burn = Potential.stoch_grad_SAGA
    else:
        raise "Not implemented error: invalid value in ULA sampler, in burn_type"
    #select method for gradient updates during main loop   
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    x = np.random.normal(scale=5.0, size=d) # initial value X_0
    for k in np.arange(N): # burn-in period
        grad_burn_val = grad_burn(x)
        x = x + step * grad_burn_val +\
            np.sqrt(2*step)*np.random.normal(size=d)
    #burn-in ended
    if main_type == "SAGA":#compute SAGA updates here
        grads_SAGA = Potential.update_gradients(np.arange(Potential.p),x)
        g_sum = np.sum(grads_SAGA,axis=0)
        for k in np.arange(n):#main loop
            batch_inds = np.random.choice(Potential.p,Potential.batch_size)
            #update gradient at batch points
            vec_g_upd = Potential.update_gradients(batch_inds,x)
            #update difference
            delta_g = np.sum(vec_g_upd,axis=0) - np.sum(grads_SAGA[batch_inds,:],axis=0)
            grad = Potential.gradlogprior(x) + Potential.ratio * delta_g + g_sum
            traj_grad[k,] = grad
            traj[k,] = x
            #SAGA step
            x = x + step*grad + np.sqrt(2*step)*np.random.normal(size=d) 
            g_sum += delta_g
            grads_SAGA[batch_inds,:] = copy.deepcopy(vec_g_upd)
        return traj,traj_grad
    else:#all other gradient schemes may be computed in the similar manner via unique interface
        if main_type == "full":
            grad_main = Potential.gradpotential
        elif main_type == "SGLD":
            grad_main = Potential.stoch_grad
        elif main_type == "SGLDFP":
            grad_main = Potential.stoch_grad_fixed_point
        else:
            raise "Not implemented error: invalid value in ULA sampler, in main_type" 
        if (main_type != "full"):#we need to re-calculate gradient
            for k in np.arange(n): # samples
                traj[k,]=x
                traj_grad[k,] = grad_main(x)
                x = x + step * grad_main(x) + np.sqrt(2*step)*np.random.normal(size=d) 
        else:#in case of full gradient we do not need to re-calculate gradients on each step
            for k in np.arange(n): # samples
                grad = grad_main(x)
                traj[k,]=x
                traj_grad[k,]=grad
                x = x + step * grad + np.sqrt(2*step)*np.random.normal(size=d)
        return traj,traj_grad 

def ULA_check(r_seed,Potential,step, N, n, d):
    """function to check correctness of gradient descent and related fields
    """
    np.random.seed(r_seed)
    #plug here the method that you would like to check
    U_grad = Potential.gradpotential
    traj = np.zeros((n, d))
    x = np.random.normal(scale=5.0, size=d) # initial value X_0
    for k in np.arange(N): # burn-in period
        x = x + step * U_grad(x)
    for k in np.arange(n): # samples
        grad = U_grad(x)
        traj[k,] = x
        x = x + step * grad  
    return traj 

def MALA(r_seed,Potential, step, N, n, d, burn_type = "SGLD",main_type = "SGLDFP"):
    """ MCMC MALA
    Args:
        step: stepsize of the algorithm
        N: burn-in period
        n: number of samples after the burn-in
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored
        traj_grad: numpy array of size (n, d), where the gradients of the 
            potential U along the trajectory are stored
        n_accepted: number of accepted moves after burn-in period
    """
    np.random.seed(r_seed)
    U = Potential.potential
    #select gradient type during burn-in period
    if burn_type == "full":
        grad_burn = Potential.gradpotential
    elif burn_type == "SGLD":
        grad_burn = Potential.stoch_grad
    elif burn_type == "SGLDFP":
        grad_burn = Potential.stoch_grad_fixed_point
    elif burn_type == "SAGA":
        grad_burn = Potential.grad_SAGA
    else:
        raise "Not implemented error: invalid value in MALA sampler, in burn_type"    
    #note that during the main loop only full gradient is allowed
    grad_main = Potential.gradpotential
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    x = np.random.normal(scale=5.0, size=d)
    for k in np.arange(N):
        y = x + step * grad_burn(x) + np.sqrt(2*step)*np.random.normal(size=d)
        #if full gradient computed during burn-in, do acceptance-rejection step
        if burn_type == "full":
            logratio = U(y)-U(x) + (1./(4*step))*((np.linalg.norm(y-x-step*grad_burn(x)))**2 \
                      - (np.linalg.norm(x-y-step*grad_burn(y)))**2)
            if np.log(np.random.uniform())<=logratio:
                x = y
        else:#no acceptance-rejection needed
            x = y
    n_accepted = 0
    for k in np.arange(n):
        grad = grad_main(x)
        traj[k,]=x
        traj_grad[k,]= grad
        #do not re-calculate gradient
        y = x + step * grad + np.sqrt(2*step)*np.random.normal(size=d)
        logratio = U(y)-U(x)+(1./(4*step))*((np.linalg.norm(y-x-step*grad_main(x)))**2 \
                      - (np.linalg.norm(x-y-step*grad_main(y)))**2)
        if np.log(np.random.uniform())<=logratio:
            n_accepted += 1
            x = y
    return traj, traj_grad, n_accepted

def RWM(r_seed,Potential, step, N, n, d):
    """ MCMC RWM
    Args:
        step: stepsize of the algorithm
        N: burn-in period
        n: number of samples after the burn-in
    Returns:
        traj: a numpy array of size (n, d), where the trajectory is stored
        traj_grad: numpy array of size (n, d), where the gradients of the 
            potential U along the trajectory are stored
        n_accepted: number of accepted moves after burn-in period
    """
    np.random.seed(r_seed)
    U = Potential.potential
    grad_U = Potential.gradpotential # for control variates only
    traj = np.zeros((n, d))
    traj_grad = np.zeros((n, d))
    x = np.random.normal(scale=5.0, size=d)
    for k in np.arange(N):
        y = x + np.sqrt(2*step)*np.random.normal(size=d)
        logratio = U(y)-U(x)
        if np.log(np.random.uniform())<=logratio:
            x = y
    n_accepted = 0
    for k in np.arange(n):
        traj[k,]=x
        traj_grad[k,]=grad_U(x)
        y = x + np.sqrt(2*step)*np.random.normal(size=d)
        logratio = U(y)-U(x)
        if np.log(np.random.uniform())<=logratio:
            n_accepted += 1
            x = y
    return traj, traj_grad, n_accepted

def MCMC_sampler(start_seed,n_traj, typ, Potential, step, N_burn, N_gen, d, burn_type = "SGLD",main_type = "SGLDFP"):
    """Uniform Wrapper for MCMC samplers
    Args:
       n_traj - number of generated trajectories;
       typ - type of MCMC sampler, currently "ULA","ULA_SAGA","MALA","RWM";
       Potential - potential function;
       step - stepsize of the algorithm;
    returns: 
       traj_all - list of length n_traj, entries - np.arrays of shape (N_gen,d) - MCMC trajectories;
       traj_grad_all - list of length n_traj, entries - np.arrays of shape (N_gen,d) - MCMC trajectories gradients;
       n_accepted - list of length n_traj, entries - number of accepted moves along each trajectory
    """
    traj_all = []
    traj_grad_all = []
    n_accepted_all = []
    for i in range(n_traj):
        if typ == "ULA":
            traj, traj_grad = ULA(start_seed+i,Potential,step, N_burn, N_gen, d, burn_type, main_type)
        elif typ == "MALA":
            traj,traj_grad,n_accepted = MALA(start_seed+i,Potential, step, N_burn, N_gen, d, burn_type, main_type)
            n_accepted_all.append(n_accepted)
        elif typ == "RWM":
            traj,traj_grad,n_accepted = RWM(start_seed+i,Potential, step, N_burn, N_gen, d)
            n_accepted_all.append(n_accepted)
        traj_all.append(traj)
        traj_grad_all.append(traj_grad)
    if typ == "ULA":
        return traj_all,traj_grad_all
    #return n_accepted too
    else:
        return traj_all,traj_grad_all,n_accepted
    
def Generate_train(n_traj,method,Potential,step,N_burn,N_gen,d):
    """Parallelized implementation of MCMC_sampler for fast train sample generation:
    """
    start_seed = 777
    # number of cores exploited for the computation of the independent trajectories
    # by default, all available cores on the machine
    nbcores = multiprocessing.cpu_count()
    print("ncores = ",nbcores)
    trav = Pool(nbcores)
    typ = method["sampler"]
    burn_type = method["burn_type"]
    main_type = method["main_type"]
    if typ == "ULA":
        res = trav.starmap(ULA, [(start_seed+i, Potential, step,N_burn,N_gen,d,burn_type,main_type) for i in range (n_traj)])
    elif typ == "MALA":
        res = trav.starmap(MALA, [(start_seed+i, Potential, step,N_burn,N_gen,d,burn_type,main_type) for i in range (n_traj)])
    elif typ == "RWM":
        res = trav.starmap(RWM, [(start_seed+i, Potential, step,N_burn,N_gen,d) for i in range (n_traj)])
    elif typ == "ULA_check":
        res = trav.starmap(ULA_check, [(start_seed+i, Potential, step,N_burn,N_gen,d) for i in range (n_traj)])
    return res
    
    
def ULA_SAGA(Potential,step_start,step_end,step_decay,cv_type,N,n,d):
    """SAGA algorithm for stochastic gradients calculations
    """
    traj = np.zeros((n, d))
    x = np.random.normal(scale=5.0, size=d) # initial value X_0
    #vector of gradients stored at data points
    vec_g_alpha = Potential.update_grad(np.arange(Potential.p),x)
    g_alpha = np.sum(vec_g_alpha,axis = 0)
    #initialize array of step sizes
    steps = Get_steps_SAGA(N,n,step_start,step_end,step_decay)
    for k in np.arange(N): # burn-in period
        batch_inds = np.random.choice(Potential.p,Potential.batch_size)
        vec_g_upd = Potential.update_grad(batch_inds,x)
        delta_g = np.sum(vec_g_upd,axis=0) - np.sum(vec_g_alpha[batch_inds,:],axis=0)
        grad = Potential.gradlogprior(x) + Potential.ratio * delta_g + g_alpha
        x = x + steps[k]*grad + np.sqrt(2*steps[k])*np.random.normal(size=d) 
        g_alpha += delta_g
        vec_g_alpha[batch_inds,:] = copy.deepcopy(vec_g_upd)
    traj_grad = np.zeros((n, d))
    for k in np.arange(n): # samples
        traj[k,]=x
        #generate batch for update
        batch_inds = np.random.choice(Potential.p,Potential.batch_size)
        vec_g_upd = Potential.update_grad(batch_inds,x)
        delta_g = np.sum(vec_g_upd,axis=0) - np.sum(vec_g_alpha[batch_inds,:],axis=0)
        grad_SAGA = Potential.gradlogprior(x) + Potential.ratio * delta_g + g_alpha
        #save gradients
        if cv_type == "SAGA":#store the same gradient, then think about bias
            traj_grad[k,:] = grad_SAGA
        elif cv_type == "full":#full gradient calculations, prohibited for large datasets
            traj_grad[k,:] = Potential.full_gradpotential(x)
        elif cv_type == "SGLD":#stochastic approximation to gradient of log-density just as in SGLD
            batch_inds_new = np.random.choice(Potential.p,Potential.batch_size)
            traj_grad[k,:] = Potential.gradpotential_SGLD(x,batch_inds_new)
        elif cv_type == "SGLDFP":#fixed point regularization added
            batch_inds_new = np.random.choice(Potential.p,Potential.batch_size)
            traj_grad[k,:] = Potential.gradpotential_FP(x,batch_inds_new)
        else:
            raise "Not implemented error in ULA_SAGA function: check cv_type parameter"
        #SAGA step
        x = x + steps[N+k]*grad_SAGA + np.sqrt(2*steps[N+k])*np.random.normal(size=d) 
        g_alpha += delta_g
        vec_g_alpha[batch_inds,:] = copy.deepcopy(vec_g_upd)
    return traj,traj_grad  

def SAGA_sampler(n_traj, Potential, step_start, step_end, step_decay, cv_type, N_burn, N_gen, d):
    """Function to sample SAGA trajectories;
    Args:
        n_traj - number of SAGA trajectories;
        Potential - currently only logisticRegression_SAGA is supported;
        step_start - float, first step, typically of order 10**(-2);
        step_end - float, final step, typically of order 10**(-4);
        step_decay - type of inperpolation between step_start and step_end, either "linear" or "spurious", i.e. a*(b+n)**(-alpha)
        cv_type - type of gradient estimate used in control functionals, may be one of the following:
            "full" - full and fair gradients, computed in O(sample_size);
            "SGLD" - simple stochastic apploximation to the gradient, as in SGLD;
            "SGLDFP" - fixed point regularization added; 
            "SAGA" - SAGA updated gradients, same as ones needed to generate \theta_{k+1}
    """
    start_seed = 777
    traj_all = []
    traj_grad_all = []
    for i in range(n_traj):
        np.random.seed(start_seed+i)
        traj, traj_grad = ULA_SAGA(Potential, step_start, step_end, step_decay, cv_type, N_burn, N_gen, d)
        traj_all.append(traj)
        traj_grad_all.append(traj_grad)
    return traj_all,traj_grad_all        
        