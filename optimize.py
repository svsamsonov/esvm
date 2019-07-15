import numpy as np
import scipy.stats as spstats
import scipy.sparse as sparse
import scipy.optimize as opt
from scipy import signal
from baselines import qform_q,set_function,PWP_fast,Spectral_var
from VR_methods import Train_1st_order,Train_2nd_order,Train_1st_order_grad,Train_2nd_order_grad,Train_kth_order, Train_kth_order_grad,set_Y_k_deg
from samplers import ULA,MALA,RWM,ULA_SAGA
from multiprocessing import Pool
import multiprocessing

def optim_1var(i,crit,inds_arr,traj,traj_grad,W_train_spec,n_restarts,tol,sigma):
    """function to run optimization for 
    """
    if (i < 2):#first order optimization
        n = traj[0].shape[0]
        d = traj[0].shape[1]
        cv_1_poly = np.zeros(d,dtype = float)
        converged = False
        cur_f = 1e100
        cur_x = np.zeros(d,dtype = float)
        cur_jac = None
        for n_iters in range(n_restarts):
            vspom = opt.minimize(Train_1st_order,sigma*np.random.randn(d),args = (crit,traj,traj_grad,W_train_spec,inds_arr[i],n),jac = Train_1st_order_grad, tol = tol)
            converged = converged or vspom.success
            if (vspom.fun < cur_f):
                cur_f = vspom.fun
                cur_x = vspom.x
                cur_jac = vspom.jac
        cv_1_poly = cur_x
        if converged:
            print("1 degree optimization terminated succesfully")
        else:
            print("requested precision not necesserily achieved, try to increase error tolerance")
        print("jacobian at termination: ")
        print(cur_jac)
        return cv_1_poly
    else:#second order optimization
        n = traj[0].shape[0]
        d = traj[0].shape[1]
        cv_2_poly = np.zeros((d+1)*d,dtype = float)
        converged = False
        cur_f = 1e100
        best_jac = None
        for n_iters in range(n_restarts):
            #ZAV
            vspom = opt.minimize(Train_2nd_order,sigma*np.random.randn(d+1,d),args=(crit,traj,traj_grad,W_train_spec,inds_arr[i-2],n), jac = Train_2nd_order_grad, tol = tol)
            converged = converged or vspom.success
            if (vspom.fun < cur_f):
                cur_f = vspom.fun
                cv_2_poly = vspom.x
                best_jac = vspom.jac
        if converged:
            print("2 degree optimization terminated succesfully")
        else:
            print("requested precision not necesserily achieved, try to increase error tolerance")
        print("Jacobian matrix at termination: ")
        print(best_jac)
        return cv_2_poly       

def optimize_parallel(crit,inds_arr,traj,traj_grad,W_train_spec,n_restarts,tol,sigma = 1.0):
    """function to run parallel implementation of ZAV implementation
    Returns:
        res - tuple of length 4, with first 2 entries - first-order CV's, second 2 entries - second-order CV's
    """
    #specify other in case of non-standard optimization
    nbcores = 4
    trav = Pool(nbcores)
    res = trav.starmap(optim_1var, [(i,crit,inds_arr,traj,traj_grad,W_train_spec,n_restarts,tol,sigma) for i in range (nbcores)])
    d = traj[0].shape[1]
    A_ZAV_1 = np.zeros((2,d),dtype = float)
    A_ZAV_2 = np.zeros((2,(d+1)*d),dtype = float)
    #returning results in correct order
    A_ZAV_1[0,:] = res[0]
    A_ZAV_1[1,:] = res[1]
    A_ZAV_2[0,:] = res[2]
    A_ZAV_2[1,:] = res[3]
    trav.close() 
    return A_ZAV_1,A_ZAV_2

def optimize_1(i,ind,f_vals,traj,traj_grad,W_train_spec,n_restarts,tol,sigma):
    """
    """
    if (i < f_vals[0].shape[1]):
        crit = "ESVM"
    elif (i < 2*f_vals[0].shape[1]):
        crit = "ZV"
    elif (i < 3*f_vals[0].shape[1]):
        crit = "LS"
    else:
        raise "Not implemented error in optimize_1"
    n = traj[0].shape[0]
    d = traj[0].shape[1]
    cv_1_poly = np.zeros(d,dtype = float)
    converged = False
    cur_f = 1e100
    cur_x = np.zeros(d,dtype = float)
    cur_jac = None
    for n_iters in range(n_restarts):
        vspom = opt.minimize(Train_1st_order,sigma*np.random.randn(d),args = (crit,f_vals,traj_grad,W_train_spec,ind,n),jac = Train_1st_order_grad, tol = tol)
        converged = converged or vspom.success
        if (vspom.fun < cur_f):
            cur_f = vspom.fun
            cur_x = vspom.x
            cur_jac = vspom.jac
    cv_1_poly = cur_x
    if converged:
        print("1 degree optimization terminated succesfully")
    else:
        print("requested precision not necesserily achieved, try to increase error tolerance")
    print("jacobian at termination: ")
    print(cur_jac)
    return cv_1_poly

def optimize_2(i,ind,f_vals,traj,traj_grad,W_train_spec,n_restarts,tol,sigma,alpha = 0.0):
    """
    """
    if (i < f_vals[0].shape[1]):#ZAV
        crit = "ESVM"
    elif (i < 2*f_vals[0].shape[1]):#
        crit = "ZV"
    elif (i < 3*f_vals[0].shape[1]):
        crit = "LS"
    else:
        raise "Not implemented error in optimize_1"
    n = traj[0].shape[0]
    d = traj[0].shape[1]
    cv_2_poly = np.zeros((d+1)*d,dtype = float)
    converged = False
    cur_f = 1e100
    best_jac = None
    for n_iters in range(n_restarts):
        #create and symmetrize starting point
        init_point = sigma*np.random.randn(d+1,d)
        init_point[1:,:] = (init_point[1:,:]+init_point[1:,:].T)
        vspom = opt.minimize(Train_2nd_order,init_point,args=(crit,f_vals,traj,traj_grad,W_train_spec,ind,n,alpha), jac = Train_2nd_order_grad, tol = tol)
        converged = converged or vspom.success
        if (vspom.fun < cur_f):
            cur_f = vspom.fun
            cv_2_poly = vspom.x
            best_jac = vspom.jac
    if converged:
        print("2 degree optimization terminated succesfully")
    else:
        print("requested precision not necesserily achieved, try to increase error tolerance")
    print("Jacobian matrix at termination: ")
    print(best_jac)
    return cv_2_poly

def optimize_k(i,ind,deg,f_vals,traj,traj_grad,W_train_spec,n_restarts,tol,sigma,alpha = 0.0):
    """
    """
    if (i < f_vals[0].shape[1]):#ZAV
        crit = "ESVM"
    elif (i < 2*f_vals[0].shape[1]):#
        crit = "ZV"
    elif (i < 3*f_vals[0].shape[1]):
        crit = "LS"
    else:
        raise "Not implemented error in optimize_1"
    n = traj[0].shape[0]
    d = traj[0].shape[1]
    cv_k_poly = np.zeros(deg*d,dtype = float)
    converged = False
    cur_f = 1e100
    best_jac = None
    for n_iters in range(n_restarts):
        #create and symmetrize starting point
        init_point = sigma*np.random.randn(deg,d)
        vspom = opt.minimize(Train_kth_order,init_point,args=(crit,f_vals,traj,traj_grad,W_train_spec,ind,n,deg), jac = Train_kth_order_grad, tol = tol)
        converged = converged or vspom.success
        if (vspom.fun < cur_f):
            cur_f = vspom.fun
            cv_k_poly = vspom.x
            best_jac = vspom.jac
    if converged:
        print("k-th degree optimization terminated succesfully")
    else:
        print("requested precision not necesserily achieved, try to increase error tolerance")
    print("Jacobian matrix at termination: ")
    print(best_jac)
    return cv_k_poly
    

def optimize_parallel_new(degree,inds_arr,f_vals,traj,traj_grad,W_train_spec,n_restarts,tol,sigma = 1.0, alpha = 0.0):
    """function to run parallel optimnization
    """
    nbcores = multiprocessing.cpu_count()
    trav = Pool(nbcores)
    if degree == 1:
        res = trav.starmap(optimize_1, [(i,i%len(inds_arr),f_vals,traj,traj_grad,W_train_spec,n_restarts,tol,sigma) for i in range (3*len(inds_arr))])
    elif degree == 2:
        res = trav.starmap(optimize_2, [(i,i%len(inds_arr),f_vals,traj,traj_grad,W_train_spec,n_restarts,tol,sigma,alpha) for i in range (3*len(inds_arr))])
    else: #for degree >2 run optimization for polynomials of separate variables
        res = trav.starmap(optimize_k, [(i,i%len(inds_arr),degree,f_vals,traj,traj_grad,W_train_spec,n_restarts,tol,sigma,alpha) for i in range (3*len(inds_arr))])
    trav.close() 
    ZAV = []
    ZV = []
    LS = []
    for i in range(len(inds_arr)):
        ZAV.append(res[i])
        ZV.append(res[i+len(inds_arr)])
        LS.append(res[i+2*len(inds_arr)])
    return np.asarray(ZAV),np.asarray(ZV),np.asarray(LS)
    

def optimize_1_deg(inds_arr,X,X_grad,W,n_restarts = 1,tol = 1e-8,crit="ZAV"):
    """
    Perform optimization by gradient descent in case for control variables - polynomials up to degree k in separate variables;
    Args:
        X - inputs;
        W - matrix of quadratic form;
        Potential - object from potentials.py (Gauss,Gauss_Mixture or Regression kernels);
        n_restarts - number of restarts for each gradient descent iteration
    returns:
        cv_1_poly - np.array of shape (d,d) - coefficients via respective polynomials 
    """
    assert crit in ["ZAV","LS","ZV"],"Value of criteria parameter should be stricly equal to ZAV,ZV or LS"
    n = X[0].shape[0]
    d = X[0].shape[1]
    n_vars = len(inds_arr)
    cv_1_poly = np.zeros((n_vars,d),dtype = np.float64)
    var_counter = 0
    for ind in inds_arr:
        converged = False
        cur_f = 1e100
        cur_x = np.zeros(d,dtype = np.float64)
        best_jac = None
        for n_iters in range(n_restarts):
            vspom = opt.minimize(Train_1st_order,3*np.random.randn(d),args = (crit,X,X_grad,W,ind,n),jac = Train_1st_order_grad, tol = tol)
            converged = converged or vspom.success
            if (vspom.fun < cur_f):
                cur_f = vspom.fun
                cur_x = vspom.x
                cur_jac = vspom.jac
        cv_1_poly[var_counter,:] = cur_x
        var_counter += 1
        if converged:
            print("1 degree optimization terminated succesfully")
        else:
            print("requested precision not necesserily achieved, try to increase error tolerance")
        print("Jacobian matrix at termination: ")
        print(cur_jac)
    return cv_1_poly

def optimize_k_deg(X,X_grad,W,k,n_restarts = 1, tol = 1e-8):
    """
    Perform optimization by gradient descent in case for control variables - polynomials up to degree k in separate variables;
    Args:
        X - inputs;
        W - matrix of quadratic form;
        Potential - object from potentials.py (Gauss,Gauss_Mixture or Regression kernels);
        k - maximum degree of polynomials
        n_restarts - number of restarts for each gradient descent iteration
    returns:
        cv_k_poly - np.array of shape(d,k,d) - coefficients via respective polynomials
    """
    n = X.shape[0]
    d = X.shape[1]
    cv_k_poly = np.zeros((d,k,d),dtype = np.float64)
    for ind in range(d):
        converged = False
        cur_f = 1e100
        cur_x = np.zeros(d,dtype = np.float64)
        best_jac = None
        for n_iters in range(n_restarts):
            vspom = opt.minimize(qform_k_sep_CV,3*np.random.randn(k,d),args=(X,X_grad,W,ind,n,k,d),jac = grad_qform_k_sep_CV,tol=tol)
            converged = converged or vspom.success
            if vspom.fun < cur_f:
                cur_f = vspom.fun
                cur_x = vspom.x
                best_jac = vspom.jac
        cv_k_poly[ind,:,:] = cur_x.reshape((k,d))
        if converged:
            print("k degree optimization terminated succesfully")
        else:
            print("requested precision not necesserily achieved, try to increase error tolerance")
            print(best_jac)
    return cv_k_poly

def optimize_2_tensor(inds_arr,X,X_grad,W,n_restarts = 1, tol = 1e-9, crit ="ZAV"):
    """
    Same as above, but now we work with polynomials up to degree 2, now control variables are in form <a,x> + <Ax,x>, i.e. polynomial up to degree 2, with mixed monomials allowed;
    returns:
        cv_2_poly - np.array of shape (d,(d+1)*d)
    """
    assert crit in ["ZAV","LS","ZV"],"Value of criteria parameter should be stricly equal to ZAV,ZV or LS"
    n = X[0].shape[0]
    d = X[0].shape[1]
    n_vars = len(inds_arr)
    cv_2_poly = np.zeros((n_vars,(d+1)*d),dtype = np.float64)
    var_counter = 0
    for ind in inds_arr:
        converged = False
        cur_f = 1e100
        cur_x = np.zeros(d,dtype = np.float64)
        best_jac = None
        for n_iters in range(n_restarts):
            #ZAV
            vspom = opt.minimize(Train_2nd_order,3*np.random.randn(d+1,d),args=(crit,X,X_grad,W,ind,n), jac = Train_2nd_order_grad, tol = tol)
            converged = converged or vspom.success
            if vspom.fun < cur_f:
                cur_f = vspom.fun
                cur_x = vspom.x
                best_jac = vspom.jac
        cv_2_poly[var_counter,:] = cur_x
        var_counter += 1
        if converged:
            print("2 degree optimization terminated succesfully")
        else:
            print("requested precision not necesserily achieved, try to increase error tolerance")
        print("Jacobian matrix at termination: ")
        print(best_jac)
    return cv_2_poly

def eval_samples(typ,f_vals,X,X_grad,A,W_spec,n,d,vars_arr):
    """Universal function to evaluate MCMC samples with ot without control functionals
    Args:
        typ - one of "Vanilla", "1st_order","2nd_order","kth_order"
        ...
    Returns:
        ...
    """
    if typ not in ["Vanilla","1st_order","2nd_order","kth_order"]:
        raise "Not implemented error in EvalSamples"
    n_vars = len(vars_arr)
    integrals = np.zeros(n_vars,dtype = np.float64)
    vars_spec = np.zeros_like(integrals)
    var_counter = 0
    for ind in range(len(vars_arr)):
        #spectral estimate for variance
        if typ == "Vanilla":
            Y = f_vals[:,ind]
        elif typ == "1st_order":
            a = A[var_counter,:]
            Y = f_vals[:,ind] + X_grad @ a
        elif typ == "2nd_order":
            b = A[var_counter,:d]
            B = A[var_counter,d:].reshape((d,d))
            Y = f_vals[:,ind] + X_grad @ b + (X_grad.dot(B + B.T)*X).sum(axis=1) + 2*np.trace(B)
        elif typ == "kth_order":
            a = A[var_counter,:]
            a = a.reshape((-1,d))
            Y = set_Y_k_deg(a,f_vals,X,X_grad,ind)
        integrals[var_counter] = np.mean(Y)
        vars_spec[var_counter] = Spectral_var(Y,W_spec)
        var_counter = var_counter + 1
    return integrals,vars_spec

def Run_eval_test(intseed,method,vars_arr,Potential,W_spec,CV_dict,step,N,n,d,params_test = None, f_type = "posterior_mean"):
    """ 
    generic function that runs a MCMC trajectory
    and computes means and variances for the ordinary samples, 
    CV1, ZV1, CV2 and ZV2 
    """
    sampler_type = method["sampler"]
    burn_type = method["burn_type"]
    main_type = method["main_type"]
    if sampler_type == "ULA":
        traj,traj_grad = ULA(intseed,Potential,step, N, n, d, burn_type, main_type)
    elif sampler_type == "MALA":
        traj,traj_grad,n_accepted = MALA(intseed,Potential,step,N,n,d, burn_type, main_type)
    elif sampler_type == "RWM":
        traj,traj_grad,n_accepted = RWM(intseed,Potential,step,N,n,d)
    else:
        raise "Not implemented error: choose ULA, MALA or RWM as sampler"
    #lists to save the results of the trajectory
    ints_all = []
    vars_all = []
    #initialize function values
    f_vals = set_function(f_type,[traj],vars_arr,params_test)
    #kill dimension which is not needed
    f_vals = f_vals[0]
    integrals,vars_spec = eval_samples("Vanilla",f_vals,traj,traj_grad,1,W_spec,n,d,vars_arr) #usual samples, without variance reduction
    ints_all.append(integrals)
    vars_all.append(vars_spec)
    if CV_dict["ZAV"] != None:
        A_ZAV_1 = CV_dict["ZAV"][0]
        A_ZAV_2 = CV_dict["ZAV"][1]
        integrals,vars_spec = eval_samples("kth_order",f_vals,traj,traj_grad,A_ZAV_1,W_spec,n,d,vars_arr) #CV - polynomials of degree 1, ZAV estimator
        ints_all.append(integrals)
        vars_all.append(vars_spec)
        integrals,vars_spec = eval_samples("2nd_order",f_vals,traj,traj_grad,A_ZAV_2,W_spec,n,d,vars_arr) #CV - polynomials of degree 2, ZAV estimator
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
        

