import numpy as np
from baselines import set_f,PWP_fast,Spectral_var,compute_poisson,compute_L_poisson,qform_q

#first-order control variates: ZAV, ZV, LS
def qform_1_ESVM(a,f_vals,X_grad,W,ind,n):
    """
    ESVM quadratic form computation: asymptotic variance estimator based on kernel W; 
    """
    x_cur = f_vals[:,ind] + X_grad @ a
    return Spectral_var(x_cur,W)

def grad_qform_1_ESVM(a,f_vals,X_grad,W,ind,n):
    """
    gradient of ESVM quadratic form
    """
    Y = f_vals[:,ind] + X_grad @ a
    return 2./n * (X_grad*PWP_fast(Y,W).reshape((n,1))).sum(axis=0)

def qform_1_ZV(a,f_vals,X_grad,ind,n):
    """
    Least squares evaluated for ZV-1
    """
    x_cur = f_vals[:,ind] + X_grad @ a
    return 1./(n-1)*np.dot(x_cur - np.mean(x_cur),x_cur - np.mean(x_cur))

def grad_qform_1_ZV(a,f_vals,X_grad,ind,n):
    """
    Gradient for quadratic form in ZV-1 method 
    """
    Y = f_vals[:,ind] + X_grad @ a
    return 2./(n-1) * (X_grad*(Y - np.mean(Y)).reshape((n,1))).sum(axis=0)

def qform_1_LS(a,f_vals,X_grad,ind,n):
    """
    Least Squares-based control variates
    """
    x_cur = f_vals[:,ind] + X_grad @ a
    return np.mean(x_cur**2)
    
def grad_qform_1_LS(a,f_vals,X_grad,ind,n):
    """
    Gradient for Least-Squares control variates
    """
    Y = f_vals[:,ind] + X_grad @ a
    return 2./n * X_grad.T @ Y

#################################################################################################################################
#second-order control variates: ZAV, ZV, LS
def qform_2_ESVM(a,f_vals,X,X_grad,W,ind,n,alpha = 0.0):
    """
    Arguments:
        a - np.array of shape (d+1,d),
             a[0,:] - np.array of shape(d) - corresponds to coefficients via linear variables
             a[1:,:] - np.array of shape (d,d) - to quadratic terms
    """
    d = X_grad.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    x_cur = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    return Spectral_var(x_cur,W) + alpha*np.sum(B**2)

def grad_qform_2_ESVM(a,f_vals,X,X_grad,W,ind,n, alpha = 0.0):
    """
    Arguments:
        a - np.array of shape (d+1,d),
             a[0,:] - np.array of shape(d) - corresponds to coefficients via linear variables
             a[1:,:] - np.array of shape (d,d) - to quadratic terms
    """
    d = X_grad.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    Y = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    #gradient w.r.t. b
    nabla_b = 2./n * (X_grad*PWP_fast(Y,W).reshape((n,1))).sum(axis=0)
    #gradient w.r.t B
    nabla_f_B = np.matmul(X_grad.reshape((n,d,1)),X.reshape((n,1,d)))
    nabla_f_B = nabla_f_B + nabla_f_B.transpose((0,2,1)) + 2*np.eye(d).reshape((1,d,d))                     
    nabla_B = 2./n*np.sum(nabla_f_B*PWP_fast(Y,W).reshape((n,1,1)),axis = 0)
    #add ridge
    nabla_B += 2*alpha*B
    #stack gradients together
    grad = np.zeros((d+1)*d,dtype = np.float64)
    grad[:d] = nabla_b
    grad[d:] = nabla_B.ravel()
    return grad
    

def qform_2_ZV(a,f_vals,X,X_grad,ind,n):
    """
    Least squares evaluated for ZV-2 method
    Arguments:
        a - np.array of shape (d+1,d),
             a[0,:] - np.array of shape(d) - corresponds to coefficients via linear variables
             a[1:,:] - np.array of shape (d,d) - to quadratic terms
    Returns:
        function value for index ind, scalar variable
    """
    d = X.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    x_cur = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    return 1./(n-1)*np.dot(x_cur - np.mean(x_cur),x_cur - np.mean(x_cur))

def grad_qform_2_ZV(a,f_vals,X,X_grad,ind,n):
    """
    Gradient for quadratic form in ZV-2 method
    """
    d = X.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    Y = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    #gradient w.r.t. b
    nabla_b = 2./(n-1)*(X_grad*(Y - np.mean(Y)).reshape((n,1))).sum(axis=0)
    #gradient w.r.t B
    nabla_f_B = np.matmul(X_grad.reshape((n,d,1)),X.reshape((n,1,d)))
    nabla_f_B = nabla_f_B + nabla_f_B.transpose((0,2,1)) + 2*np.eye(d).reshape((1,d,d))                     
    nabla_B = 2./(n-1)*np.sum(nabla_f_B*(Y-np.mean(Y)).reshape((n,1,1)),axis = 0)
    #stack gradients together
    grad = np.zeros((d+1)*d,dtype = np.float64)
    grad[:d] = nabla_b
    grad[d:] = nabla_B.ravel()
    return grad

def qform_2_LS(a,f_vals,X,X_grad,ind,n):
    """
    Least squares evaluation for 2nd order polynomials as control variates;
    Arguments:
        a - np.array of shape (d+1,d),
             a[0,:] - np.array of shape(d) - corresponds to coefficients via linear variables
             a[1:,:] - np.array of shape (d,d) - to quadratic terms
    Returns:
        function value for index ind, scalar variable
    """
    d = X.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    x_cur = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    return np.mean(x_cur**2)

def grad_qform_2_LS(a,f_vals,X,X_grad,ind,n):
    """
    Gradient for quadratic form in ZV-2 method
    """
    d = X.shape[1]
    b = a[:d]
    B = a[d:].reshape((d,d))
    Y = f_vals[:,ind] + X_grad @ b + qform_q(B+B.T,X_grad,X) + 2*np.trace(B)
    #gradient w.r.t. b
    nabla_b = 2./n * X_grad.T @ Y
    #gradient w.r.t B
    nabla_f_B = np.matmul(X_grad.reshape((n,d,1)),X.reshape((n,1,d)))
    nabla_f_B = nabla_f_B + nabla_f_B.transpose((0,2,1)) + 2*np.eye(d).reshape((1,d,d))                     
    nabla_B = 2./n*np.sum(nabla_f_B*Y.reshape((n,1,1)),axis = 0)
    #stack gradients together
    grad = np.zeros((d+1)*d,dtype = np.float64)
    grad[:d] = nabla_b
    grad[d:] = nabla_B.ravel()
    return grad

#################################################################################################################################
#k-th degree polynomials of separate variables
#################################################################################################################################
def set_Y_k_deg(a,f_vals,X,X_grad,ind):
    """
    auxiliary function for qform_k_sep and grad_qform_k_sep;
    target functional is computed here
    """
    k = a.shape[0]
    d = a.shape[1]
    nabla_phi = np.zeros_like(X)
    nabla_phi += a[0,:].reshape(1,d)
    for i in range(1,k):
        nabla_phi += (i+1)*(X**i)*a[i,:]
    delta_phi = np.zeros(X.shape[0],dtype = np.float64)
    for i in range(1,k):
        delta_phi += i*(i+1)*np.sum(X**(i-1)*a[i,:],axis=1)
    Y = f_vals[:,ind] + np.sum(X_grad*nabla_phi,axis = 1) + delta_phi
    return Y

def qform_k_sep_ESVM(a,f_vals,X,X_grad,W,ind,n,k):
    """
    Args:
        a - np.array of shape (k,d), where k - degree of polynomial
    returns:
        spectral variance estimate based on given matrix W
    """
    d = X.shape[1]
    a = a.reshape((k,d))
    Y = set_Y_k_deg(a,f_vals,X,X_grad,ind)
    return Spectral_var(Y,W)

def grad_qform_k_sep_ESVM(a,f_vals,X,X_grad,W,ind,n,k):
    """
    returns:
        gradients w.r.t. a - object of the same shape as a, i.e. (k,d)
    """
    d = X.shape[1]
    a = a.reshape((k,d))
    Y = set_Y_k_deg(a,f_vals,X,X_grad,ind)
    #start actually computing gradients
    grad_a = np.zeros_like(a)
    nabla_qf = 2./n*PWP_fast(Y,W).reshape((n,1))
    #actually this expression is just the same as in 1-dimensional case, since laplacian is exactly 0
    grad_a[0,:] = (X_grad*nabla_qf).sum(axis=0)
    for i in range(1,k):
        nabla_a_i = (i+1)*X_grad*(X**i) + (i+1)*i*X**(i-1) 
        grad_a[i,:] = (nabla_a_i*nabla_qf).sum(axis=0)
    return grad_a.ravel()

def qform_k_sep_ZV(a,f_vals,X,X_grad,ind,n,k):
    """
    Args:
        a - np.array of shape (k,d), where k - degree of polynomial
    returns:
        iid empirical variance estimate
    """
    d = X.shape[1]
    a = a.reshape((k,d))
    Y = set_Y_k_deg(a,f_vals,X,X_grad,ind)
    return 1./(n-1)*np.dot(Y - np.mean(Y),Y - np.mean(Y))

def grad_qform_k_sep_ZV(a,f_vals,X,X_grad,ind,n,k):
    """
    Args:
        a - np.array of shape (k,d), where k - degree of polynomial
    returns: gradient of iid empirical variance stimate w.r.t. parameters
    """
    d = X.shape[1]
    a = a.reshape((k,d))
    Y = set_Y_k_deg(a,f_vals,X,X_grad,ind)
    #start actually computing gradients
    grad_a = np.zeros_like(a)
    nabla_qf = 2./(n-1)*(Y-np.mean(Y)).reshape((n,1))
    #actually this expression is just the same as in 1-dimensional case, since laplacian is exactly 0
    grad_a[0,:] = (X_grad*nabla_qf).sum(axis=0)
    for i in range(1,k):
        nabla_a_i = (i+1)*X_grad*(X**i) + (i+1)*i*X**(i-1) 
        grad_a[i,:] = (nabla_a_i*nabla_qf).sum(axis=0)
    return grad_a.ravel()

def qform_k_sep_LS(a,f_vals,X,X_grad,ind,n,k):
    """
    Args:
        a - np.array of shape (k,d), where k - degree of polynomial
    returns:
        least squares statistics value
    """
    d = X.shape[1]
    a = a.reshape((k,d))
    Y = set_Y_k_deg(a,f_vals,X,X_grad,ind)
    return np.mean(Y**2)

def grad_qform_k_sep_LS(a,f_vals,X,X_grad,ind,n,k):
    """
    Args:
        a - np.array of shape (k,d), where k - degree of polynomial
    returns: gradient of iid empirical variance stimate w.r.t. parameters
    """
    d = X.shape[1]
    a = a.reshape((k,d))
    Y = set_Y_k_deg(a,f_vals,X,X_grad,ind)
    #start actually computing gradients
    grad_a = np.zeros_like(a)
    nabla_qf = 2./n*Y.reshape((n,1))
    #actually this expression is just the same as in 1-dimensional case, since laplacian is exactly 0
    grad_a[0,:] = (X_grad*nabla_qf).sum(axis=0)
    for i in range(1,k):
        nabla_a_i = (i+1)*X_grad*(X**i) + (i+1)*i*X**(i-1) 
        grad_a[i,:] = (nabla_a_i*nabla_qf).sum(axis=0)
    return grad_a.ravel()

#################################################################################################################################
#wrappers for quadratic forms and their gradients calculations
#first-order methods
def Train_1st_order(a,typ,f_vals,traj_grad_list,W,ind,n):
    """
    Universal wrapper for ZAV, ZV and LS quadratic forms
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(f_vals)
    val_list = np.zeros(n_traj)
    for i in range(len(val_list)):
        if typ == "ESVM":
            val_list[i] = qform_1_ESVM(a,f_vals[i],traj_grad_list[i],W,ind,n)
        elif typ == "ZV":
            val_list[i] = qform_1_ZV(a,f_vals[i],traj_grad_list[i],ind,n)
        elif typ == "LS":
            val_list[i] = qform_1_LS(a,f_vals[i],traj_grad_list[i],ind,n)
        else:
            raise "Not implemented error in Train_1st_order: something goes wrong"
    return np.mean(val_list)

def Train_1st_order_grad(a,typ,f_vals,traj_grad_list,W,ind,n):
    """
    Universal wrapper for ZAV,ZV and LS quadratic forms gradients calculations
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(f_vals)
    grad_vals = np.zeros_like(a)
    for i in range(n_traj):
        if typ == "ESVM":
            grad_vals += grad_qform_1_ESVM(a,f_vals[i],traj_grad_list[i],W,ind,n)
        elif typ == "ZV":
            grad_vals += grad_qform_1_ZV(a,f_vals[i],traj_grad_list[i],ind,n)
        elif typ == "LS":
            grad_vals += grad_qform_1_LS(a,f_vals[i],traj_grad_list[i],ind,n)
    grad_vals /= n_traj
    return grad_vals

#second-order methods
def Train_2nd_order(a,typ,f_vals,traj_list,traj_grad_list,W,ind,n,alpha):
    """
    average spectral variance estimation for given W matrix, based on len(traj_list) trajectories
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(traj_list)
    val_list = np.zeros(n_traj)
    for i in range(len(val_list)):
        if typ == "ESVM":
            val_list[i] = qform_2_ESVM(a,f_vals[i],traj_list[i],traj_grad_list[i],W,ind,n,alpha)
        elif typ == "ZV":
            val_list[i] = qform_2_ZV(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n)
        elif typ == "LS":
            val_list[i] = qform_2_LS(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n)
        else:
            raise "Not implemented error in Train_1st_order: something goes wrong"
    return np.mean(val_list)

def Train_2nd_order_grad(a,typ,f_vals,traj_list,traj_grad_list,W,ind,n,alpha):
    """
    gradient for average SV estimate for given W matrix
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(traj_list)
    grad_vals = np.zeros_like(a)
    for i in range(n_traj):
        if typ == "ESVM":
            grad_vals += grad_qform_2_ESVM(a,f_vals[i],traj_list[i],traj_grad_list[i],W,ind,n,alpha)
        elif typ == "ZV":
            grad_vals += grad_qform_2_ZV(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n)
        elif typ == "LS":
            grad_vals += grad_qform_2_LS(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n)
    grad_vals /= n_traj
    return grad_vals

#k-th order polynomials of separate variables
def Train_kth_order(a,typ,f_vals,traj_list,traj_grad_list,W,ind,n,k):
    """
    average spectral variance estimation for given W matrix, based on len(traj_list) trajectories
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(traj_list)
    val_list = np.zeros(n_traj)
    for i in range(len(val_list)):
        if typ == "ESVM":
            val_list[i] = qform_k_sep_ESVM(a,f_vals[i],traj_list[i],traj_grad_list[i],W,ind,n,k)
        elif typ == "ZV":
            val_list[i] = qform_k_sep_ZV(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n,k)
        elif typ == "LS":
            val_list[i] = qform_k_sep_LS(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n,k)
        else:
            raise "Not implemented error in Train_kth_order: something goes wrong"
    return np.mean(val_list)

def Train_kth_order_grad(a,typ,f_vals,traj_list,traj_grad_list,W,ind,n,k):
    """
    average spectral variance estimation for given W matrix, based on len(traj_list) trajectories
    Args:
        ...
    Returns:
        ...
    """
    n_traj = len(traj_list)
    grad_vals = np.zeros_like(a)
    for i in range(n_traj):
        if typ == "ESVM":
            grad_vals += grad_qform_k_sep_ESVM(a,f_vals[i],traj_list[i],traj_grad_list[i],W,ind,n,k)
        elif typ == "ZV":
            grad_vals += grad_qform_k_sep_ZV(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n,k)
        elif typ == "LS":
            grad_vals += grad_qform_k_sep_LS(a,f_vals[i],traj_list[i],traj_grad_list[i],ind,n,k)
    grad_vals /= n_traj
    return grad_vals


#################################################################################################################################
#staff below is not currently used, but is potentially useful
#CV and ZV implementations following Brosse, Moulines
#################################################################################################################################
def CVpolyOne(traj,traj_grad):
    """ Computation of the control variates estimator based on 1st order
        polynomials, CV1, of \theta and \theta^2, first and
        second order moments, along the trajectory, and of the associated 
        asymptotic variance, using a spectral variance estimator.
    Args:
        traj: numpy array (n, d) that contains the trajectory of the MCMC algorithm
        traj_grad: numpy array (n, d) that contrains the gradients of the 
            log posterior evaluated along the trajectory
    Returns:
        parameters of CV1 estimates: array with coefficients
    """
    n, d = traj.shape
    #samples = np.concatenate((traj, np.square(traj)), axis=1)
    samples = init_samples(traj)
    covariance = np.cov(np.concatenate((traj, samples), axis=1), rowvar=False)
    paramCV1 = covariance[:d, d:]
    return paramCV1

def CVpolyTwo(traj, traj_grad):
    """ Computation of the control variates estimator based on 2nd order
        polynomials, CV2, of \theta and \theta^2, first and
        second order moments, along the trajectory, and of the associated 
        asymptotic variance, using a spectral variance estimator.
    Args:
        traj: numpy array (n, d) that contains the trajectory of the MCMC algorithm
        traj_grad: numpy array (n, d) that contrains the gradients of the 
            log posterior evaluated along the trajectory
    Returns:
        parameters of CV2 estimates: array with coefficients
    """
    n, d = traj.shape
    #samples = np.concatenate((traj, np.square(traj)), axis=1)
    samples = init_samples(traj)
    poisson = compute_poisson(traj)
    Lpoisson = compute_L_poisson(traj,traj_grad)
    cov1 = np.cov(np.concatenate((poisson, -Lpoisson), axis=1), rowvar=False)
    A = np.linalg.inv(cov1[0:int(d*(d+3)/2), int(d*(d+3)/2):d*(d+3)])
    cov2 = np.cov(np.concatenate((poisson, samples),axis=1), rowvar=False)
    B = cov2[0:int(d*(d+3)/2), int(d*(d+3)/2):]
    paramCV2 = np.dot(A,B)
    return paramCV2


def ZVpolyOne(traj, traj_grad):
    """ Computation of the zero variance estimator based on 1st order
        polynomials, ZV1, of \theta and \theta^2, first and
        second order moments, along the trajectory, and of the associated 
        asymptotic variance, using a spectral variance estimator.
    Args:
        traj: numpy array (n, d) that contains the trajectory of the MCMC algorithm
        traj_grad: numpy array (n, d) that contrains the gradients of the 
            log posterior evaluated along the trajectory
    Returns:
        parameters of ZV1 estimates: array with coefficients
    """
    n, d = traj.shape
    samples = init_samples(traj)
    cov1 = np.cov(traj_grad, rowvar=False)
    A = np.linalg.inv(cov1)
    covariance = np.cov(np.concatenate((-traj_grad, samples), axis=1), rowvar=False)
    paramZV1 = -np.dot(A,covariance[:d, d:])
    return paramZV1

def ZVpolyTwo(traj, traj_grad):
    """ Computation of the zero variance estimator based on 2nd order
        polynomials, ZV2, of \theta and \theta^2, first and
        second order moments, along the trajectory, and of the associated 
        asymptotic variance, using a spectral variance estimator.
    Args:
        traj: numpy array (n, d) that contains the trajectory of the MCMC algorithm
        traj_grad: numpy array (n, d) that contrains the gradients of the 
            log posterior evaluated along the trajectory
    Returns:
        parameters of ZV2 estimates: array with coefficients
    """
    n, d = traj.shape
    samples = init_samples(traj)
    Lpoisson = compute_L_poisson(traj,traj_grad)
    cov1 = np.cov(Lpoisson, rowvar=False)
    A = np.linalg.inv(cov1)
    cov2 = np.cov(np.concatenate((Lpoisson, samples),axis=1), rowvar=False)
    B = cov2[0:int(d*(d+3)/2), int(d*(d+3)/2):]
    paramZV2 = - np.dot(A,B)
    return paramZV2     