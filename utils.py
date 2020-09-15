import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def set_axis_style_violplot(ax, labels, parts):
    colors = (sns.color_palette("muted")[1:4])
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(0.5, len(labels) + 0.5)
    ax.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for pc,i in zip(parts['bodies'],range(len(labels))):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('r')
        pc.set_linewidth(0.1)
        for partname in ('cbars','cmins','cmaxes','cmeans','cmeans'):
            pc = parts[partname]
            pc.set_edgecolor('black')
            pc.set_linewidth(0.7)

def set_axis_style_boxplot(ax, labels, parts):
    colors = (sns.color_palette("muted")[1:5])
    ax.grid(color='black', linestyle='-', linewidth=0.15, alpha=0.6)    
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(0.5, len(labels) + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    for pc,i in zip(parts['boxes'],range(len(labels))):
        pc.set(facecolor=colors[i],alpha=0.65)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.65)
    
    
def violplot_2ind(data1, data2, title, labels):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, frameon=False,dpi=100)      
        
    fig.suptitle(title, fontsize=20)
    ax1.set_title('First order CV', fontsize=13)
    parts = ax1.violinplot(data1, showmeans=True, showmedians=False,widths=0.6)
    set_axis_style_violplot(ax1, labels, parts)

    ax2.set_title('Second order CV', fontsize=13)
    parts = ax2.violinplot(data2, showmeans=True, showmedians=False,widths=0.6)
    set_axis_style_violplot(ax2, labels, parts)

    fig.tight_layout()
    fig.subplots_adjust(top=0.78)

    plt.show()
    
def violplot_ind(data, title, labels):

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), sharey=True, frameon=False,dpi=100)      
        
    fig.suptitle(title, fontsize=20)
    parts = ax1.violinplot(data, showmeans=True, showmedians=False,widths=0.6)
    set_axis_style_violplot(ax1, labels, parts)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    plt.show()

    
def boxplot_2ind(data1, data2, title, labels):

    meanprops = dict(linestyle='-', linewidth=1, color='black')
    medianprops = dict(linestyle='', linewidth=0)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, frameon=False,dpi=100)      
        
    fig.suptitle(title, fontsize=20)
    
    ax1.set_title('First order CV', fontsize=13)
    
    parts = ax1.boxplot(data1,  widths=0.6, patch_artist=True, meanline=True, showmeans=True, medianprops=medianprops,meanprops = meanprops, showfliers=False)
    set_axis_style_boxplot(ax1, labels, parts)
    
    ax2.set_title('Second order CV', fontsize=13)
    parts = ax2.boxplot(data2,  widths=0.6, patch_artist=True, meanline=True, showmeans=True, medianprops=medianprops,meanprops = meanprops, showfliers=False)
    set_axis_style_boxplot(ax2, labels, parts)

    fig.tight_layout()
    fig.subplots_adjust(top=0.78)

    plt.show()
    
def boxplot_ind(data, title, labels):
    meanprops = dict(linestyle='-', linewidth=1, color='black')
    medianprops = dict(linestyle='', linewidth=0)

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), sharey=True, frameon=False,dpi=100)      
    fig.suptitle(title, fontsize=20)
    parts = ax1.boxplot(data,  widths=0.6, patch_artist=True, meanline=True, showmeans=True, medianprops=medianprops,meanprops = meanprops, showfliers=False)
    set_axis_style_boxplot(ax1, labels, parts)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    plt.show()   
    

def plot_vr_rates_1st_order(res,var_ind = 0):
    print(res.shape)
    average_init_variance = np.mean(res[:,1,0,:],axis=0)
    print("initial variances: ",average_init_variance)
    izav_variances_1 = np.mean(res[:,1,1,:],axis=0)
    print("iZAV method, 1-st order, variances: ",izav_variances_1)
    print("iZAV rate: ",average_init_variance/izav_variances_1)
    zv_variances_1 = np.mean(res[:,1,3,:],axis=0)
    print("ZV method, 1-st order, variances: ",zv_variances_1)
    print("ZV rate: ",average_init_variance/zv_variances_1)
    ls_variances_1 = np.mean(res[:,1,5,:],axis=0)
    print("LS method, 1-st order, variances: ",ls_variances_1)
    print("LS rate: ",average_init_variance/ls_variances_1)
    #without VR
    total_res = [res[:,0,0,var_ind]]
    #LS-1
    total_res.append(res[:,0,5,var_ind])
    #ZV-1
    total_res.append(res[:,0,3,var_ind])
    #iZAV
    total_res.append(res[:,0,1,var_ind])
    plt.figure(figsize=(10,10))
    plt.violinplot(total_res, showmeans=True, showmedians=False)
    plt.xticks(np.arange(1,5), ('Vanilla samples', 'LS', 'ZV', 'ZAV'))
    plt.grid()
    plt.show()
    return

def plot_quantiles_1st_order(res,true_val,var_ind = 0):
    vanilla_res = res[:,0,0,var_ind]
    zav_res = res[:,0,1,var_ind]
    zv_res = res[:,0,3,var_ind]
    
    
    observations = [vanilla_res, zav_res, zv_res]
    
    #data = [data, d2, d2[::2,0]]
    plt.figure(figsize=(10,10))
    #fig7, ax7 = plt.subplots()
    plt.title('Multiple Samples with Different sizes')
    plt.boxplot(observations,showmeans=True, whis = 90)

    plt.show()
    """
    plt.figure(figsize=(10,10))
    box = plt.boxplot(observations, showmeans=True, whis=90)
    plt.ylim([1.24, 1.32]) # y axis gets more space at the extremes
    plt.grid(True, axis='y') # let's add a grid on y-axis
    plt.title('First order control functionals', fontsize=18) # chart title
    plt.ylabel('Function values') # y axis title
    plt.xticks([1,2,3], ['Vanilla','ZAV','ZV']) # x axis labels
    plt.show()
    """
    return

def plot_quantiles_2nd_order(res,true_val,var_ind = 0):
    zav_res = res[:,0,2,var_ind]
    zv_res = res[:,0,4,var_ind]
    
    quantiles = np.zeros((2,2))
    quantiles[0,0] = np.sort(zav_res)[5]
    quantiles[0,1] = np.sort(zav_res)[95]
    quantiles[1,0] = np.sort(zv_res)[5]
    quantiles[1,1] = np.sort(zv_res)[95]
    observations = [zav_res, zv_res]
    
    #data = [data, d2, d2[::2,0]]
    plt.figure(figsize=(10,10))
    plt.title('Multiple Samples with Different sizes')
    plt.boxplot(observations, showmeans=True, conf_intervals = quantiles)

    plt.show()
    """
    plt.figure(figsize=(10,10))
    box = plt.boxplot(observations, showmeans=True, whis=90)
    plt.ylim([1.24, 1.32]) # y axis gets more space at the extremes
    plt.grid(True, axis='y') # let's add a grid on y-axis
    plt.title('First order control functionals', fontsize=18) # chart title
    plt.ylabel('Function values') # y axis title
    plt.xticks([1,2,3], ['Vanilla','ZAV','ZV']) # x axis labels
    plt.show()
    """
    return

def plot_vr_rates_2nd_order(res,var_ind = 0):
    print(res.shape)
    average_init_variance = np.mean(res[:,1,0,:],axis=0)
    print("initial variances: ",average_init_variance)
    izav_variances_2 = np.mean(res[:,1,2,:],axis=0)
    print("iZAV method, 2-nd order, variances: ",izav_variances_2)
    print("iZAV rate: ",average_init_variance/izav_variances_2)
    zv_variances_2 = np.mean(res[:,1,4,:],axis=0)
    print("ZV method, 2-nd order, variances: ",zv_variances_2)
    print("ZV rate: ",average_init_variance/zv_variances_2)
    ls_variances_2 = np.mean(res[:,1,6,:],axis=0)
    print("LS method, 2-nd order, variances: ",ls_variances_2)
    #without VR
    total_res = [res[:,0,0,var_ind]]
    #LS-2
    total_res.append(res[:,0,6,var_ind])
    #ZV-2
    total_res.append(res[:,0,4,var_ind])
    #ZAV-2
    total_res.append(res[:,0,2,var_ind])
    plt.figure(figsize=(10,10))
    plt.violinplot(total_res, showmeans=True, showmedians=False)
    plt.xticks(np.arange(1,5), ('Vanilla samples', 'LS', 'ZV','ZAV'))
    plt.grid()
    plt.show()
    return

def compute_sample_confidence(res,true_val,ind):
    stats = res[:,0,ind,:]
    sample_std = np.std(stats,axis=0)
    average = np.mean(stats,axis=0)
    print(average)
    return average-true_val,1.96*sample_std

def compute_asympt_confidence(res,true_val,ind,N_test):
    stats = res[:,0,ind,:]
    asympt_std = res[:,1,ind,:]
    average = np.mean(stats,axis=0)
    std = np.mean(asympt_std,axis=0)
    print(average)
    return average-true_val,1.96*np.sqrt(std/N_test)

def get_density_values(pts_lifetime,x_pts):
    kernel = stats.gaussian_kde(pts_lifetime)
    dens_values = kernel(x_pts)
    return dens_values

def PlotKDE(x,n_pts):
    N_intervals = n_pts
    x_left = np.min(x)
    x_right = np.max(x)
    x_pts = np.linspace(x_left, x_right, N_intervals+1)
    dens_values = get_density_values(x,x_pts)
    plt.figure(figsize=(10, 10))
    plt.plot(x_pts,dens_values,color='r')
    plt.show()

"""
function to check manually sampling correctness procedure
plots KDE based on ULA training sample;
Args:
    traj - trajectory;
    ind - index of variable to visualize
"""
def visualize_projection(traj,ind):
    n_pts = 5000
    y = traj[:,ind]
    PlotKDE(y,n_pts)

"""
function to visualize data samples in R^2
Args:
    traj - trajectory
"""
def visualize_scatter_2d(traj):
    plt.figure(figsize = (10,10))
    plt.scatter(traj[:,0],traj[:,1],c='r')
    plt.show()
    return
    
    