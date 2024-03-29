# This code belongs to the paper
#
# J. Hertrich, L. Nguyen, J-F. Aujol, D. Bernard, Y. Berthoumieu, G. Steidl.
# PCA Reduced Gaussian Mixture Models with Applications in Superresolution.
# Inverse Problems and Imaging, 2021.
#
# Please cite the paper, if you use the code.
#
import tensorflow as tf
import numpy as np
import math
from funs.palm_algs import *
import time
from scipy.io import loadmat, savemat
import os

def opt_em_hddc(samps,alphas_init,Sigmas_init,Us_init,bs_init,sigma_sq,regularize=1e-5,steps=100,batch_size=1000):
    # Implements the EM algorithm for estimating the parameters of a PCA-GMM model.
    # INPUTS:
    #   samps       - N x n numpy array, where samps[i] contains the i-th data point
    #
    #   Initial parameters:    
    #   alphas_init - numpy array of length K
    #   Sigmas_init - K x d x d numpy array
    #   Us_init     - K x n x d numpy array
    #   bs_init     - K x n numpy array
    #
    #   sigma_sq    - weight parameter for the PCA
    #   regularize  - regularization constant for the covariance matrix. Default value: 1e-5.
    #   steps       - number of steps. Default value: 100.
    #   batch_size  - Parameter for the computation order. Does not effect the results, but the
    #                 execution time. Default: 10000
    #
    # OUTPUTS:
    #   Resulting parameters
    #   alphas      - numpy array of length K
    #   Sigmas      - K x d x d numpy array
    #   Us          - K x n x d numpy array
    #   bs          - K x n numpy array
    #
    K=alphas_init.shape[0]
    n=samps.shape[0]
    d=samps.shape[1]
    n_minus_d=tf.constant(d-Us_init.shape[2],dtype=tf.float64)
    @tf.function
    def compute_betas(inputs,alphas,Sigmas_logdet,Sigmas_inv,Us,bs,sigma_sq):
        log_fun_vals=[]
        beta_nenner=0
        n_inp=inputs.shape[0]
        for k in range(K):
            alpha=alphas[k]
            Sigma_inv=Sigmas_inv[k]
            Sigma_logdet=Sigmas_logdet[k]
            U=Us[k]
            b=bs[k]
            low_dim_inp=tf.matmul(inputs-b,U)
            cent_inp=low_dim_inp
            deltas=tf.reduce_sum(tf.tile(Sigma_inv[tf.newaxis,:],(n_inp,1))*cent_inp**2,1)
            factor=-0.5*Sigma_logdet-0.5*deltas
            high_low_dim_inp=tf.matmul(low_dim_inp,U,transpose_b=True)+b
            pca_loss=-(1./(2*sigma_sq[k]))*tf.reduce_sum((inputs-high_low_dim_inp)**2,1)
            log_line=factor+pca_loss+tf.math.log(alpha)
            log_fun_vals.append(log_line)
        log_fun_vals=tf.stack(log_fun_vals)
        const=tf.reduce_max(log_fun_vals,0)
        log_beta_nenner=log_fun_vals-tf.tile(tf.expand_dims(const,0),(K,1))
        log_beta_nenner=tf.math.log(tf.reduce_sum(tf.exp(log_beta_nenner),0))+const
        log_betas=[]
        for k in range(K):
            log_betas.append(log_fun_vals[k,:]-log_beta_nenner)
        log_betas=tf.stack(log_betas)
        obj=-tf.reduce_sum(log_beta_nenner)
        betas=tf.exp(log_betas)
        betas+=1e-10
        betas=betas/tf.reduce_sum(betas,0)
        alphas_new=tf.reduce_sum(betas,1)/n
        m=[]
        C=[]
        for k in range(K):
            beta_inps=inputs*tf.tile(tf.transpose(betas[k:(k+1),:]),(1,d))
            m.append(tf.reduce_sum(beta_inps,0))
            C.append(tf.matmul(tf.transpose(beta_inps),inputs))
        m=tf.stack(m)
        C=tf.stack(C)
        return alphas_new,m,C,obj
   
    
    alphas=tf.constant(alphas_init,dtype=tf.float64)
    Sigmas=tf.constant(Sigmas_init,dtype=tf.float64) 
    Us=tf.constant(Us_init,dtype=tf.float64)
    ds=tf.data.Dataset.from_tensor_slices(samps).batch(batch_size)
    bs=tf.constant(bs_init,dtype=tf.float64)
    sigma_sq=tf.constant(sigma_sq,dtype=tf.float64)
    if sigma_sq.shape==[]:
        sigma_sq=sigma_sq*tf.ones(K,dtype=tf.float64)
    tic=time.time()
    times_E=[]
    times_M=[]
    old_obj=0
    for step in range(steps):     
        objective=0
        alphas_new=0
        m=0
        C=0
        counter=0
        print('E-Step')
        tic_E=time.time()
        Sigmas_logdet=tf.reduce_sum(tf.math.log(Sigmas),-1)+n_minus_d*tf.math.log(sigma_sq)
        Sigmas_inv=1/Sigmas
        for smps in ds:
            counter+=1
            out=compute_betas(smps,alphas,Sigmas_logdet,Sigmas_inv,tf.identity(Us),bs,sigma_sq)
            alphas_new+=out[0]
            m+=out[1]
            C+=out[2]
            objective+=out[3]
        diff=objective.numpy()-old_obj
        print('Step '+str(step)+' Objective '+str(objective.numpy())+' Diff: ' +str(diff))
        toc_E=time.time()-tic_E
        if step>1:
            times_E.append(toc_E)
        print('M-Step')
        tic_M=time.time()
        old_obj=objective.numpy()
        bs_new=m/(n*tf.expand_dims(alphas_new,-1))
        S=C-(tf.matmul(m[:,:,tf.newaxis],m[:,:,tf.newaxis],transpose_b=True)/(n*alphas_new[:,tf.newaxis,tf.newaxis]))+1e-10*tf.eye(d,dtype=tf.float64,batch_shape=[K])
        S_e,S_v=tf.linalg.eigh(S/(n*alphas_new[:,tf.newaxis,tf.newaxis]))
        Us_new=S_v[:,:,-Us.shape[2]:]
        Sigmas_new=S_e[:,-Us.shape[2]:]+regularize
        sigma_sq_new=tf.reduce_mean(S_e[:,:-Us.shape[2]],-1)+regularize
        toc_M=time.time()-tic_M
        if step>1:
            times_M.append(toc_M)
        sigma_sq=sigma_sq_new
        print('Step '+str(step+1)+' Time: '+str(time.time()-tic))
        alphas=alphas_new
        Us=Us_new
        bs=bs_new
        Sigmas=Sigmas_new
        if np.abs(diff/n)<1e-5:
            return alphas,Sigmas,Us,bs,sigma_sq,np.mean(times_E),np.mean(times_M)
    return alphas,Sigmas,Us,bs,sigma_sq,np.mean(times_E),np.mean(times_M)

def initialize_U(samps,alphas,mus,Sigmas,Us,sigma_sq,learn_sigma_sq=False,batch_size=10000,regularize=1e-5):
    # Refinement of the initialization of U by performing one step of the EM algorithm with fixed alpha,mu,Sigma
    # INPUTS:
    # Required:
    #       - samps                 - n x d matrix with samples
    #       - alphas, mus, Sigmas   - parameters of a d dimensional GMM
    #       - Us                    - initialization of Us
    #       - sigma_sq              - weight parameter for the PCA
    # Optional:
    #       - batch_size            - does not effect the results. Larger batch size leads into a faster execution but
    #                                 higher RAM requirements. Default: 10000
    #       - regularize            - regularization parameter. Default: 1e-5
    #
    # OUTPUTS:
    #       - Us                    - Initialization for the Us.
    #       - bs                    - Initialization for the bs.
    #       - Sigmas                - Dimension reduced initialization for Sigmas.
    #
    n=samps.shape[0]
    K=alphas.shape[0]
    d=samps.shape[1]
    Us=tf.constant(Us,dtype=tf.float64) 
    n_minus_d=tf.constant(d-Us.shape[2],dtype=tf.float64)
    @tf.function
    def compute_betas(inputs,alphas,mus,Sigmas):
        log_fun_vals=[]
        beta_nenner=0
        n_inp=inputs.shape[0]
        for k in range(K):
            alpha=alphas[k]
            mu=mus[k]
            Sigma=Sigmas[k]
            Sigma_inv=tf.linalg.inv(Sigma)
            cent_inp=inputs-tf.tile(tf.expand_dims(mu,0),(n_inp,1))
            deltas=tf.reduce_sum(tf.matmul(cent_inp,Sigma_inv)*cent_inp,1)
            factor=-0.5*tf.linalg.logdet(Sigma)-0.5*deltas
            line=factor+tf.math.log(alpha)
            log_fun_vals.append(line)
        log_fun_vals=tf.stack(log_fun_vals)
        const=tf.reduce_max(log_fun_vals,0)
        log_beta_nenner=log_fun_vals-tf.tile(tf.expand_dims(const,0),(K,1))
        log_beta_nenner=tf.math.log(tf.reduce_sum(tf.exp(log_beta_nenner),0))+const
        log_betas=[]
        for k in range(K):
            log_betas.append(log_fun_vals[k,:]-log_beta_nenner)
        log_betas=tf.stack(log_betas)
        betas=tf.exp(log_betas)
        alphas_new=tf.reduce_sum(betas,1)/n
        m=[]
        C=[]
        for k in range(K):
            beta_inps=inputs*tf.tile(tf.transpose(betas[k:(k+1),:]),(1,d))
            m.append(tf.reduce_sum(beta_inps,0))
            C.append(tf.matmul(tf.transpose(beta_inps),inputs))
        m=tf.stack(m)
        C=tf.stack(C)
        return alphas_new,m,C


    alphas=tf.constant(alphas,dtype=tf.float64)
    mus=tf.constant(mus,dtype=tf.float64)
    Sigmas=tf.constant(Sigmas,dtype=tf.float64)   

    ds=tf.data.Dataset.from_tensor_slices(samps).batch(batch_size)
    print('Initialize U:')
    print('E-Step')
    m=0
    C=0
    alphas_new=0.
    counter=0
    for smps in ds:
        counter+=1
        out=compute_betas(smps,alphas,mus,Sigmas)
        m+=out[1]
        C+=out[2]
        alphas_new+=out[0]
    print('M-Step')
    bs_new=m/(n*tf.expand_dims(alphas_new,-1))
    S=C-(tf.matmul(m[:,:,tf.newaxis],m[:,:,tf.newaxis],transpose_b=True)/(n*alphas_new[:,tf.newaxis,tf.newaxis]))
    S_e,S_v=tf.linalg.eigh(S/(n*alphas_new[:,tf.newaxis,tf.newaxis]))
    Us_new=S_v[:,:,-Us.shape[2]:]
    Sigmas_new=S_e[:,-Us.shape[2]:]+regularize
    sigma_sq_new=tf.reduce_mean(S_e[:,:-Us.shape[2]],-1)+regularize
    print('Initialization of U complete!')
    return Us_new.numpy(), bs_new,Sigmas_new,sigma_sq_new.numpy()
    
def initialize_one_U(samps,s):
    # Generates an initialization for one U based on the PCA of all data points.
    # INPUTS:
    #       - samps     - samples
    #       - s         - reduced dimension
    # OUTPUT: Initialization U
    samps=samps.transpose()
    u,_,_=np.linalg.svd(samps,full_matrices=False)
    U=u[:,:s]
    return U



def load_initialization(name,K):
    # Loads an initialization
    # INPUT:
    #       - name          - String. Image identifyer.
    # OUTPUTS:
    #       - samples
    #       - initial data alphas, mus, Sigmas, Us
    #       - Number of classes K.
    samps=np.load('data/samples'+name+'.npy')
    filename='data/initialization'+name+'K'+str(K)+'_gauss.mat'
    data=loadmat(filename)
    alphas=np.reshape(data['alphas'],[-1])
    mus=data['mus']
    Sigmas=np.transpose(data['sigmas'],(2,0,1))
    Us=np.transpose(data['Us'],(2,0,1))
    K=mus.shape[1]
    return samps.transpose(),alphas,mus.transpose(),Sigmas,Us,K

def run_MM(name,s,use_bias=True,batch_size=10000,K=100):
    # Loads the initialization declared by the parameter name, runs the EM algorithm to compute
    # the ML-estimator of the parameters of a PCA-reduced MM GMM and saves the parameters.
    # INPUTS:
    # Required:
    #   name        - string. name of the intialization.
    #   s           - reduced dimension
    # Optional:
    #   batch_size  - batch_size paramter of opt_em. Default value: 10000.
    #   use_bias    - declares whether to use the bias in the PCA.
    regularize=1e-8
    sigma_sq=1e-4
    samples,alphas_init,mus_init,Sigmas_init,Us_init,K=load_initialization(name,K)
    d=samples.shape[1]
    dim_hr=64
    Us_init=initialize_one_U(samples,s)
    Us_init=tf.tile(tf.expand_dims(Us_init,0),(K,1,1))
    Us_init,bs_init,Sigmas_new,sigma_sq=initialize_U(samples,alphas_init,mus_init,Sigmas_init,Us_init,regularize,batch_size=batch_size,regularize=regularize,learn_sigma_sq=True)
    
    Sigmas_init=Sigmas_new
    alphas,Sigmas,Us,bs,sigma_sq,time_E,time_M=opt_em_hddc(samples,alphas_init,Sigmas_init,Us_init,bs_init,sigma_sq,regularize=regularize,steps=100,batch_size=batch_size)
    mus=tf.zeros((K,Us.shape[2]),dtype=tf.float64)
    if not os.path.isdir('mixture_models'):
        os.mkdir('mixture_models')
    savemat('mixture_models/MM_HDDC_'+str(s)+name+'K'+str(K)+'.mat',{'alphas': np.reshape(alphas.numpy(),[K,1]), 'mus': mus.numpy().transpose(), 'sigmas': np.transpose(tf.linalg.diag(Sigmas).numpy(),(1,2,0)), 'us': np.transpose(Us.numpy(),(1,2,0)),'bs': bs.numpy().transpose(),'sigma_sq': np.reshape(sigma_sq.numpy(),[K,1])})
    print(time_E)
    print(time_M)
    return time_E,time_M

