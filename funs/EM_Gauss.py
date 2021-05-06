# This code belongs to the paper
#
# L. Nguyen, J. Hertrich, J-F. Aujol, D. Bernard, Y. Berthoumieu, G. Steidl.
# PCA Reduced Gaussian Mixture Models with Applications in Superresolution.
# ArXiv preprint arXiv:2009.07520, 2020
#
# Please cite the paper, if you use the code.
#
import tensorflow as tf
import numpy as np
import math
import time
from scipy.io import loadmat, savemat
import os

def opt_em(samps,alphas_init,mus_init,Sigmas_init,regularize=1e-5,steps=100,batch_size=10000,stop_crit=1e-5):
    # Implements the EM algorithm for estimating the parameters of a Gaussian Mixture Models
    # INPUTS:
    #   samps       - N x n numpy array, where samps[i] contains the i-th data point
    #
    #   Initial parameters:    
    #   alphas_init - numpy array of length K
    #   mus_init    - K x n numpy array
    #   Sigmas_init - K x n x n numpy array
    #
    #   regularize  - regularization constant for the covariance matrix. Default value: 1e-5.
    #   steps       - number of steps. Default value: 100.
    #   batch_size  - Parameter for the computation order. Does not effect the results, but the
    #                 execution time. Default: 10000
    #
    # OUTPUTS:
    #   Resulting parameters
    #   alphas      - numpy array of length K
    #   mus         - K x n numpy array
    #   Sigmas      - K x n x n numpy array
    #
    K=alphas_init.shape[0]
    n=samps.shape[0]
    d=samps.shape[1]
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
        obj=-tf.reduce_sum(log_beta_nenner)
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
        return alphas_new,m,C,obj
   
    
    alphas=tf.constant(alphas_init,dtype=tf.float64)
    mus=tf.constant(mus_init,dtype=tf.float64)
    Sigmas=tf.constant(Sigmas_init,dtype=tf.float64)
    ds=tf.data.Dataset.from_tensor_slices(samps).batch(batch_size)
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
        for smps in ds:
            counter+=1
            out=compute_betas(smps,alphas,mus,Sigmas)
            alphas_new+=out[0]
            m+=out[1]
            C+=out[2]
            objective+=out[3]
        mus_new=[]
        Sigmas_new=[]
        diff=objective.numpy()-old_obj
        print('Step '+str(step)+' Objective '+str(objective.numpy())+' Diff: ' +str(diff))
        toc_E=time.time()-tic_E
        if step>1:
            times_E.append(toc_E)
        old_obj=objective.numpy()
        print('M-Step')
        tic_M=time.time()
        for k in range(K):
            mu_new_k=m[k]/(n*alphas_new[k])
            mus_new.append(mu_new_k)
            Sigma_new_k=C[k]/(n*alphas_new[k])-tf.matmul(mu_new_k[:,tf.newaxis],tf.transpose(mu_new_k[:,tf.newaxis]))
            Sigmas_new.append(Sigma_new_k+regularize*tf.eye(C.shape[2],dtype=tf.float64))
        mus_new=tf.stack(mus_new)
        Sigmas_new=tf.stack(Sigmas_new)   
        toc_M=time.time()-tic_M      
        if step>1:
            times_M.append(toc_M)   
        eps=tf.reduce_sum((alphas-alphas_new)**2)+tf.reduce_sum((mus-mus_new)**2)
        eps+=tf.reduce_sum((Sigmas-Sigmas_new)**2)
        print('Step '+str(step+1)+' Time: '+str(time.time()-tic))
        alphas=alphas_new
        mus=mus_new
        Sigmas=Sigmas_new
        if np.abs(diff/n)<stop_crit:
            return alphas,mus,Sigmas,np.mean(times_E),np.mean(times_M)
    return alphas,mus,Sigmas,np.mean(times_E),np.mean(times_M)

def run_MM(name,batch_size=10000,stop_crit=1e-5,K=100):
    # Loads the initialization declared by the parameter name, runs the EM algorithm to compute
    # the ML-estimator of the parameters of a GMM and saves the parameters.
    # INPUTS:
    #   name        - string. name of the intialization.
    #   batch_size  - batch_size paramter of opt_em. Default value: 10000.
    regularize=1e-6
    def load_initialization(name,K):       
        samps=np.load('data/samples'+name+'.npy')
        filename='data/initialization'+name+'K'+str(K)+'_gauss.mat'
        data=loadmat(filename)
        alphas=np.reshape(data['alphas'],[-1])
        mus=data['mus']
        Sigmas=np.transpose(data['sigmas'],(2,0,1))
        K=mus.shape[1]
        return samps.transpose(),alphas,mus.transpose(),Sigmas,K
    samples,alphas_init,mus_init,Sigmas_init,K=load_initialization(name,K)
    d=samples.shape[1]
    alphas,mus,Sigmas,time_E,time_M=opt_em(samples,alphas_init,mus_init,Sigmas_init,regularize=regularize,steps=100,batch_size=batch_size,stop_crit=stop_crit)
    if not os.path.isdir('mixture_models'):
        os.mkdir('mixture_models')
    savemat('mixture_models/MM_EM_gauss'+name+'K'+str(K)+'.mat',{'alphas': np.reshape(alphas.numpy(),[K,1]), 'mus': mus.numpy().transpose(), 'sigmas': np.transpose(Sigmas.numpy(),(1,2,0))})
    print(time_E)
    print(time_M)
    return time_E,time_M





        
