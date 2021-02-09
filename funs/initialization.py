# This code belongs to the paper
#
# L. Nguyen, J. Hertrich, J-F. Aujol, D. Bernard, Y. Berthoumieu, G. Steidl.
# PCA Reduced Gaussian Mixture Models with Applications in Superresolution.
# ArXiv preprint arXiv:2009.07520, 2020
#
# Please cite the paper, if you use the code.
#
import numpy as np
import random
import scipy.io

def init_MM(X,K,regularize):
    # Initialization for learning a PCA reduced GMM.
    # INPUTS:
    #       - X             - d x n Numpy array conataining the Samples
    #       - K             - int. Number of classes
    #       - regularize    - float. regularization parameter
    # OUTPUTS:
    #       - Initial parameters alphas, mus, sigmas, Us
    d=X.shape[0]
    n=X.shape[1]
    mus=np.zeros((d,K))
    sigmas=np.zeros((d,d,K))
    Us=np.zeros((d,d,K))
    alphas=np.ones(K)*1.0/K
    centers=random.sample(range(n),K)
    num_neighbors=max(40,2*d)
    for i in range(K):
        dists=np.sum((X-np.tile(X[:,centers[i]:centers[i]+1],(1,n)))**2,axis=0)
        points=np.argpartition(dists,num_neighbors)
        points=points[:num_neighbors]
        points=X[:,points]
        mus[:,i]=np.sum(points,axis=1)/points.shape[1]
        sigmas[:,:,i]=points.dot(points.transpose())/points.shape[1]+regularize*np.eye(d)
        u,s,vh=np.linalg.svd(points,full_matrices=False)
        Us[:,:,i]=u
    return alphas,mus,sigmas,Us

def initialize(name,K,regularize=1e-4):
    # Loads samples from the data directory, computes an initialization and saves it again in the data directory.
    # INPUTs:
    #       - name          - String. Image identifyer.
    #       - K             - int. Number of classes.
    #       - regularize    - float. regularization parameter.
    samps=np.load('data/samples'+name+'.npy',allow_pickle=True)
    alphas,mus,sigmas,Us=init_MM(samps,K,regularize)
    scipy.io.savemat('data/initialization'+name+'_gauss.mat',{'alphas': alphas, 'mus': mus, 'sigmas': sigmas, 'Us':Us})

