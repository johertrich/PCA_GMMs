# This code belongs to the paper
#
# L. Nguyen, J. Hertrich, J-F. Aujol, D. Bernard, Y. Berthoumieu, G. Steidl.
# PCA Reduced Gaussian Mixture Models with Applications in Superresolution.
# ArXiv preprint arXiv:2009.07520, 2020
#
# Please cite the paper, if you use the code.
#
import numpy as np
import tensorflow as tf
import scipy.io
import scipy.ndimage
import os

def reconstruction_gauss(obs,alphas,mus,Sigmas,avg_gam=0.12,tau=4,q=2):
    # Reconstruction of a 2D high resolution image based on a low resolution observation and a joint GMM using the
    # method from citation [3] of the readme file.
    # Inputs:
    # Required:
    #       - obs                   - Observation.
    #       - alphas, mus, Sigmas   - Parameters of the GMM.
    # Optional:
    #       - avg_gam               - Parameter for the patch averaging. Default=0.12
    #       - tau                   - patch size in the low resolution image. Default: 4
    #       - q                     - magnification factor. Default: 2
    #
    # Output: Reconstructed high resolution image.
    #
    dim_hr=tau**2*q**2
    dim_lr=tau**2
    mus_lr=tf.constant(mus[:,-dim_lr:],dtype=tf.float64)
    mus_hr=tf.constant(mus[:,:dim_hr],dtype=tf.float64)
    Sigmas_lr=tf.constant(Sigmas[:,-dim_lr:,-dim_lr:],dtype=tf.float64)
    Sigmas_hr=tf.constant(Sigmas[:,:dim_hr,:dim_hr],dtype=tf.float64)
    Sigmas_hrlr=tf.constant(Sigmas[:,:dim_hr,-dim_lr:],dtype=tf.float64)
    Sigmas_lr_inv=tf.linalg.inv(Sigmas_lr)
    logdet_Sigmas_lr=tf.linalg.logdet(tf.constant(Sigmas_lr))
    m=obs.shape[0]
    n=obs.shape[1]
    p_min_x=0
    p_min_y=0
    p_max_x=m-tau
    p_max_y=n-tau
    patch_ind=np.array(range(tau))
    patch_x=np.tile(patch_ind[:,np.newaxis],(1,tau))
    patch_y=np.tile(patch_ind[np.newaxis,:],(tau,1))
    patch_x=np.reshape(patch_x,[-1])
    patch_y=np.reshape(patch_y,[-1])
    weights_ind=np.array(range(tau*q))-0.5*(tau*q-1)
    weights_ind=np.tile(weights_ind[:,np.newaxis],(1,tau*q))
    weights=np.exp(-.5*avg_gam*(weights_ind**2+weights_ind.transpose()**2))
    x_hat=np.zeros((m*q,n*q))
    weight_sum=np.zeros_like(x_hat)
    obs_vec=np.reshape(obs,[-1])
    @tf.function
    def logf(x):
        K=alphas.shape[0]
        cent_xks=tf.tile(x[:,tf.newaxis,:],(1,K,1))-tf.tile(mus_lr[tf.newaxis,:,:],(x.shape[0],1,1))
        Sigx=tf.linalg.matvec(Sigmas_lr_inv,cent_xks)
        out=tf.tile(tf.expand_dims(-2*tf.math.log(alphas)+logdet_Sigmas_lr,0),(x.shape[0],1))+tf.reduce_sum(cent_xks*Sigx,axis=2)
        return out
    @tf.function
    def compute_hr_patch(x):
        if len(x.shape)==1:
            x=x[tf.newaxis,:]
        vals=logf(x)
        k_star=tf.math.argmin(vals,axis=1)
        my_mus_hr=tf.gather(mus_hr,k_star)
        my_mus_lr=tf.gather(mus_lr,k_star)
        my_Sigmas_hrlr=tf.gather(Sigmas_hrlr,k_star)
        my_Sigmas_lr_inv=tf.gather(Sigmas_lr_inv,k_star)
        z_ij_hat=my_mus_hr+tf.linalg.matvec(my_Sigmas_hrlr,tf.linalg.matvec(my_Sigmas_lr_inv,x-my_mus_lr)) 
        return tf.squeeze(z_ij_hat)
    for i in range(p_min_x,p_max_x+1):
        patch_x_i=patch_x+i
        j_inds=np.array(range(p_min_y,p_max_y+1))
        inds=np.tile(n*patch_x_i[np.newaxis,:],(j_inds.shape[0],1))+np.tile(patch_y[np.newaxis,:],(j_inds.shape[0],1))+np.tile(j_inds[:,np.newaxis],(1,tau**2))
        z_ijs=obs_vec[inds]
        z_ij_hats=compute_hr_patch(tf.constant(z_ijs,dtype=tf.float64)).numpy()
        z_ij_hat_patches=np.reshape(z_ij_hats,[j_inds.shape[0],q*tau,q*tau])
        for j in range(p_min_y,p_max_y+1):
            z_ij_hat_patch=z_ij_hat_patches[j]
            x_hat[i*q:(i*q+q*tau),j*q:(j*q+q*tau)]+=weights*z_ij_hat_patch
            weight_sum[i*q:(i*q+q*tau),j*q:(j*q+q*tau)]+=weights
    x_hat/=weight_sum
    return x_hat

def reconstruction_gauss_3D(obs,alphas,mus,Sigmas,avg='Gauss',avg_gam=0.12,tau=[4,4,4],q=[2,2,2]):
    # Reconstruction of a 3D high resolution image based on a low resolution observation and a joint GMM using the
    # method from citation [3] of the readme file.
    # Inputs:
    # Required:
    #       - obs                   - Observation.
    #       - alphas, mus, Sigmas   - Parameters of the GMM.
    # Optional:
    #       - avg_gam               - Parameter for the patch averaging. Default=0.12
    #       - tau                   - patch dimensions in the low resolution image. Default: [4,4,4]
    #       - q                     - magnification factor in each direction. Default: [2,2,2]
    #
    # Output: Reconstructed high resolution image.
    #
    if type(tau)==int:
        tau=[tau]*3
    if type(q)==int:
        q=[q]*3
    dim_hr=np.prod(tau)*np.prod(q)
    dim_lr=np.prod(tau)
    mus_lr=mus[:,-dim_lr:]
    mus_hr=mus[:,:dim_hr]
    Sigmas_lr=Sigmas[:,-dim_lr:,-dim_lr:]
    Sigmas_hrlr=Sigmas[:,:dim_hr,-dim_lr:]
    Sigmas_lr_inv=np.linalg.inv(Sigmas_lr)
    logdet_Sigmas_lr=tf.linalg.logdet(tf.constant(Sigmas_lr))
    m=obs.shape[0]
    n=obs.shape[1]
    o=obs.shape[2]
    p_min_x=0
    p_min_y=0
    p_min_z=0
    p_max_x=m-tau[0]
    p_max_y=n-tau[1]
    p_max_z=o-tau[2]
    patch_ind=[]
    patch_ind.append(np.array(range(tau[0])))
    patch_ind.append(np.array(range(tau[1])))
    patch_ind.append(np.array(range(tau[2])))
    patch_x=np.tile(patch_ind[0][:,np.newaxis,np.newaxis],(1,tau[1],tau[2]))
    patch_y=np.tile(patch_ind[1][np.newaxis,:,np.newaxis],(tau[0],1,tau[2]))
    patch_z=np.tile(patch_ind[2][np.newaxis,np.newaxis,:],(tau[0],tau[1],1))
    patch_x=np.reshape(patch_x,[-1])
    patch_y=np.reshape(patch_y,[-1])
    patch_z=np.reshape(patch_z,[-1])
    weights_ind=[]
    weights_ind.append(np.array(range(tau[0]*q[0]))-0.5*(tau[0]*q[0]-1))
    weights_ind.append(np.array(range(tau[1]*q[1]))-0.5*(tau[1]*q[1]-1))
    weights_ind.append(np.array(range(tau[2]*q[2]))-0.5*(tau[2]*q[2]-1))
    weights_ind[0]=np.tile(weights_ind[0][:,np.newaxis,np.newaxis],(1,tau[1]*q[1],tau[2]*q[2]))
    weights_ind[1]=np.tile(weights_ind[1][np.newaxis,:,np.newaxis],(tau[0]*q[0],1,tau[2]*q[2]))
    weights_ind[2]=np.tile(weights_ind[2][np.newaxis,np.newaxis,:],(tau[0]*q[0],tau[1]*q[1],1))
    weights=np.exp(-.5*avg_gam*(weights_ind[0]**2+weights_ind[1]**2+weights_ind[2]**2))
    x_hat=np.zeros((m*q[0],n*q[1],o*q[2]))
    weight_sum=np.zeros_like(x_hat)
    obs_vec=np.reshape(obs,[-1])
    @tf.function
    def logf(x):
        K=alphas.shape[0]
        cent_xks=tf.tile(x[:,tf.newaxis,:],(1,K,1))-tf.tile(mus_lr[tf.newaxis,:,:],(x.shape[0],1,1))
        Sigx=tf.linalg.matvec(Sigmas_lr_inv,cent_xks)
        out=tf.tile(tf.expand_dims(-2*tf.math.log(alphas)+logdet_Sigmas_lr,0),(x.shape[0],1))+tf.reduce_sum(cent_xks*Sigx,axis=2)
        return out
    @tf.function
    def compute_hr_patch(x):
        if len(x.shape)==1:
            x=x[tf.newaxis,:]
        vals=logf(x)
        k_star=tf.math.argmin(vals,axis=1)
        my_mus_hr=tf.gather(mus_hr,k_star)
        my_mus_lr=tf.gather(mus_lr,k_star)
        my_Sigmas_hrlr=tf.gather(Sigmas_hrlr,k_star)
        my_Sigmas_lr_inv=tf.gather(Sigmas_lr_inv,k_star)
        z_ij_hat=my_mus_hr+tf.linalg.matvec(my_Sigmas_hrlr,tf.linalg.matvec(my_Sigmas_lr_inv,x-my_mus_lr)) 
        return tf.squeeze(z_ij_hat) 
    for i in range(p_min_x,p_max_x+1):
        print(i)
        patch_x_i=patch_x+i
        for j in range(p_min_y,p_max_y+1):
            patch_y_j=patch_y+j
            jj_inds=np.array(range(p_min_z,p_max_z+1))
            inds=np.tile(o*n*patch_x_i[np.newaxis,:],(jj_inds.shape[0],1))+np.tile(o*patch_y_j[np.newaxis,:],(jj_inds.shape[0],1))+np.tile(patch_z[np.newaxis,:],(jj_inds.shape[0],1))+np.tile(jj_inds[:,np.newaxis],(1,np.prod(tau)))
            z_ijs=obs_vec[inds]
            z_ij_hats=compute_hr_patch(tf.constant(z_ijs,dtype=tf.float64)).numpy()
            z_ij_hat_patches=np.reshape(z_ij_hats,[jj_inds.shape[0],q[0]*tau[0],q[1]*tau[1],q[2]*tau[2]])
            for jj in range(p_min_z,p_max_z+1):
                z_ij_hat_patch=z_ij_hat_patches[jj]
                x_hat[i*q[0]:(i*q[0]+q[0]*tau[0]),j*q[1]:(j*q[1]+q[1]*tau[1]),jj*q[2]:(jj*q[2]+q[2]*tau[2])]+=weights*z_ij_hat_patch
                weight_sum[i*q[0]:(i*q[0]+q[0]*tau[0]),j*q[1]:(j*q[1]+q[1]*tau[1]),jj*q[2]:(jj*q[2]+q[2]*tau[2])]+=weights
    x_hat/=weight_sum
    return x_hat

def psnr(img,ref):
    # Computes the PSNR of img with reference ref.
    # INPUTS:
    #       - img           - image.
    #       - ref           - reference.
    # OUTPUT: psnr.
    img_vec=np.reshape(img,[-1])
    ref_vec=np.reshape(ref,[-1])    
    MSE=np.sum((img_vec-ref_vec)**2)/img_vec.shape[0]
    psnr=-10*np.log10(MSE)
    return psnr

def superresolution_gauss(MM_name,img_name,q=2,tau=4):
    # Loads a GMM and a low resolution image and reconstructs the corresponding high resolution image using the method
    # from citation [3] from the readme file and saves the reconstruction.
    # Inputs:
    # Required:
    #       - MM_name       - identifyer of the GMM.
    #       - img_name      - identifyer of the image.
    # Optional:
    #       - q             - magnification factor.
    #       - tau           - patch size in the low resolution image.
    # Output: psnr of the reconstruction
    MM=scipy.io.loadmat('mixture_models/'+MM_name)
    alphas=np.reshape(MM['alphas'],[-1])
    mus=MM['mus'].transpose()
    Sigmas=np.transpose(MM['sigmas'],(2,0,1))
    img_m=scipy.io.loadmat('./imgs_superres/'+img_name+'.mat')
    img_obs=img_m['img_lr']
    img_ground_truth=img_m['img_hr']
    if len(img_obs.shape)==2:
        reconstructed=reconstruction_gauss(img_obs,alphas,mus,Sigmas,q=q,tau=tau)
    elif len(img_obs.shape)==3:
        reconstructed=reconstruction_gauss_3D(img_obs,alphas,mus,Sigmas,q=q,tau=tau)        
    else:
        raise ValueError('Dimension of the observation has to be 2 or 3!')
    if not os.path.isdir('reconstructions'):
        os.mkdir('reconstructions')
    scipy.io.savemat('./reconstructions/'+img_name+'_'+MM_name,{'reconstructed':reconstructed})
    res_psnr=psnr(reconstructed,img_ground_truth)
    print(res_psnr)
    return res_psnr

def superresolution_PCA(MM_name,img_name,q=2,tau=4,sigma_sq=1e-4):
    # Loads a PCA-reduced GMM and a low resolution image and reconstructs the corresponding high resolution image using the method
    # from citation [3] from the readme file and saves the reconstruction.
    # Inputs:
    # Required:
    #       - MM_name       - identifyer of the PCA-reduced GMM.
    #       - img_name      - identifyer of the image.
    # Optional:
    #       - q             - magnification factor.
    #       - tau           - patch size in the low resolution image.
    #       - sigma_sq      - parameter sigma^2 from the PCA-reduced GMM
    # Output: psnr of the reconstruction
    MM=scipy.io.loadmat('mixture_models/'+MM_name)
    alphas=np.reshape(MM['alphas'],[-1])
    mus=MM['mus'].transpose()
    Sigmas=np.transpose(MM['sigmas'],(2,0,1))
    Us=np.transpose(MM['us'],(2,0,1))
    print(Us.shape)
    bs=MM['bs'].transpose()
    K=alphas.shape[0]
    d=Us.shape[1]
    Sigmas_full=np.zeros((K,d,d))
    mus_full=np.zeros((K,d))
    for k in range(K):
        Sigmas_full[k]=np.linalg.inv(1/sigma_sq *(np.eye(d)-Us[k].dot(Us[k].transpose()))+Us[k].dot(np.linalg.inv(Sigmas[k]).dot(Us[k].transpose())))
        mus_full[k]=Sigmas_full[k].dot(Us[k].dot(np.linalg.inv(Sigmas[k]).dot(mus[k])))+bs[k]
    img_m=scipy.io.loadmat('./imgs_superres/'+img_name+'.mat')
    img_obs=img_m['img_lr']
    img_ground_truth=img_m['img_hr']
    if len(img_obs.shape)==2:
        reconstructed=reconstruction_gauss(img_obs,alphas,mus_full,Sigmas_full,q=q,tau=tau)
    elif len(img_obs.shape)==3:      
        reconstructed=reconstruction_gauss_3D(img_obs,alphas,mus_full,Sigmas_full,q=q,tau=tau)
    else:
        raise ValueError('Dimension of the observation has to be 2 or 3!')
    if not os.path.isdir('reconstructions'):
        os.mkdir('reconstructions')
    scipy.io.savemat('./reconstructions/'+img_name+'_'+MM_name,{'reconstructed':reconstructed})
    res_psnr=psnr(reconstructed,img_ground_truth)
    print(res_psnr)
    return res_psnr
