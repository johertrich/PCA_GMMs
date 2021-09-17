# This code belongs to the paper
#
# J. Hertrich, L. Nguyen, J-F. Aujol, D. Bernard, Y. Berthoumieu, G. Steidl.
# PCA Reduced Gaussian Mixture Models with Applications in Superresolution.
# Inverse Problems and Imaging, 2021.
#
# Please cite the paper, if you use the code.
#
import numpy as np
import imageio
import scipy.io
import os

def extract_joint_patches(img_hr,img_lr,p_lr,save_name=''):
    # Extracts patches from the 2D images img_hr and img_lr for superresolution and saves them as samples in the data directory
    # INPUTS:
    #       - img_hr            - 2D Numpy array. High resolution image
    #       - img_lr            - 2D Numpy array. Low resolution image
    #       - p_lr              - int. Patch size in the low resolution image
    #       - save_name         - string. Identifyer of the image (e.g. 'diam' or 'FS')
    # OUTPUT:
    #       - samps             - Samples
    if not os.path.isdir('data'):
        os.mkdir('data')
    magnif=img_hr.shape[1]//img_lr.shape[1]
    p_hr=magnif*p_lr
    x_coords=np.array(range(img_lr.shape[0]-p_lr+1))
    x_coords=np.tile(x_coords[:,np.newaxis],(1,img_lr.shape[1]-p_lr+1))
    y_coords=np.array(range(img_lr.shape[1]-p_lr+1))
    y_coords=np.tile(y_coords[np.newaxis,:],(img_lr.shape[0]-p_lr+1,1))
    xs=np.reshape(x_coords,[-1])
    n=len(xs)
    ys=np.reshape(y_coords,[-1])
    xs_hr=xs*magnif    
    ys_hr=ys*magnif
    inds_lr=np.reshape(ys+xs*img_lr.shape[0],[-1])
    inds_hr=np.reshape(ys_hr+xs_hr*img_hr.shape[0],[-1])
    img_hr_vec=np.reshape(img_hr,[-1])
    img_lr_vec=np.reshape(img_lr,[-1])

    
    patch_lr=np.array(range(p_lr))
    patch_lr=np.tile(patch_lr[:,np.newaxis],(1,p_lr))*img_lr.shape[0]+np.tile(patch_lr[np.newaxis,:],(p_lr,1))
    patch_hr=np.array(range(p_hr))
    patch_hr=np.tile(patch_hr[:,np.newaxis],(1,p_hr))*img_hr.shape[0]+np.tile(patch_hr[np.newaxis,:],(p_hr,1))
    patches_lr_ind=np.tile(patch_lr[:,:,np.newaxis],(1,1,n))+np.tile(inds_lr,(p_lr,p_lr,1))
    patches_hr_ind=np.tile(patch_hr[:,:,np.newaxis],(1,1,n))+np.tile(inds_hr,(p_hr,p_hr,1))
    patches_lr=img_lr_vec[patches_lr_ind]
    patches_hr=img_hr_vec[patches_hr_ind]
    samps=[]
    samps_hr=[]
    for i in range(n):
        samp1=patches_hr[:,:,i]
        samp2=patches_lr[:,:,i]
        samp1=np.reshape(samp1,[-1])
        samp2=np.reshape(samp2,[-1])
        samps.append(np.concatenate([samp1,samp2],0))
        samps_hr.append(samp1)
    samps=np.stack(samps).transpose()
    print(samps.shape)
    samps_hr=np.stack(samps_hr).transpose()
    np.save('data/samples_hr'+save_name+'.npy',samps_hr)
    np.save('data/samples_joint_downsampled'+save_name+'.npy',samps) 
    return samps

def extract_joint_patches_3D(img_hr,img_lr,p_lr,save_name='',n_max=1000000):
    # Extracts patches from the 3D images img_hr and img_lr for superresolution and saves them as samples in the data directory.
    # INPUTS:
    #       - img_hr            - 3D Numpy array. High resolution image
    #       - img_lr            - 3D Numpy array. Low resolution image
    #       - p_lr              - list of ints with length 3. Patch size in the low resolution image
    #       - save_name         - string. Identifyer of the image (e.g. 'diam' or 'FS')
    #       - n_max             - maximal number of samples. Default: 1e6.
    # OUTPUT:
    #       - samps             - Samples
    if not os.path.isdir('data'):
        os.mkdir('data')
    if type(p_lr)==int:
        p_lr=[p_lr]*3
    magnif=np.zeros(3,dtype=np.int64)
    magnif[0]=img_hr.shape[0]//img_lr.shape[0]
    magnif[1]=img_hr.shape[1]//img_lr.shape[1]
    magnif[2]=img_hr.shape[2]//img_lr.shape[2]
    p_hr=magnif*p_lr
    n=(img_lr.shape[0]-p_lr[0]+1)*(img_lr.shape[1]-p_lr[1]+1)*(img_lr.shape[2]-p_lr[2]+1)
    print(n)
    n_max=min(n_max,n)
    take_inds=np.random.permutation(n)
    take_inds=take_inds[:n_max]
    take_inds=np.sort(take_inds)
    up_lr=0
    up_hr=0
    front_lr=0
    front_hr=0
    samps=[]
    samps_hr=[]
    ind=0
    i=0
    while up_lr+p_lr[0]<=img_lr.shape[0]:
        left_lr=0
        left_hr=0
        while left_lr+p_lr[1]<=img_lr.shape[1]:
            front_lr=0
            front_hr=0
            while front_lr+p_lr[2]<=img_lr.shape[2]:
                if i==n_max:
                    break
                if take_inds[i]==ind:
                    patch_lr=img_lr[up_lr:(up_lr+p_lr[0]),left_lr:(left_lr+p_lr[1]),front_lr:(front_lr+p_lr[2])]
                    patch_hr=img_hr[up_hr:(up_hr+p_hr[0]),left_hr:(left_hr+p_hr[1]),front_hr:(front_hr+p_hr[2])]
                    samp1=np.reshape(patch_hr,[-1])
                    samp2=np.reshape(patch_lr,[-1])
                    samp1=np.reshape(samp1,[-1])
                    samp2=np.reshape(samp2,[-1])
                    samps.append(np.concatenate([samp1,samp2],0))
                    samps_hr.append(samp1)
                    i+=1
                ind+=1
                front_lr+=1
                front_hr+=magnif[2]
            left_lr+=1
            left_hr+=magnif[1]
        up_hr+=magnif[0]
        up_lr+=1
    print(left_lr)
    print(up_lr)
    print(front_lr)
    samps=np.stack(samps).transpose()
    print(samps.shape)
    samps_hr=np.stack(samps_hr).transpose()
    np.save('data/samples_hr'+save_name+'.npy',samps_hr)
    np.save('data/samples_joint_downsampled'+save_name+'.npy',samps)   
    return samps
