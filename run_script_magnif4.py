# This code belongs to the paper
#
# L. Nguyen, J. Hertrich, J-F. Aujol, D. Bernard, Y. Berthoumieu, G. Steidl.
# PCA Reduced Gaussian Mixture Models with Applications in Superresolution.
# ArXiv preprint arXiv:2009.07520, 2020
#
# Please cite the paper, if you use the code.
#
from funs.extract_patches import extract_joint_patches
import scipy.io
from funs.initialization import initialize
import funs.EM_Gauss
from funs.reconstruction import superresolution_gauss, superresolution_PCA
import funs.EM_PCA
import numpy as np

use_pretrained_model=True

names=['goldhill4','diam4','FS4']
s_=[4,8,12,16,20]
K=100
tau=4
magnif=4
psnrs=np.zeros((len(names),len(s_)+1))
count_out=0
for name in names:
    if not use_pretrained_model:
        data=scipy.io.loadmat('learn_imgs/'+name+'.mat')
        img_hr=data['img_hr'].astype(np.float64)
        img_lr=data['img_lr'].astype(np.float64)
        extract_joint_patches(img_hr,img_lr,tau,save_name='_'+name)
        initialize('_joint_downsampled_'+name,K)
        funs.EM_Gauss.run_MM('_joint_downsampled_'+name)
    psnrs[count_out,0]=superresolution_gauss('MM_EM_gauss_joint_downsampled_'+name+'.mat',name,q=magnif,tau=tau)
    count_in=1
    for s in s_:
        if not use_pretrained_model:
            funs.EM_PCA.run_MM('_joint_downsampled_'+name,s)
        psnrs[count_out,count_in]=superresolution_PCA('MM_PCA_'+str(s)+'_joint_downsampled_'+name+'.mat',name,q=magnif,tau=tau)
        count_in+=1
    count_out+=1
print(psnrs)
