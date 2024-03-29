# This code belongs to the paper
#
# J. Hertrich, L. Nguyen, J-F. Aujol, D. Bernard, Y. Berthoumieu, G. Steidl.
# PCA Reduced Gaussian Mixture Models with Applications in Superresolution.
# Inverse Problems and Imaging, 2021.
#
# Please cite the paper, if you use the code.
#
from funs.extract_patches import extract_joint_patches_3D
import scipy.io
from funs.initialization import initialize
import funs.EM_Gauss
from funs.reconstruction import superresolution_gauss, superresolution_PCA
import funs.EM_PCA
import funs.EM_HDDC
import numpy as np

use_pretrained_model=True

names=['FS3D','diam3D']
s_=[20,40,60]
K=100
tau=4
magnif=2
batch_size=5000
psnrs_EM=np.zeros((len(names),))
times_E_EM=np.zeros((len(names),))
times_M_EM=np.zeros((len(names),))
psnrs=np.zeros((len(names),len(s_)))
times_E=np.zeros((len(names),len(s_)))
times_M=np.zeros((len(names),len(s_)))
psnrs_hddc=np.zeros((len(names),len(s_)))
times_E_hddc=np.zeros((len(names),len(s_)))
times_M_hddc=np.zeros((len(names),len(s_)))
count_out=0
for name in names:
    if not use_pretrained_model and False:
        data=scipy.io.loadmat('learn_imgs/'+name+'.mat')
        img_hr=data['img_hr'].astype(np.float64)
        img_lr=data['img_lr'].astype(np.float64)
        extract_joint_patches_3D(img_hr,img_lr,tau,save_name='_'+name,n_max=1000000)
        del img_hr
        del img_lr
        initialize('_joint_downsampled_'+name,K)
        time_E,time_M=funs.EM_Gauss.run_MM('_joint_downsampled_'+name,batch_size=batch_size)
        times_E_EM[count_out]=time_E
        times_M_EM[count_out]=time_M
    psnrs_EM[count_out]=superresolution_gauss('MM_EM_gauss_joint_downsampled_'+name+'K'+str(K)+'.mat',name,q=magnif,tau=tau)
    count_in=0
    for s in s_:
        if not use_pretrained_model:
            time_E,time_M=funs.EM_PCA.run_MM('_joint_downsampled_'+name,s,batch_size=batch_size,learn_sigma_sq=False,palm_steps=300)
            times_E[count_out,count_in]=time_E
            times_M[count_out,count_in]=time_M
            time_E,time_M=funs.EM_HDDC.run_MM('_joint_downsampled_'+name,s,batch_size=batch_size)
            times_E_hddc[count_out,count_in]=time_E
            times_M_hddc[count_out,count_in]=time_M
        psnrs[count_out,count_in]=superresolution_PCA('MM_PCA_'+str(s)+'_joint_downsampled_'+name+'K'+str(K)+'.mat',name,q=magnif,tau=tau)
        psnrs_hddc[count_out,count_in]=superresolution_PCA('MM_HDDC_'+str(s)+'_joint_downsampled_'+name+'K'+str(K)+'.mat',name,q=magnif,tau=tau)
        count_in+=1
    count_out+=1
print('Standard GMM:')
print('PSNRs:')
print(psnrs_EM)
if not use_pretrained_model:
    print('Times E-step:')
    print(times_E_EM)
    print('Times M-step:')
    print(times_M_EM)
print('\nPCA-GMM, fixed sigma:')
print(psnrs)
if not use_pretrained_model:
    print('Times E-step:')    
    print(times_E)
    print('Times M-step:')
    print(times_M)
print('\nHDDC:')
print(psnrs_hddc)
if not use_pretrained_model:
    print('Times E-step:')    
    print(times_E_hddc)
    print('Times M-step:')
    print(times_M_hddc)
