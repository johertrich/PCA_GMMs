# PCA reduced Gaussian Mixture Models with Application in Superresolution

This code belongs to the paper [1]. Please cite the paper, if you use this code.

It is available at  
https://arxiv.org/abs/2009.07520.

The repository contains an implementation of the superresolution method from [3] using the EM algorithm for PCA reduced Gaussian mixture models (GMMs) as introduced in [1].
As comparison also the EM algorithm for HDDC [4] is implemented.
For learning these PCA reduced GMMs, we use the implementation framework from [2], which is available at https://github.com/johertrich/Inertial-Stochastic-PALM.
In particular, it contains the code to reproduce the numerical examples from the paper [1].

For questions and bug reports, please contact Johannes Hertrich (j.hertrich(at)math.tu-berlin.de).

## CONTENTS

1. REQUIREMENTS
2. USAGE
3. CLASSES AND FUNCTIONS
4. EXAMPLES
5. REFERENCES

## 1. REQUIREMENTS

The code requires several Python packages. We tested the code with Python 3.7.9 and the following package versions:

Tensorflow 2.2.0  
Numpy 1.18.7  
Scipy 1.5.2  
Imageio 2.9.0

Usually the code is also compatible with some other versions of the corresponding Python packages.

## 2. USAGE

Download the code and import `funs.EM_PCA` for using the EM algorithm for PCA reduced Gaussian mixture models.

The scripts `run_script.py`, `run_script_magnif4.py` and `run_script_3D.py` implment the numerical examples from [1]. 
Note that for executing `run_script_3D.py` you have to include the 3D data into the directories `learn_imgs` and `imgs_superres`.
The data is not included in this repository due to the large size of 3D images.

For the usage of the functions in `funs.EM_PCA` we refer to the numerical examples and the documentation in Section 3.

## 3. CLASSES AND FUNCTIONS

In this section we provide a short description of the functions, which are implemented in this repository and the corresponding
input and output parameters. Note that `funs.palm_algs` belongs to the code of [2]. Thus, we refer for its documentation
to https://github.com/johertrich/Inertial-Stochastic-PALM.

### In `funs.extract_patches`

#### function `extract_joint_patches`

Extracts patches from the 2D images img\_hr and img\_lr for superresolution and saves them as samples in the data directory.  
Inputs:

- **img_hr** - 2D Numpy array. High resolution image
- **img_lr** - 2D Numpy array. Low resolution image
- **p_lr** - int. Patch size in the low resolution image
- **save_name** - string. Identifyer of the image (e.g. 'diam' or 'FS')

Output:

- **samps** - Samples

#### function `extract_joint_patches_3D`

Extracts patches from the 3D images img\_hr and img\_lr for superresolution and saves them as samples in the data directory.  
Inputs:

- **img_hr** - 3D Numpy array. High resolution image
- **img_lr** - 3D Numpy array. Low resolution image
- **p_lr** - int for isotropic patches, list of ints for anisotropic patches. Patch size in the low resolution image
- **save_name** - string. Identifyer of the image (e.g. 'diam' or 'FS')
- **n_max** - optional. Maximal number of samples. Default: 1e6.

Output:

- **samps** - Samples

### In `funs.initialization`

#### function `init_MM`

Initialization for learning a PCA reduced GMM.  
Inputs:

- **X** - d x n Numpy array conataining the Samples
- **K** - int. Number of classes
- **regularize** - float. regularization parameter

Outputs:

- **Initial parameters** alphas, mus, sigmas, Us

#### function `initialize`

Loads samples from the data directory, computes an initialization and saves it again in the data directory.
Inputs:

- **name** - String. Image identifyer.
- **K** - int. Number of classes.
- **regularize** - float. regularization parameter.

### In `funs.EM_Gauss`

#### function `opt_em`

Implements the EM algorithm for estimating the parameters of a Gaussian Mixture Models.  
Inputs:
Required:

- **samps** - N x n numpy array, where samps[i] contains the i-th data point
- **Initial parameters** - alphas\_init (numpy array of length K), mus\_init (K x n numpy array), Sigmas\_init (K x n x n numpy array)

Optional:

- **regularize** - regularization constant for the covariance matrix. Default value: 1e-5.
- **steps** - number of steps. Default value: 100.
- **batch_size** - Parameter for the computation order. Does not effect the results, but the execution time. Default: 10000

Outputs:  

Resulting parameters:
- **alphas** - numpy array of length K
- **mus** - K x n numpy array
- **Sigmas** - K x n x n numpy array  

Computation times:
- **time_E** - average computation time for the E-step
- **time_M** - average computation time for the M-step

#### function `run_MM`

Loads the initialization declared by the parameter name, runs the EM algorithm to compute
the ML-estimator of the parameters of a GMM and saves the parameters.  
Inputs:

- **name** - string. name of the intialization.
- **batch_size** - optional batch\_size paramter of opt\_em. Default value: 10000.

Outputs:  
- **time_E** - average computation time for the E-step
- **time_M** - average computation time for the M-step

### In `funs.EM_PCA`

#### function `opt_em_pca`

Implements the EM algorithm for estimating the parameters of a PCA-GMM model.  
Iputs:
Required:
- **samps** - N x n numpy array, where samps[i] contains the i-th data point
- **Initial parameters** - alphas\_init (numpy array of length K), Sigmas\_init (K x d x d numpy array), Us\_init (K x n x d numpy array), bs\_init (K x n numpy array)
- **sigma_sq** - weight parameter for the PCA

Optional:

- **regularize** - regularization constant for the covariance matrix. Default value: 1e-5.
- **steps** - number of steps. Default value: 100.
- **batch_size** - Parameter for the computation order. Does not effect the results, but the execution time. Default: 10000
- **learn_sigma_sq** - True for learning sigma\_sq within the EM algorithm, False for fixed sigma\_sq. Default: False

Outputs:  
Resulting parameters:  
- **alphas** - numpy array of length K
- **Sigmas** - K x d x d numpy array
- **Us** - K x n x d numpy array
- **bs** - K x n numpy array  

Computation times:
- **time_E** - average computation time for the E-step
- **time_M** - average computation time for the M-step

#### function `initialize_U`

Refinement of the initialization of U by performing one step of the EM algorithm with fixed alpha,mu,Sigma  
Inputs:  
Required:

- **samps** - n x d matrix with samples
- **alphas**, **mus**, **Sigmas** - parameters of a d dimensional GMM
- **Us** - initialization of Us
- **sigma_sq** - weight parameter for the PCA

Optional:

- **batch_size** - does not effect the results. Larger batch size leads into a faster execution but higher RAM requirements. Default: 10000
- **regularize** - regularization parameter. Default: 1e-5

Outputs:
- **Us** - Initialization for the Us.
- **bs** - Initialization for the bs.
- **Sigmas** - Dimension reduced initialization for Sigmas.

#### function `initialize_one_U`

Generates an initialization for one U based on the PCA of all data points.  
Inputs:

- **samps** - samples
- **s** - reduced dimension

Output: Initialization U

#### function `load_initialization`

Loads an initialization
Input:
- **name** - String. Image identifyer.

Outputs:
- **samples**
- initial data **alphas**, **mus**, **Sigmas**, **Us**
- Number of classes **K**.

#### function `run_MM`

Loads the initialization declared by the parameter name, runs the EM algorithm to compute
the ML-estimator of the parameters of a PCA-reduced MM GMM and saves the parameters.  
Inputs:  
Required:

- **name** - string. name of the intialization.
- **s** - reduced dimension

Optional:

- **batch_size** - batch\_size paramter of opt\_em. Default value: 10000.
- **use_bias** - declares whether to use the bias in the PCA.
- **learn_sigma_sq** - True for learning sigma\_sq within the EM algorithm, False for fixed sigma\_sq. Default: False

Outputs:  
- **time_E** - average computation time for the E-step
- **time_M** - average computation time for the M-step

### In `funs.EM_HDDC`

Similar functions as in `funs.EM_PCA`. The only difference is that the EM-algorithm is perforemed for a HDDC [4] model and not for a PCA-GMM model [1].

### In `funs.reconstruction`

#### function `reconstruction_gauss`

Reconstruction of a 2D high resolution image based on a low resolution observation and a joint GMM using the
method from citation [3] of the readme file.  
Inputs:  
Required:

- **obs** - Observation.
- **alphas**, **mus**, **Sigmas** - Parameters of the GMM.

Optional:

- **avg_gam** - Parameter for the patch averaging. Default=0.12
- **tau** - patch size in the low resolution image. Default: 4
- **q** - magnification factor. Default: 2

Output: Reconstructed high resolution image.

#### function `reconstruction_gauss_3D`

Reconstruction of a 3D high resolution image based on a low resolution observation and a joint GMM using the
method from citation [3] of the readme file.  
Inputs:  
Required:

- **obs** - Observation.
- **alphas**, **mus**, **Sigmas** - Parameters of the GMM.

Optional:

- **avg_gam** - Parameter for the patch averaging. Default=0.12
- **tau** - patch size in the low resolution image. Default: 4
- **q** - magnification factor. Default: 2
 
Output: Reconstructed high resolution image.

#### function `psnr`

Computes the PSNR of img with reference ref.
Inputs:

- **img** - image.
- **ref** - reference.

OUTPUT: psnr.

#### function `superresolution_gauss`

Loads a GMM and a low resolution image and reconstructs the corresponding high resolution image using the method
from citation [3] from the readme file and saves the reconstruction.  
Inputs:  
Required:

- **MM_name** - identifyer of the GMM.
- **img_name** - identifyer of the image.

Optional:

- **q** - magnification factor.
- **tau** - patch size in the low resolution image.

Output: psnr of the reconstruction

#### function `superresolution_PCA`

Loads a PCA-reduced GMM and a low resolution image and reconstructs the corresponding high resolution image using the method
from citation [3] from the readme file and saves the reconstruction.  
Inputs:  
Required:

- **MM_name** - identifyer of the PCA-reduced GMM.
- **img_name** - identifyer of the image.

Optional:

- **q** - magnification factor.
- **tau** - patch size in the low resolution image.
- **sigma_sq** - parameter sigma^2 from the PCA-reduced GMM

Output: psnr of the reconstruction

## 4. EXAMPLES

### 2D-Superresolution with magnification 2

The script `run_script.py` contains the implementation of the first numerical example from [1]. That is, it applies the 
superresolution method for 2D-images and a magnification factor of 2 with PCA reduced GMMs with reduced dimension 4, 8, 12, 16 and 20.

### 2D-Superresolution with magnification 4

The script `run_script_magnif4.py` contains the implementation of the second numerical example from [1]. That is, it applies the 
superresolution method for 2D-images and a magnification factor of 4 with PCA reduced GMMs with reduced dimension 4, 8, 12, 16 and 20.

### 3D-Superresolution with magnification 2

The script `run_script_magnif4.py` contains the implementation of the third numerical example from [1]. That is, it applies the 
superresolution method for 3D-images and a magnification factor of 2 with PCA reduced GMMs with reduced dimension 20, 40, 60.

Note that the data for this example is not contained in the repository because its large size.

## 5. REFERENCES

[1] J. Hertrich, D. P. L. Nguyen, J.-F. Aujol, D. Bernard, Y. Berthoumieu, A. Saadaldin and G. Steidl.  
PCA reduced Gaussian mixture models with application in superresolution.  
ArXiv preprint arXiv:2009.07520, 2020.

[2] J. Hertrich and G. Steidl.  
Inertial Stochastic PALM (iSPALM) and Applications in Machine Learning.  
ArXiv preprint arXiv:2005.02204, 2020.

[3] P. Sandeep and T. Jacob.  
Single image super-resolution using a joint GMM method.  
IEEE Transactions on Image Processing, 25(9):4233–4244, 2016.

[4] C. Bouveyron, S. Girard and C. Schmid.  
High-dimensional data clustering.  
Computational statistics & data analysis 52.1: 502-519, 2007.

