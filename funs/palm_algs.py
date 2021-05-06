# This code belongs to the paper
#
# J. Hertrich and G. Steidl. 
# Inertial Stochastic PALM (iSPALM) and Applications in Machine Learning.
# ArXiv preprint arXiv:2005.02204, 2020.
#
# Please cite the paper, if you use the code.

from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import time
import os

class PALM_Model(Model):
    # Models functions of the Form F(x_1,...,x_n)=H(x_1,...,x_n)+\sum_{i=1}^n f_i(x_i),
    # where H is continuously differentiable and f_i is lower semicontinuous.
    # Inputs of the Constructor:
    #       initial_values          - List of numpy arrays for initialize the x_i
    #       dtype                   - model type
    def __init__(self,initial_values,dtype='float32'):
        super(PALM_Model,self).__init__(dtype=dtype)
        self.num_blocks=len(initial_values)
        self.model_type=tf.constant(initial_values[0]).dtype
        self.prox_funs=[]
        self.X=[]
        self.H=lambda X,batch: 0
        self.f=[]
        id_prox=lambda arg,lam:tf.identity(arg)
        for i in range(self.num_blocks):
            self.prox_funs.append(id_prox)
            init=tf.constant_initializer(initial_values[i])
            self.X.append(self.add_weight("X"+str(i),initializer=init,shape=initial_values[i].shape,trainable=True))
            self.f.append(lambda X: 0)

    def call(self,batch,X=None):
        if X is None:
            X=self.X
        #print(batch)
        return self.H(X,batch)
    
    def objective(self,X=None,batch=None):
        if X is None:
            X=self.X
        out=0.
        out+=self(batch)
        for i in range(self.num_blocks):
            out+=self.f[i](X[i])
        return out


    @tf.function
    def grad_hess_batch(self,batch,i):
        with tf.GradientTape(persistent=True) as tape:
            val=self(batch)
            g=tape.gradient(val,self.X[i])
            if isinstance(g,tf.IndexedSlices):
                g2=g.values
            else:
                g2=g
            grad_sum=tf.reduce_sum(tf.multiply(g2,g2))
        h=tape.gradient(grad_sum,self.X[i])   
        fac=0.5/tf.sqrt(grad_sum)
        h*=fac 
        g=tf.identity(g)
        h=tf.identity(h)
        return g,h,val
    

    @tf.function
    def grad_batch(self,batch,i):
        with tf.GradientTape() as tape:
            val=self(batch)
        g=tape.gradient(val,self.X[i])
        return g,val

class PALM_Optimizer:
    # Optimizer class for functions implemented as PALM_Model.
    # Constructor arguments:
    #       - model                 - PALM_Model for the objective function
    #       - steps_per_epoch       - int. maximal numbers of PALM/iPALM/SPRING/iSPALM steps in one epoch
    #                                 Default value: Infinity, that is pass the whole data set in each epoch
    #       - data                  - Numpy array of type model.model_type. Information to choose the minibatches. 
    #                                 Required for SPRING and iSPALM. 
    #                                 To run PALM/iPALM on functions, which are not data based, use data=None.
    #                                 For SPRING and iSPALM a value not equal to None is required.
    #                                 Default value: None
    #       - test_data             - Numpy array of type model.model_type. Data points to evaluate the objective   
    #                                 function in the test step after each epoch. 
    #                                 For test_data=None, the function uses data as test_data.
    #                                 Default value: None
    #       - batch_size            - int. If data is None: No effect. Otherwise: batch_size for data driven models.
    #                                 Default value: 1000
    #       - method                - String value, which declares the optimization method. Valid choices are: 'PALM', 
    #                                 'iPALM', 'SPRING-SARAH' and 'iSPALM-SARAH'. Raises an error for other inputs.
    #                                 Default value: 'iSPALM-SARAH'
    #       - inertial_step_size    - float variable. For method=='PALM' or method=='SPRING-SARAH': No effect.      
    #                                 Otherwise: the inertial parameters in iPALM/iSPALM are chosen by 
    #                                 inertial_step_size*(k-1)/(k+2), where k is the current step number.
    #                                 For inertial_step_size=None the method choses 1 for PALM and iPALM, 0.5 for 
    #                                 SPRING and 0.4 for iSPALM.
    #                                 Default value: None
    #       - step_size             - float variable. The step size parameters tau are choosen by step_size*L where L 
    #                                 is the estimated partial Lipschitz constant of H.
    #       - sarah_seq             - This input should be either None or a sequence of uniformly on [0,1] distributed
    #                                 random float32-variables. The entries of sarah_seq determine if the full
    #                                 gradient in the SARAH estimator is evaluated or not.
    #                                 For sarah_seq=None such a sequence is created inside this function.
    #                                 Default value: None
    #       - sarah_p               - float in (1,\infty). Parameter p for the sarah estimator. If sarah_p=None the 
    #                                 method uses p=20
    #                                 Default value: None
    #       - test_batch_size       - int. test_batch_size is the batch size used in the test step and in the steps
    #                                 where the full gradient is evaluated. This does not effect the algorithm itself.
    #                                 But it may effect the runtime. For test_batch_size=None it is set to batch_size.
    #                                 If test_batch_size<batch_size and method=SPRING-SARAH or method=iSPALM-SARAH,
    #                                 then also in the steps, where not the full gradient is evaluated only batches
    #                                 of size test_batch_size are passed through the function H.
    #                                 Default value: None
    #       - ensure_full           - Boolean or int. For method=='SPRING-SARAH' or method=='iSPALM-SARAH': If
    #                                 ensure_full is True, we evaluate in the first step of each epoch the full
    #                                 gradient. We observed numerically, that this sometimes increases stability and
    #                                 convergence speed of SPRING and iSPALM. For PALM and iPALM: no effect.
    #                                 If a integer value p is provided, every p-th step is forced to be a full step
    #                                 Default value: False
    #       - estimate_lipschitz    - Boolean. If estimate_lipschitz==True, the Lipschitz constants are estimated based
    #                                 on the first minibatch in all steps, where the full gradient is evaluated.
    #                                 Default: True
    #       - backup_dir            - String or None. If a String is provided, the variables X[i] are saved after
    #                                 every epoch. The weights are not saved if backup_dir is None.
    #                                 Default: 'backup'
    #       - mute                  - Boolean. For mute=True the evaluation of the objective function and all prints
    #                                 will be suppressed.
    #                                 Default: False
    #
    # Provides the following functions:
    #       - evaluate_objective    - evaluates the objective function for the current parameters
    #       - precompile            - compiles parts of the functions to tensorflow graphs to compare runtimes
    #       - exec_epoch            - executes one epoch of the algorithm
    #       - optimize              - executes a fixed number of epochs
    # 
    # The following functions should not be called directly
    #       - train_step_full       - performs one step of the algorithm, where the full gradient is evaluated
    #       - train_step_not_full   - performs one step of the algorithm, where not the full gradient is evaluated
    #
    def __init__(self,model,steps_per_epoch=np.inf,data=None,test_data=None,batch_size=1000,method='iSPALM-SARAH',inertial_step_size=None,step_size=None,sarah_seq=None,sarah_p=None,test_batch_size=None,ensure_full=False,estimate_lipschitz=False,backup_dir='backup',mute=False):
        if not (method=='PALM' or method=='iPALM' or method=='SPRING-SARAH' or method=='iSPALM-SARAH'):
            raise ValueError('Method '+str(method)+' is unknown!')
        self.method=method
        self.batch_size=batch_size
        self.model=model
        if test_batch_size is None:
            self.test_batch_size=batch_size
        else:
            self.test_batch_size=test_batch_size
        if backup_dir is None:
            self.backup=False
        else:
            self.backup=True
            self.backup_dir=backup_dir
            if not os.path.isdir(backup_dir):
                os.mkdir(backup_dir)
        if step_size is None:
            if method=='PALM':
                self.step_size=1.
            elif method=='iPALM':
                self.step_size=1.
            elif method=='SPRING-SARAH':
                self.step_size=0.5
            elif method=='iSPALM-SARAH':
                self.step_size=0.4
        else:
            self.step_size=step_size
        self.test_d=True
        if test_data is None:
            self.test_d=False
            self.test_data=data
        else:
            self.test_data=test_data
        self.data=data

        if method=='SPRING-SARAH' or method=='iSPALM-SARAH':
            self.sarah_seq=sarah_seq
            if sarah_p is None:
                sarah_p=20
            self.sarah_p_inv=1./sarah_p
        self.step_size=tf.constant(self.step_size,dtype=model.model_type)
        if method=='iSPALM-SARAH' or method=='SPRING-SARAH':
            if data is None:
                raise ValueError('Batch information is required for SPRING and iSPALM!')
            self.my_ds=tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)
        if method=='PALM' or method=='SPRING-SARAH':
            inertial_step_size=0
        if inertial_step_size is None:
            if method=='iPALM':
                inertial_step_size=1.
            elif method=='iSPALM-SARAH':
                inertial_step_size=0.5
        self.inertial_step_size=tf.constant(inertial_step_size,dtype=model.model_type)
        if not self.data is None:
            dat=self.data[:1]
            self.batch_version=True
            self.normal_ds=tf.data.Dataset.from_tensor_slices(self.data).batch(self.test_batch_size)
            self.test_ds=tf.data.Dataset.from_tensor_slices(self.test_data).batch(self.test_batch_size)
            self.n=data.shape[0]
        else:
            dat=None
            self.test_d=False
            self.batch_version=False
        self.estimate_lipschitz=estimate_lipschitz
        self.steps_per_epoch=steps_per_epoch
        model(dat)
        X_vals=[]
        for i in range(model.num_blocks):
            X_vals.append(model.X[i].numpy())
        self.model_for_old_values=PALM_Model(X_vals,dtype=model.dtype)
        self.model_for_old_values.H=model.H
        if type(ensure_full)==int:
            self.full_steps=ensure_full
            self.ensure_full=False
        else:
            self.full_steps=np.inf    
            self.ensure_full=ensure_full
        self.small_batches=self.batch_size>self.test_batch_size

        self.step=0
        self.component=0
        self.grads=[None]*model.num_blocks
        self.old_args=[None]*model.num_blocks
        self.mute=mute
        if not mute:
            print('evaluate objective')
            if self.batch_version and self.test_d:
                obj,train_obj=self.eval_objective()
                self.test_vals=[obj.numpy()]    
                self.train_vals=[train_obj.numpy()]
                template='Initial objective training: {0:2.4f} testing: {1:2.4f}'
                print(template.format(self.train_vals[0],self.test_vals[0]))
            else:
                self.test_vals=[self.eval_objective().numpy()]
                template='Initial objective: {0:2.4f}'
                print(template.format(self.test_vals[0]))
        self.my_time=0.
        self.my_times=[0.]
        self.epoch=0

    #@tf.function
    def eval_objective(self):
        if self.batch_version:
            obj=tf.constant(0.,dtype=self.model.model_type)
            for batch in self.test_ds:
                obj+=self.model.objective(batch=batch)
            if self.test_d:
                train_obj=tf.constant(0.,dtype=self.model.model_type)
                for batch in self.normal_ds:
                    train_obj+=self.model.objective(batch=batch)
                return obj,train_obj
            return obj
        else:
            return self.model.objective()          

    def train_step_full(self,step,i):
        extr=self.inertial_step_size*(step-1.)/(step+2.)
        Xi_save=tf.identity(self.model.X[i])
        self.model.X[i].assign_sub(extr*(self.model_for_old_values.X[i]-self.model.X[i]))
        old_arg=tf.identity(self.model.X[i])
        if self.batch_version:
            grad=tf.zeros_like(self.model.X[i])
            hess=tf.zeros_like(self.model.X[i])
            eval_hess=True
            for batch in self.normal_ds:
                if eval_hess or not self.estimate_lipschitz:
                    g,h,val=self.model.grad_hess_batch(batch,i)
                    grad+=g
                    hess+=h
                    self.eval_hess=False
                else:
                    g,val=self.model.grad_batch(batch,i)
                    grad+=g
        else:
            grad,hess,val=self.model.grad_hess_batch(None,i)
        if tf.reduce_any(tf.math.is_nan(grad)):
            print(model.X)
            raise ValueError('NaNs appeard!')
        Lip=tf.sqrt(tf.reduce_sum(tf.multiply(hess,hess)))        
        if self.estimate_lipschitz:        
            Lip*=self.n*1.0/self.test_batch_size
        tau_i=tf.identity(Lip)
        tau_i=tf.math.multiply_no_nan(tau_i,1.-tf.cast(tf.math.is_nan(Lip),dtype=self.model.model_type))+1e10*tf.cast(tf.math.is_nan(Lip),dtype=self.model.model_type)
        self.model.X[i].assign(self.model.prox_funs[i](self.model.X[i]-grad/tau_i*self.step_size,tau_i/self.step_size))
        self.model_for_old_values.X[i].assign(Xi_save)
        return grad,old_arg,val

    def train_step_not_full(self,step,grads,batch,old_arg,i):
        extr=self.inertial_step_size*(step-1.)/(step+2.)
        Xi_save=tf.identity(self.model.X[i])
        self.model.X[i].assign_sub(extr*(self.model_for_old_values.X[i]-self.model.X[i]))
        old_arg_new=tf.identity(self.model.X[i])
        if self.small_batches:
            step_ds=tf.data.Dataset.from_tensor_slices(batch).batch(self.test_batch_size)
            g=tf.zeros_like(self.model.X[i])
            h=tf.zeros_like(self.model.X[i])            
            for small_batch in step_ds:
                g_b,h_b,val=self.model.grad_hess_batch(small_batch,i)
                g+=g_b
                h+=h_b
        else:
            g,h,val=self.model.grad_hess_batch(batch,i)
        Lip=tf.sqrt(tf.reduce_sum(tf.multiply(h,h)))
        tau_i=self.n*1.0/self.batch_size*tf.identity(Lip)
        tau_i=tf.math.multiply_no_nan(tau_i,1.-tf.cast(tf.math.is_nan(Lip),dtype=self.model.model_type))+1e10*tf.cast(tf.math.is_nan(Lip),dtype=self.model.model_type)
        self.model_for_old_values.X[i].assign(old_arg)
        if self.small_batches:
            g_o=tf.zeros_like(self.model.X[i])
            for small_batch in step_ds:
                g_o+=self.model_for_old_values.grad_batch(small_batch,i)
        else:
            g_o=self.model_for_old_values.grad_batch(batch,i)
        if isinstance(g,tf.IndexedSlices):
            g_diff=tf.IndexedSlices(self.n*1.0/self.batch_size*(g.values-g_o.values),g.indices,g.dense_shape)
        else:
            g_diff=self.n*1.0/self.batch_size*(g-g_o)
        grad=grads+g_diff
        if tf.reduce_any(tf.math.is_nan(grad)):
            print(self.model.X)
            raise ValueError('NaNs appeard!')
        self.model.X[i].assign(self.model.prox_funs[i](self.model.X[i]-grad/tau_i*self.step_size,tau_i/self.step_size))
        self.model_for_old_values.X[i].assign(Xi_save)
        return grad,old_arg_new,val

    def precompile(self):
        # Compiles parts of the functions to tensorflow graphs to compare runtimes.
        # INPUTS: None
        # OUTPUTS: None
        print('precompile functions for comparing runtimes')
        grads=[None]*self.model.num_blocks
        old_args=[None]*self.model.num_blocks
        X_save=[]
        for i in range(self.model.num_blocks):
            X_save.append(tf.identity(self.model.X[i]))
        print('Compile full steps')
        for i in range(self.model.num_blocks):
            out=self.train_step_full(tf.convert_to_tensor(1,dtype=self.model.model_type),i)
            grads[i]=out[0]
            old_args[i]=out[1]
        if self.method=='SPRING-SARAH' or self.method=='iSPALM-SARAH':
            print('Compile stochastic steps')
            for i in range(self.model.num_blocks):
                self.train_step_not_full(tf.convert_to_tensor(1,dtype=self.model.model_type),grads[i],self.data[:self.batch_size],old_args[i],i)
        for i in range(self.model.num_blocks):
            self.model.X[i].assign(X_save[i])
            self.model_for_old_values.X[i].assign(X_save[i])
        print('precompiling finished')

    def exec_epoch(self):
        # Executes one epoch of the algorithm
        # INPUTS: None
        # OUTPUTS: None
        if self.batch_version:
            count=0
            if self.method=='PALM' or self.method=='iPALM':
                self.step+=1
                for i in range(self.model.num_blocks):
                    tic=time.time()
                    out=self.train_step_full(tf.convert_to_tensor(self.step,dtype=self.model.model_type),i)
                    toc=time.time()-tic
                    self.my_time+=toc
                    val=out[-1]
            else:
                cont=True
                while cont:
                    if self.steps_per_epoch==np.inf:
                        cont=False
                    for batch in self.my_ds:
                        if self.step==0:
                            self.step+=1
                        if self.component==self.model.num_blocks:
                            self.step+=1
                            self.component=0
                            count+=1
                            if count>=self.steps_per_epoch:
                                cont=False
                                break
                        if self.sarah_seq is None:
                            rand_num=tf.random.uniform(shape=[1],minval=0,maxval=1,dtype=tf.float32)
                        else:
                            rand_num=self.sarah_seq[(self.step-1)*self.model.num_blocks+self.component]
                        full=False
                        if self.step==1 or rand_num<self.sarah_p_inv:
                            full=True
                        if count==0 and self.ensure_full:
                            full=True
                        if (self.step-1)%self.full_steps==0:
                            full=True
                        tic=time.time()
                        if full:
                            print('full step')
                            out=self.train_step_full(tf.convert_to_tensor(self.step,dtype=self.model.model_type),self.component)
                        else:
                            out=self.train_step_not_full(tf.convert_to_tensor(self.step,dtype=self.model.model_type),self.grads[self.component],batch,self.old_args[self.component],self.component)
                        val=out[-1]
                        toc=time.time()-tic
                        self.my_time+=toc
                        self.grads[self.component]=out[0]
                        self.old_args[self.component]=out[1]
                        self.component+=1
                
        else:
            self.step+=1
            for i in range(self.model.num_blocks):
                out=self.train_step_full(tf.convert_to_tensor(self.step,dtype=self.model.model_type),i)
                val=out[-1]
        if not self.mute:
            print('evaluate objective')
            if self.batch_version and self.test_d:
                obj,train_obj=self.eval_objective()
                template = 'Epoch {0}, Objective training: {1:2.4f}, Objective test: {2:2.4f}, Time: {3:2.2f}'
                print(template.format(self.epoch+1,train_obj,obj,self.my_time))
            else:
                obj=self.eval_objective()
                template = 'Epoch {0}, Objective: {1:2.4f}, Time: {2:2.2f}'
                print(template.format(self.epoch+1,obj,self.my_time))
        if self.backup:
            for i in range(self.model.num_blocks):
                self.model.X[i].numpy().tofile(self.backup_dir+'/epoch'+str(self.epoch+1)+'X'+str(i))
        if not self.mute:        
            if self.batch_version and self.test_d:
                self.train_vals.append(train_obj.numpy())
            self.test_vals.append(obj.numpy())
        self.my_times.append(self.my_time)
        self.epoch+=1
        return val
    
    def optimize(self,EPOCHS=10):
        # Executes a fixed number of epochs
        # Inputs:
        #       - EPOCHS            - Number of epochs.
        # Outputs:
        #       - my_times          - Cummulated execution time of the epochs
        #       - test_vals         - Objective function on the test set. If test_data is None, then the objective function
        #                             on the data set. Only returned if mute==False.
        #       - train_vals        - Objective function on the train set. Only returned if mute==False and test_data is not None.
        for epoch in range(EPOCHS):
            self.exec_epoch()
        if not self.mute:
            if self.batch_version and self.test_d:
                return self.my_times,self.test_vals,self.train_vals
            return self.my_times,self.test_vals
        return self.my_times
        

def optimize_PALM(model,EPOCHS=10,steps_per_epoch=np.inf,data=None,test_data=None,batch_size=1000,method='iSPALM-SARAH',inertial_step_size=None,step_size=None,sarah_seq=None,sarah_p=None,precompile=False,test_batch_size=None,ensure_full=False,estimate_lipschitz=False,backup_dir='backup',mute=False,stop_crit=-np.inf):
    # Minimizes the PALM_model using PALM/iPALM/SPRING-SARAH or iSPALM-SARAH.
    # Inputs:
    #       - model                 - PALM_Model for the objective function
    #       - EPOCHS                - int. Number of epochs to optimize
    #                                 Default value: 10
    #       - steps_per_epoch       - int. maximal numbers of PALM/iPALM/SPRING/iSPALM steps in one epoch
    #                                 Default value: Infinity, that is pass the whole data set in each epoch
    #       - data                  - Numpy array of type model.model_type. Information to choose the minibatches. 
    #                                 Required for SPRING and iSPALM. 
    #                                 To run PALM/iPALM on functions, which are not data based, use data=None.
    #                                 For SPRING and iSPALM a value not equal to None is required.
    #                                 Default value: None
    #       - test_data             - Numpy array of type model.model_type. Data points to evaluate the objective   
    #                                 function in the test step after each epoch. 
    #                                 For test_data=None, the function uses data as test_data.
    #                                 Default value: None
    #       - batch_size            - int. If data is None: No effect. Otherwise: batch_size for data driven models.
    #                                 Default value: 1000
    #       - method                - String value, which declares the optimization method. Valid choices are: 'PALM', 
    #                                 'iPALM', 'SPRING-SARAH' and 'iSPALM-SARAH'. Raises an error for other inputs.
    #                                 Default value: 'iSPALM-SARAH'
    #       - inertial_step_size    - float variable. For method=='PALM' or method=='SPRING-SARAH': No effect.      
    #                                 Otherwise: the inertial parameters in iPALM/iSPALM are chosen by 
    #                                 inertial_step_size*(k-1)/(k+2), where k is the current step number.
    #                                 For inertial_step_size=None the method choses 1 for PALM and iPALM, 0.5 for 
    #                                 SPRING and 0.4 for iSPALM.
    #                                 Default value: None
    #       - step_size             - float variable. The step size parameters tau are choosen by step_size*L where L 
    #                                 is the estimated partial Lipschitz constant of H.
    #       - sarah_seq             - This input should be either None or a sequence of uniformly on [0,1] distributed
    #                                 random float32-variables. The entries of sarah_seq determine if the full
    #                                 gradient in the SARAH estimator is evaluated or not.
    #                                 For sarah_seq=None such a sequence is created inside this function.
    #                                 Default value: None
    #       - sarah_p               - float in (1,\infty). Parameter p for the sarah estimator. If sarah_p=None the 
    #                                 method uses p=20
    #                                 Default value: None
    #       - precompile            - Boolean. If precompile=True, then the functions are compiled before the time
    #                                 measurement starts. Otherwise the functions are compiled at the first call.
    #                                 precompiele=True makes only sense if you are interested in the runtime of the
    #                                 algorithms without the compile time of the functions.
    #                                 Default value: False
    #       - test_batch_size       - int. test_batch_size is the batch size used in the test step and in the steps
    #                                 where the full gradient is evaluated. This does not effect the algorithm itself.
    #                                 But it may effect the runtime. For test_batch_size=None it is set to batch_size.
    #                                 If test_batch_size<batch_size and method=SPRING-SARAH or method=iSPALM-SARAH,
    #                                 then also in the steps, where not the full gradient is evaluated only batches
    #                                 of size test_batch_size are passed through the function H.
    #                                 Default value: None
    #       - ensure_full           - Boolean or int. For method=='SPRING-SARAH' or method=='iSPALM-SARAH': If
    #                                 ensure_full is True, we evaluate in the first step of each epoch the full
    #                                 gradient. We observed numerically, that this sometimes increases stability and
    #                                 convergence speed of SPRING and iSPALM. For PALM and iPALM: no effect.
    #                                 If a integer value p is provided, every p-th step is forced to be a full step
    #                                 Default value: False
    #       - estimate_lipschitz    - Boolean. If estimate_lipschitz==True, the Lipschitz constants are estimated based
    #                                 on the first minibatch in all steps, where the full gradient is evaluated.
    #                                 Default: True
    #       - backup_dir            - String or None. If a String is provided, the variables X[i] are saved after
    #                                 every epoch. The weights are not saved if backup_dir is None.
    #                                 Default: 'backup'
    #       - mute                  - Boolean. For mute=True the evaluation of the objective function and all prints
    #                                 will be suppressed.
    #                                 Default: False
    #
    # Outputs:
    #       - my_times              - list of floats. Contains the evaluation times of the training steps for each 
    #                                 epochs.
    #       - test_vals             - list of floats. Contains the objective function computed in the test steps for
    #                                 each epoch. Note returned if mute=True.

    optimizer=PALM_Optimizer(model,steps_per_epoch=steps_per_epoch,data=data,test_data=test_data,batch_size=batch_size,method=method,inertial_step_size=inertial_step_size,step_size=step_size,sarah_seq=sarah_seq,sarah_p=sarah_p,test_batch_size=test_batch_size,ensure_full=ensure_full,estimate_lipschitz=estimate_lipschitz,backup_dir=backup_dir,mute=mute)
    if precompile:   
        optimizer.precompile()
    old_val=np.inf
    for epoch in range(EPOCHS):
        val=optimizer.exec_epoch()
        eps=old_val-val
        old_val=val
        if eps<stop_crit:
            break
    if not mute:
        if optimizer.batch_version and optimizer.test_d:
            return optimizer.my_times,optimizer.test_vals,optimizer.train_vals
        return optimizer.my_times,optimizer.test_vals
    return optimizer.my_times


