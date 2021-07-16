#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os,sys,inspect
import os
import joblib
import tensorflow as tf
import numpy as np
import h5py
import scipy.sparse.linalg as la
import scipy.sparse as sp
import scipy
import time

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path_dataset = 'synthetic_netflix.mat'


# In[ ]:


# auxiliary functions:

# import matlab files in python
def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


# In[ ]:


#loading of the required matrices
M = load_matlab_file(path_dataset, 'M')
O = load_matlab_file(path_dataset, 'O')
Otraining = load_matlab_file(path_dataset, 'Otraining')
Otest = load_matlab_file(path_dataset, 'Otest')
Wrow = load_matlab_file(path_dataset, 'Wrow') #sparse
Wcol = load_matlab_file(path_dataset, 'Wcol') #sparse


# In[ ]:


np.random.seed(0)

pos_tr_samples = np.where(Otraining)

num_tr_samples = len(pos_tr_samples[0])
list_idx = list(range(num_tr_samples))
np.random.shuffle(list_idx)
idx_data = list_idx[:num_tr_samples//2]
idx_train = list_idx[num_tr_samples//2:]

pos_data_samples = (pos_tr_samples[0][idx_data], pos_tr_samples[1][idx_data])
pos_tr_samples = (pos_tr_samples[0][idx_train], pos_tr_samples[1][idx_train])

Odata = np.zeros(M.shape)
Otraining = np.zeros(M.shape)

for k in range(len(pos_data_samples[0])):
    Odata[pos_data_samples[0][k], pos_data_samples[1][k]] = 1
    
for k in range(len(pos_tr_samples[0])):
    Otraining[pos_tr_samples[0][k], pos_tr_samples[1][k]] = 1
    
print ('Num data samples: %d' % (np.sum(Odata),))
print ('Num train samples: %d' % (np.sum(Otraining),))
print ('Num train+data samples: %d' % (np.sum(Odata+Otraining),))


# In[ ]:


#computation of the normalized laplacians
Lrow = sp.csgraph.laplacian(Wrow, normed=True)
Lcol = sp.csgraph.laplacian(Wcol, normed=True)


# In[ ]:


class Train_test_matrix_completion:
    
    """
    The neural network model.
    """
    
    def frobenius_norm(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(input_tensor=square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
        return frobenius_norm
    
    
    def bid_conv(self, W, b):
        X = tf.reshape(self.X, [tf.shape(input=self.M)[0], tf.shape(input=self.M)[1]])
        
        feat = []
        #collect features
        for k_r in range(self.ord_row):
            for k_c in range(self.ord_col):
                row_lap = self.list_row_cheb_pol[k_r] 
                col_lap = self.list_col_cheb_pol[k_c]
                                                     
                #dense implementation
                c_feat = tf.matmul(row_lap, X, a_is_sparse=False)
                c_feat = tf.matmul(c_feat, col_lap, b_is_sparse=False)
                feat.append(c_feat)
                
        all_feat = tf.stack(feat, 2)
        all_feat = tf.reshape(all_feat, [-1, self.ord_row*self.ord_col])
        conv_feat = tf.matmul(all_feat, W) + b
        conv_feat = tf.nn.relu(conv_feat)
        conv_feat = tf.reshape(conv_feat, [tf.shape(input=self.M)[0], tf.shape(input=self.M)[1], self.n_conv_feat])
        return conv_feat
                
    def compute_cheb_polynomials(self, L, ord_cheb, list_cheb):
        for k in range(ord_cheb):
            if (k==0):
                list_cheb.append(tf.cast(tf.linalg.tensor_diag(tf.ones([tf.shape(input=L)[0],])), 'float32'))
            elif (k==1):
                list_cheb.append(tf.cast(L, 'float32'))
            else:
                list_cheb.append(2*tf.matmul(L, list_cheb[k-1])  - list_cheb[k-2])
        
    
    def __init__(self, M, Lr, Lc, Odata, Otraining, Otest, order_chebyshev_col = 5, order_chebyshev_row = 5,
                 num_iterations = 10, gamma=1.0, learning_rate=1e-4, idx_gpu = '/gpu:2'):
        
        #order of the spectral filters
        self.ord_col = order_chebyshev_col 
        self.ord_row = order_chebyshev_row
        self.num_iterations = num_iterations
        self.n_conv_feat = 32
        
        with tf.Graph().as_default() as g:
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                self.graph = g
                tf.compat.v1.set_random_seed(0)
                with tf.device(idx_gpu):
                    
                        #loading of the laplacians
                        self.Lr = tf.constant(Lr.astype('float32'))
                        self.Lc = tf.constant(Lc.astype('float32'))
                        
                        self.norm_Lr = self.Lr - tf.linalg.tensor_diag(tf.ones([Lr.shape[0], ]))
                        self.norm_Lc = self.Lc - tf.linalg.tensor_diag(tf.ones([Lc.shape[0], ]))
                        #compute all chebyshev polynomials a priori
                        self.list_row_cheb_pol = list()
                        self.compute_cheb_polynomials(self.norm_Lr, self.ord_row, self.list_row_cheb_pol)
                        self.list_col_cheb_pol = list()
                        self.compute_cheb_polynomials(self.norm_Lc, self.ord_col, self.list_col_cheb_pol)
                        
                        #definition of constant matrices
                        self.M = tf.constant(M, dtype=tf.float32)
                        self.Odata = tf.constant(Odata, dtype=tf.float32)
                        self.Otraining = tf.constant(Otraining, dtype=tf.float32) #training mask
                        self.Otest = tf.constant(Otest, dtype=tf.float32) #test mask
                         
                        #definition of the NN variables
                        self.W_conv = tf.compat.v1.get_variable("W_conv", shape=[self.ord_col*self.ord_row, self.n_conv_feat], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.b_conv = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        
                        #recurrent N parameters
                        self.W_f = tf.compat.v1.get_variable("W_f", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.W_i = tf.compat.v1.get_variable("W_i", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.W_o = tf.compat.v1.get_variable("W_o", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.W_c = tf.compat.v1.get_variable("W_c", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.U_f = tf.compat.v1.get_variable("U_f", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.U_i = tf.compat.v1.get_variable("U_i", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.U_o = tf.compat.v1.get_variable("U_o", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.U_c = tf.compat.v1.get_variable("U_c", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
                        self.b_f = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_i = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_o = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_c = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        
                        #output parameters
                        self.W_out = tf.compat.v1.get_variable("W_out", shape=[self.n_conv_feat,1], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")) 
                        self.b_out = tf.Variable(tf.zeros([1,1]))
                        
                        #########definition of the NN
                        self.X = tf.multiply(self.M, self.Odata) #we may initialize it at random here
                        self.list_X = list()
                        self.list_X.append(tf.identity(self.X))
                        self.X = tf.reshape(self.X, [-1,])
                        
                        #RNN
                        self.h = tf.zeros([M.shape[0]*M.shape[1], self.n_conv_feat])
                        self.c = tf.zeros([M.shape[0]*M.shape[1], self.n_conv_feat])
                        
                        for k in range(self.num_iterations):
                            #bidimensional convolution
                            self.x_conv = self.bid_conv(self.W_conv, self.b_conv) #N, N, n_conv_feat
                            self.x_conv = tf.reshape(self.x_conv, [-1, self.n_conv_feat])
                            
                            self.f = tf.sigmoid(tf.matmul(self.x_conv, self.W_f) + tf.matmul(self.h, self.U_f) + self.b_f)
                            self.i = tf.sigmoid(tf.matmul(self.x_conv, self.W_i) + tf.matmul(self.h, self.U_i) + self.b_i)
                            self.o = tf.sigmoid(tf.matmul(self.x_conv, self.W_o) + tf.matmul(self.h, self.U_o) + self.b_o)
                            
                            self.update_c = tf.sigmoid(tf.matmul(self.x_conv, self.W_c) + tf.matmul(self.h, self.U_c) + self.b_c)
                            self.c = tf.multiply(self.f, self.c) + tf.multiply(self.i, self.update_c)
                            self.h = tf.multiply(self.o, tf.sigmoid(self.c))
                            
                            #compute update of matrix X
                            self.delta_x = tf.tanh(tf.matmul(self.c, self.W_out) + self.b_out)
                            self.X += tf.squeeze(self.delta_x)
                            self.list_X.append(tf.identity(tf.reshape(self.X, [tf.shape(input=self.M)[0], tf.shape(input=self.M)[1]])))
                            
                            
                        self.X = tf.reshape(self.X, [tf.shape(input=self.M)[0], tf.shape(input=self.M)[1]])
                        #########loss definition
                        
                        #computation of the accuracy term
                        self.norm_X = 1+4*(self.X-tf.reduce_min(input_tensor=self.X))/(tf.reduce_max(input_tensor=self.X-tf.reduce_min(input_tensor=self.X)))
                        frob_tensor = tf.multiply(self.Otraining + self.Odata, self.norm_X - M)
                        self.loss_frob = tf.square(self.frobenius_norm(frob_tensor))/np.sum(Otraining+Odata)
                        
                        #computation of the regularization terms
                        trace_col_tensor = tf.matmul(tf.matmul(self.X, self.Lc), self.X, transpose_b=True)
                        self.loss_trace_col = tf.linalg.trace(trace_col_tensor)
                        trace_row_tensor = tf.matmul(tf.matmul(self.X, self.Lr, transpose_a=True), self.X)
                        self.loss_trace_row = tf.linalg.trace(trace_row_tensor)
                        
                        #training loss definition
                        self.loss = self.loss_frob + (gamma/2)*(self.loss_trace_col + self.loss_trace_row)
                        
                        #test loss definition
                        self.predictions = tf.multiply(self.Otest, self.norm_X - self.M)
                        self.predictions_error = self.frobenius_norm(self.predictions)

                        #definition of the solver
                        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
                        
                        self.var_grad = tf.gradients(ys=self.loss, xs=tf.compat.v1.trainable_variables())
                        self.norm_grad = self.frobenius_norm(tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0))

                        # Create a session for running Ops on the Graph.
                        config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
                        config.gpu_options.allow_growth = True
                        self.session = tf.compat.v1.Session(config=config)

                        # Run the Op to initialize the variables.
                        init = tf.compat.v1.initialize_all_variables()
                        self.session.run(init)


# In[ ]:


ord_col = 5
ord_row = 5

learning_obj = Train_test_matrix_completion(M, Lrow.toarray(), Lcol.toarray(), Odata, Otraining, Otest, 
                                            order_chebyshev_col = ord_col, order_chebyshev_row = ord_row, 
                                            gamma=1e-8, learning_rate=1e-3)

num_iter_test = 10
num_total_iter_training = 25000

num_iter = 0

list_training_loss = list()
list_training_norm_grad = list()
list_test_pred_error = list()
list_predictions = list()
list_X = list()

list_training_times = list()
list_test_times = list()
list_grad_X = list()

list_X_evolutions = list()


# In[ ]:


num_iter = 0
for k in range(num_iter, num_total_iter_training):

    tic = time.time()
    _, current_training_loss, norm_grad, X_grad = learning_obj.session.run([learning_obj.optimizer, learning_obj.loss, 
                                                                            learning_obj.norm_grad, learning_obj.var_grad]) 
    training_time = time.time() - tic

    list_training_loss.append(current_training_loss)
    list_training_norm_grad.append(norm_grad)
    list_training_times.append(training_time)
        
    if (np.mod(num_iter, num_iter_test)==0):
        msg = "[TRN] iter = %03i, cost = %3.2e, |grad| = %.2e (%3.2es)"                                     % (num_iter, list_training_loss[-1], list_training_norm_grad[-1], training_time)
        print (msg)
            
        #Test Code
        tic = time.time()
        pred_error, preds, X = learning_obj.session.run([learning_obj.predictions_error, learning_obj.predictions,
                                                                             learning_obj.norm_X]) 
        c_X_evolutions = learning_obj.session.run(learning_obj.list_X)
        list_X_evolutions.append(c_X_evolutions)

        test_time = time.time() - tic

        list_test_pred_error.append(pred_error)
        list_X.append(X)
        list_test_times.append(test_time)
        msg =  "[TST] iter = %03i, cost = %3.2e (%3.2es)" % (num_iter, list_test_pred_error[-1], test_time)
        print (msg)
            
    num_iter += 1


# In[ ]:


fig, ax1 = plt.subplots(figsize=(20,10))

ax2 = ax1.twinx()
ax1.plot(np.arange(len(list_training_loss)), list_training_loss, 'g-')
ax2.plot(np.arange(len(list_test_pred_error))*num_iter_test, list_test_pred_error, 'b-')

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Training loss', color='g')
ax2.set_ylabel('Test loss', color='b')

best_iter = (np.where(np.asarray(list_training_loss)==np.min(list_training_loss))[0][0]//num_iter_test)*num_iter_test
best_pred_error = list_test_pred_error[best_iter//num_iter_test]
print ('Best predictions at iter: %d (error: %f)' % (best_iter, best_pred_error))
RMSE = np.sqrt(np.square(best_pred_error)/np.sum(Otest))
print ('RMSE: %f' % RMSE)


# In[ ]:


#best X generated
plt.figure(figsize=(20,10))
plt.imshow(list_X[best_iter//num_iter_test])
plt.colorbar()


# In[ ]:




