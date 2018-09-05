# Ver 1.1 Updates:
# 1. get parameters from npy file
# 2. pruning the parameters in different layers with defferent thresholds
# Ver 1.2 Updates:
# 1. apply mask to matmul, both in inference and backprop
# Author Cooper Qiu

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from tensorflow.python.ops import variable_scope as vs
import shutil
import time
from datetime import datetime
import scipy as sci

class Mnist(object):
    def __init__(self):
        self.mnist_data = input_data.read_data_sets('./mnist_data', one_hot=True)
        self.train_lr = 0.005
        self.retrain_lr = 0.005
        self.x_data = tf.placeholder(tf.float32, [None, 784], name='x_data')
        self.x_data_reshape = tf.reshape(self.x_data, [-1, 28, 28, 1])
        self.y_data = tf.placeholder(tf.float32, [None, 10], name='y_data')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.store_path = 'model_saved/'
        self.pruned_retrained_path = 'pruned_retrained/'
        self.graph_path = 'graph/'
        self.flag = 1
        
        self.global_step = 0
        #self.phase = phase
        self.neuron_list = [784, 300, 100, 10]
        
        self.logits, self.y_hat = self.Build_networks(self.x_data_reshape)
        self.loss = self.loss_func(self.logits, self.y_data)
        # in order to achieve the gradients mask, it's better to apply gradient descents without optimizer.
        self.trainable_variable = tf.trainable_variables()
        

        # we use the tf.gradients to construct the optimizer
        #self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
        self.correct_prediction = tf.equal(tf.argmax(self.y_hat, -1), tf.argmax(self.y_data, -1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        # It's bad to applay gradients_list_op here.
        #self.masks = None
        #self.gradients_list_op, self.gradients_masks_list = self.gradients_cal(self.masks)
        
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10) 
        self.summary_op = self.summary_func()
        
    # def the Fully connected network
    def Fc_layer(self, X, input_size, output_size, Scope, acti='relu'):
        with tf.variable_scope(Scope, tf.AUTO_REUSE) as scope:
            W1 = tf.get_variable('w1', shape=[input_size, output_size], initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))
            b1 = tf.get_variable('b1', shape=[output_size], initializer=tf.constant_initializer(0.1))
        if acti == 'relu':
            net = tf.nn.relu(tf.matmul(X, W1) + b1)
        elif acti == 'tanh':
            net = tf.nn.tanh(tf.matmul(X, W1) + b1)
        elif acti == 'sigmoid':
            net = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
        elif acti == 'softmax':
            net = tf.nn.softmax(tf.matmul(X, W1) + b1)
        else:
            net = tf.matmul(X, W1) + b1
        return net
    
    # Conv layer
    def Conv_layer(self, X, channels, ker_size, Scope, stride=[1,1,1,1], padding='SAME'):
        with tf.variable_scope(Scope, tf.AUTO_REUSE) as scope:
            ker_shape = [ker_size, ker_size, X.shape[-1], channels]
            W = tf.get_variable('W', shape=ker_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('b', shape=[channels], initializer=tf.constant_initializer(0.1))
        Z = tf.nn.conv2d(X, W, strides=stride, padding=padding)
        A = tf.nn.relu(Z + b)
        return A
    
    # Pooling layer
    def Poolinglayer(self, X, ksize, stride, padding='SAME', pool_type='max'):
        if pool_type == 'max':
            return tf.nn.max_pool(X, ksize, stride, padding=padding)
        elif pool_type == 'average':
            return tf.nn.avg_pool(X, ksize, stride, padding=padding)
    
    # build the network
    def Build_networks(self, X):
        #net1 = 1.0*X/255
        net = self.Conv_layer(X, 32, 5, Scope='Conv1')
        net = self.Poolinglayer(net, [1,2,2,1], [1,2,2,1])
        
        # layer 2
        net = self.Conv_layer(net, 64, 5, Scope='Conv2')
        net = self.Poolinglayer(net, [1,2,2,1], [1,2,2,1])
        
        # layer 3
        net = tf.reshape(net, [-1, 7*7*64])
        net = self.Fc_layer(net, 7*7*64, 1024, 'Fc1')
        net = tf.nn.dropout(net, self.keep_prob)
        
        #layer 4
        logits = self.Fc_layer(net, 1024, 10, 'Fc2', acti='Linear')
    
        y_hat = tf.nn.softmax(logits)
        return logits, y_hat
 
    # loss func
    def loss_func(self, logits, labels):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy'))
        #trainable_vars = tf.trainable_variables()
        #l2_loss = tf.reduce_sum([tf.nn.l2_loss(i)/(i.) for i in trainable_vars])
        #total_loss = cross_entropy + l2_loss
        
        return cross_entropy
        #return total_loss
        
    # def masked gradients
    def gradients_cal(self, lr = 0.001, mask=None):
        var_list = self.trainable_variable
        vs.get_variable_scope().reuse_variables()
        gradients_list = tf.gradients(xs=var_list, ys=self.loss)
        
        gradients_op_list = []
        gradients_mask_list = []
        # lr * gradients
        gradients = [lr*g for g in gradients_list]

        for i in range(len(var_list)):
            # collect the gradients op
            # apply mask
            if mask != None:
                #print()
                gradients_mask = gradients[i] * mask[i]
                gradients_mask_list.append(gradients_mask)
                new_var = tf.subtract(var_list[i], gradients_mask)
            else:
                #print('ah ha')
                new_var = tf.subtract(var_list[i], gradients[i])
            gradients_op = var_list[i].assign(new_var)
            gradients_op_list.append(gradients_op)
        
        return gradients_op_list, gradients_mask_list
    
    
    # def summary
    def summary_func(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('hist_loss', self.loss)
        sum_op = tf.summary.merge_all()
        return sum_op
    
    # train func
    def train(self):
        #if the model_saved dir is exist then remove it.
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)   
        gradients_list_op, _ = self.gradients_cal(lr=self.train_lr)
        
        time_stamp = '{0:%Y-%m-%dT%H-%M-%S/}'.format(datetime.now())
        graph_path = self.graph_path + time_stamp
        writer = tf.summary.FileWriter(graph_path, self.sess.graph)
        
        
        for epoch in range(5000):
            
            for i in range(10):
                stime = time.time()
                self.global_step = self.global_step + 1
                x_batch, y_batch = self.mnist_data.train.next_batch(128)
                _, loss, summary = self.sess.run([gradients_list_op, self.loss, self.summary_op], 
                                        feed_dict={self.x_data:x_batch, self.y_data:y_batch, self.keep_prob:0.5})
                #summary = self.sess.run(self.summary_op, 
                                        #feed_dict={self.x_data:x_batch, self.y_data:y_batch, self.keep_prob:0.5})
                
                writer.add_summary(summary, global_step=self.global_step)
                etime = time.time()
            if epoch % 100 == 0:
                # After 50 epoch lr decays
                self.train_lr = self.train_lr*0.9
                
                if os.path.exists(self.store_path):
                    self.saver.save(self.sess, self.store_path+'mnist.ckpt', global_step=self.global_step)
                else:
                    os.mkdir(self.store_path)
                    self.saver.save(self.sess, self.store_path+'mnist.ckpt', global_step=self.global_step)
                acc = self.sess.run(self.accuracy, feed_dict={self.x_data:self.mnist_data.test.images,
                                                    self.y_data:self.mnist_data.test.labels, self.keep_prob:1})
                print('After %d epochs, global_step %d with lr:%.7f, loss: %.7f, accuracy: %.7f, time: %.4f' % 
                      (epoch, self.global_step, self.train_lr, loss, acc, (etime-stime)))  
        #reset lr
        self.train_lr = 0.005        
    
    # get threshold according to different layers.
    def get_thresholds(self):
        ckpt = tf.train.get_checkpoint_state(self.store_path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        trainable_vars = self.sess.run(self.trainable_variable)
        thresholds = []
        var_stds = []
        var_means = []
        
        for i in range(len(trainable_vars)):
            var_std = np.std(trainable_vars[i])
            var_mean = np.average(trainable_vars[i])
            var_stds.append(var_std)
            var_means.append(var_mean)
            if self.trainable_variable[i].name.startswith('Conv'):
                coef = 1
            else:
                coef = 2
            #thresholds.append(var_mean+2*(i+coef)/len(trainable_vars)*var_std)
            thresholds = [0.06, 0.06, 0.08,0.08, 0.16, 0.18, 0.16, 0.15]
        #print(var_means)
        #print(var_stds)
        #print(thresholds)
        return thresholds
        
    
    def prunning(self, thresholds):
        stime = time.time()
        print('Start Pruning...')
            
        if self.flag == 1:
            if os.path.exists(self.pruned_retrained_path):
                # remove old parameters.
                shutil.rmtree(self.pruned_retrained_path)
            
            ckpt = tf.train.get_checkpoint_state(self.store_path)
            self.flag = self.flag - 1
        else:
            ckpt = tf.train.get_checkpoint_state(self.pruned_retrained_path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        # set variables reusable
        vs.get_variable_scope().reuse_variables()
        
        trainable_variables = tf.trainable_variables()
        #trainable_variables.
        list_masks = []
        list_weights = []
        pre_matrix = np.transpose(np.ones(trainable_variables[0].shape))
        
        for var in trainable_variables:
            
            weights_temp_np = self.sess.run(var)
            list_weights.append(weights_temp_np)
            
            # create mask for var using the threshold
            mask_temp = np.asarray(np.abs(weights_temp_np)>thresholds[trainable_variables.index(var)], np.float)
            
            # enhance the mask using the dead connection            
            if len(var.shape) == 2:
                now_matrix = mask_temp
                for i in range(pre_matrix.shape[1]):
                    judge = (pre_matrix[:,i] == np.zeros_like(pre_matrix[:,i]))
                    if np.average(np.asarray(judge, np.float)) == 1:
                        now_matrix[i,:] = np.zeros_like(now_matrix[i,:])
                for j in range(now_matrix.shape[0]):
                    judge2 = (now_matrix[j,:] == np.zeros_like(now_matrix[j,:]))
                    if np.average(np.asarray(judge2, np.float)) == 1:
                        pre_matrix[:, j] = np.zeros_like(pre_matrix[:, j])
                pre_matrix = now_matrix
                list_masks.append(now_matrix)
            else:
                list_masks.append(mask_temp)
            
            self.sess.run(tf.assign(var, np.multiply(weights_temp_np, mask_temp)))
        self.saver.save(self.sess, self.pruned_retrained_path+'mnist_pruned.ckpt', global_step=self.global_step)
        etime = time.time()
        print('Done Pruning! Cost %.4f seconds' % (etime-stime))
        #np.save('pruned'+self.global_step+'.txt', list_mask[4]*list_weights[4])
        return list_masks, list_weights
    
    #retrain
    def retrain(self, masks):
        print('Start retraining...')
        stime = time.time()
        #shutil.rmtree()
        ckpt = tf.train.get_checkpoint_state(self.pruned_retrained_path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)  
        gradients_list_op, gradients_masks_list = self.gradients_cal(lr=self.retrain_lr, mask=masks)
        self.masks = masks
        #comp_rate = np.average(masks)
        ele_count = 0
        one_count = 0
        for i in range(len(masks)):
            ele_count = ele_count + masks[i].size
            one_count = one_count + np.sum(masks[i])
        print('compress rate: %.4f: ' % (one_count*1.0 / ele_count))
        # Dropout should be scaled down
        keep_prob = np.average(np.asarray(masks[-4] > 0, np.float))*0.5
        
        
        time_stamp = '{0:%Y-%m-%dT%H-%M-%S/}'.format(datetime.now())
        graph_path = self.graph_path + time_stamp        
        writer = tf.summary.FileWriter(graph_path, self.sess.graph)
        for epoch in range(20000):
            
            for i in range(10):
                sstime = time.time()
                self.global_step = self.global_step + 1
                x_batch, y_batch = self.mnist_data.train.next_batch(64)
                _, summary = self.sess.run([gradients_list_op, self.summary_op],
                                           feed_dict={self.x_data:x_batch, self.y_data:y_batch, self.keep_prob:keep_prob})
                
                # summary = self.sess.run(self.summary_op, feed_dict={self.x_data:x_batch, self.y_data:y_batch, self.keep_prob:keep_prob})
                writer.add_summary(summary, global_step=self.global_step)
                # we should set the gradients to zero.
                #tf.gradients()
                eetime = time.time()
            if (epoch) % 100 == 0:
                acc, loss = self.sess.run([self.accuracy, self.loss], feed_dict={self.x_data:self.mnist_data.test.images,
                                                    self.y_data:self.mnist_data.test.labels, self.keep_prob:1})
                print('After %d epochs, global_step %d with lr:%.7f, loss: %.7f, accuracy: %.7f, time: %.4f' % 
                      (epoch,self.global_step, self.retrain_lr, loss, acc, (eetime-sstime)))            
                self.retrain_lr = self.retrain_lr*0.95
            
            #np.savetxt('glist.txt', glist[4])        
        self.saver.save(self.sess, self.pruned_retrained_path+'mnist_retrained.ckpt', global_step=self.global_step)
        etime = time.time()
        # reset lr
        self.retrain_lr = 0.005        
        print('Done retraining! Cost %.4f seconds' % (etime-stime))
        #np.save('retrained'+self.global_step+'.txt', )                      
    
    # dense
    
    def compare_parameters(self):
        
        ckpt = tf.train.get_checkpoint_state(self.pruned_retrained_path)
        
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        trainable_var1 = tf.trainable_variables()
        trainable_var1_array = self.sess.run(trainable_var1)
        # save it to .npy format
        #np.save('par.npy', trainable_var1_array[4])
        # append csr format values to a list
        var_csrs = []
        var_shapes = []# used to store every layers shape. 

        
        for ele in trainable_var1_array:
            # reshape the multi-dim matrix to 2 dims matrix and then apply csr
            var_shape = ele.shape
            var_shapes.append(var_shape)
            if len(var_shape) > 2:
                
                ele = np.reshape(ele, (var_shape[0]*var_shape[1], -1))
            trainable_var1_sparse = sci.sparse.csr_matrix(ele)
            # csr_elements
            csr_indptr = trainable_var1_sparse.indptr
            csr_indices = trainable_var1_sparse.indices            
            csr_data = trainable_var1_sparse.data
            #csr_data = trainable_var1_sparse.data.astype(np.float16)
            var_csrs.append([csr_indptr, csr_indices, csr_data])
        
        '''
        data type: 
        [0]: var_csrs, and in every ele, we have three elements:
            [0]: csr_indptr
            [1]: csr_indices
            [2]: csr_data
        [1]: var_shapes, every layers' variable shape  
        '''
        np.save('var_csr_shape.npy', [var_csrs, var_shapes])

        for i in trainable_var1_array:
            comp_rate = np.average(np.asarray(np.abs(i)>0, np.float))
            print('After %d global steps, the compress rate: %.4f' % (self.global_step, comp_rate))      
    
    # def inference function, restore parameters from the saved files
    def inference(self, file_path):
        layer_vars_shapes = np.load(file_path)
        layer_vars = layer_vars_shapes[0]
        layer_shapes = layer_vars_shapes[1]
        
        assign_op = []
        
        for i in range(len(layer_shapes)):
            # get the corresponding layers var matrix.
            indptr = layer_vars[i][0]
            indices = layer_vars[i][1]
            data = layer_vars[i][2]
            #data = layer_vars[i][2].astype(np.float32)
            # form the csr matrix, some of them have shape not equal 2, we use reshape or make up to form 2 dims
            if len(layer_shapes[i]) > 2:
                shape_dim2 = (layer_shapes[i][0]*layer_shapes[i][1], layer_shapes[i][2]*layer_shapes[i][3])
                #print(shape_dim2)
            elif len(layer_shapes[i]) < 2:
                # shape=(n,), make it become (1, n)
                shape_dim2 = (1, layer_shapes[i][0])
                #print('dim<2')
                #print(shape_dim2)
            else:
                shape_dim2 = layer_shapes[i]
            weights_dims2 = sci.sparse.csr_matrix((data, indices, indptr), shape=shape_dim2).toarray()
            #print(weights_dims2.shape)
            weights_dims2_orig = np.reshape(weights_dims2, layer_shapes[i])
            assign_op.append(tf.assign(self.trainable_variable[i], weights_dims2_orig))
        
        self.sess.run(assign_op)
        stime = time.time()
        acc, loss = self.sess.run([self.accuracy, self.loss], feed_dict={self.x_data:self.mnist_data.test.images,
                                                                                     self.y_data:self.mnist_data.test.labels, self.keep_prob:1})
        etime = time.time()
        print('Inference loss: %.7f, accuracy: %.7f, time: %.4f' % (loss, acc, (etime-stime)))                 
        
        
        
    # deep compression
    def Deep_comp(self, thresholds):

        #threshold = 1.5e-1
        #for i in range(3):
        #print('Phase %d' % (i+1))
        mask, _ = self.prunning(thresholds)
        #print(mask[0])
        
        self.retrain(mask)
        #threshold = 1.2*threshold
        #print('Phase %d complete' % (i+1))
        print('compress rate:')
        self.compare_parameters()
    
def main():
    mnist = Mnist()
    #print('Start training...')
    #mnist.train()
    #print('Done training.')
    
    thresholds = mnist.get_thresholds()
        
    mnist.Deep_comp(thresholds)

    #mnist.inference('var_csr_shape.npy')
    #threshold = threshold*1.1
    
if __name__ =='__main__':
    main()
    
    

