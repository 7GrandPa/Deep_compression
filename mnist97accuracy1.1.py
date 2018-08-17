import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from tensorflow.python.ops import variable_scope as vs
import shutil
import time
from datetime import datetime

class Mnist(object):
    def __init__(self, phase='train'):
        self.mnist_data = input_data.read_data_sets('./mnist_data', one_hot=True)
        self.lr = 0.001
        self.x_data = tf.placeholder(tf.float32, [None, 784], name='x_data')
        self.y_data = tf.placeholder(tf.float32, [None, 10], name='y_data')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.store_path = 'model_saved/'
        self.pruned_retrained_path = 'pruned_retrained/'
        self.graph_path = 'graph/'
        self.flag = 1
        
        self.global_step = 0
        self.phase = phase
        self.neuron_list = [784, 300, 100, 10]
        
        self.logits, self.y_hat = self.Build_networks(self.neuron_list, self.x_data)
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
        
    # def the network
    def Build_layer(self, X, input_size, output_size, Scope, acti='relu'):
        with tf.variable_scope(Scope, tf.AUTO_REUSE) as scope:
            W1 = tf.get_variable('w1', shape=[input_size, output_size], initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))
            b1 = tf.get_variable('b1', initializer=tf.constant(0.1))
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
    
    # loss func
    def loss_func(self, logits, labels):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy'))
        #trainable_vars = tf.trainable_variables()
        #l2_loss = tf.reduce_sum([tf.nn.l2_loss(i)/(i.) for i in trainable_vars])
        #total_loss = cross_entropy + l2_loss
        
        return cross_entropy
        #return total_loss
 
    # build the network
    def Build_networks(self, neuronlist, X):
        #net1 = 1.0*X/255
        net1 = X
        for num in range(len(neuronlist)-2):
            sco = 'layer'+str(num+1)
            net1 = self.Build_layer(net1, neuronlist[num], neuronlist[num+1], sco)
            #net1 = tf.nn.dropout(net1, keep_prob=0.6)
        logits = self.Build_layer(net1, neuronlist[-2], neuronlist[-1], 'net'+str(len(neuronlist)-1), acti='linear')
        y_hat = tf.nn.softmax(logits)
        return logits, y_hat
    
    # def masked gradients
    def gradients_cal(self, mask=None):
        var_list = self.trainable_variable
        vs.get_variable_scope().reuse_variables()
        gradients_list = tf.gradients(xs=var_list, ys=self.loss)
        
        gradients_op_list = []
        gradients_mask_list = []
        # lr * gradients
        gradients = [self.lr*g for g in gradients_list]

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
        gradients_list_op, _ = self.gradients_cal()
        
        time_stamp = '{0:%Y-%m-%dT%H-%M-%S/}'.format(datetime.now())
        graph_path = self.graph_path + time_stamp
        writer = tf.summary.FileWriter(graph_path, self.sess.graph)
        for epoch in range(2500):
            # After 50 epoch lr decays
            self.lr = 0.001*0.9**(epoch/50)
            
            for i in range(100): 
                x_batch, y_batch = self.mnist_data.train.next_batch(100)
                _, loss = self.sess.run([gradients_list_op, self.loss], feed_dict={self.x_data:x_batch, self.y_data:y_batch})
                summary = self.sess.run(self.summary_op, feed_dict={self.x_data:x_batch, self.y_data:y_batch})
                
                self.global_step = self.global_step + 1
                writer.add_summary(summary, global_step=self.global_step)

            if epoch % 50 == 0:
                if os.path.exists(self.store_path):
                    self.saver.save(self.sess, self.store_path+'mnist.ckpt', global_step=self.global_step)
                else:
                    os.mkdir(self.store_path)
                    self.saver.save(self.sess, self.store_path+'mnist.ckpt', global_step=self.global_step)
                acc = self.sess.run(self.accuracy, feed_dict={self.x_data:self.mnist_data.test.images,
                                                    self.y_data:self.mnist_data.test.labels})
                print('After %d epochs, global_step %d with lr:%.7f, loss: %.4f, accuracy: %.4f' % 
                      (epoch, self.global_step, self.lr, loss, acc))            
    
    def prunning(self, threshold=1e-1):
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
            mask_temp = np.asarray(np.abs(weights_temp_np)>threshold, np.float)
            
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
        gradients_list_op, gradients_masks_list = self.gradients_cal(mask=masks)
        self.masks = masks
        
        time_stamp = '{0:%Y-%m-%dT%H-%M-%S/}'.format(datetime.now())
        graph_path = self.graph_path + time_stamp        
        writer = tf.summary.FileWriter(graph_path, self.sess.graph)
        for epoch in range(1200):
            if (epoch+1) % 100 == 0:
                acc, _, loss = self.sess.run([self.accuracy, self.summary_op, self.loss], feed_dict={self.x_data:self.mnist_data.test.images,
                                                    self.y_data:self.mnist_data.test.labels})
                print('After %d epochs, globale_step %d with lr:%.7f, loss: %.4f, accuracy: %.4f' % 
                      (epoch,self.global_step, self.lr, loss, acc))            
            self.lr = 0.001*0.95**(epoch/100)
            for i in range(100):
                x_batch, y_batch = self.mnist_data.train.next_batch(100)
                _, glist = self.sess.run([gradients_list_op, gradients_masks_list],
                                           feed_dict={self.x_data:x_batch, self.y_data:y_batch})
                
                self.global_step = self.global_step + 1
                summary = self.sess.run(self.summary_op, feed_dict={self.x_data:x_batch, self.y_data:y_batch})
                writer.add_summary(summary, global_step=self.global_step)
                # we should set the gradients to zero.
                #tf.gradients()
            np.savetxt('glist.txt', glist[4])        
        self.saver.save(self.sess, self.pruned_retrained_path+'mnist_retrained.ckpt', global_step=self.global_step)
        etime = time.time()
        print('Done retraining! Cost %.4f seconds' % (etime-stime))
        #np.save('retrained'+self.global_step+'.txt', )                      
    
    def compare_parameters(self):
        
        ckpt = tf.train.get_checkpoint_state(self.pruned_retrained_path)
        
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)   
        trainable_var1 = tf.trainable_variables()
        trainable_var1_array = self.sess.run(trainable_var1)
        for i in trainable_var1_array:
            comp_rate = np.average(np.asarray(np.abs(i)>0, np.float))
            print('After %d global steps, the compress rate: %.4f' % (self.global_step, comp_rate))      
        
    # deep compression
    def Deep_comp(self):

        threshold = 1.2e-1
        for i in range(3):
            print('Phase %d' % (i+1))
            mask, _ = self.prunning(threshold)
            #print(mask[0])
            
            self.retrain(mask)
            #threshold = 1.2*threshold
            print('Phase %d complete' % (i+1))
        print('compress rate:')
        self.compare_parameters()
        
def main():
    mnist = Mnist()
    print('Start training...')
    mnist.train()
    print('Done training.')
    mnist.Deep_comp()
    
if __name__ =='__main__':
    main()
    

