##########################################################
# @author: Najeeb Khan                                   #
# @github: najeeb97khan                                  #
# @email: najeeb.khan96@gmail.com                        #
# @version: 1.0.0, 20/01/2018                            #
# @license: MIT                                          #
##########################################################

# Importing Libraries
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os


class Neural_Network(object):
    
    def __init__(self, params):
        
        '''
        Initialising the parameters
        '''
        
        self.params = params
        self.model = {'keep_prob':None}
        
    def create_placeholders(self):
        
        '''
        Creating placeholders for input data and labels
        '''

        self.model['X'] = tf.placeholder(dtype=tf.float32, shape=[None, self.params['num_features']], name='X')
        self.model['y'] = tf.placeholder(dtype=tf.float32, shape=[None, self.params['num_classes']], name='y')
        self.model['keep_prob'] = tf.placeholder(tf.float32)
        return self.model['X'], self.model['y']
    
    def _fully_connected(self, prev_layer, n_output, name):
        
        '''
        Fully connected layer with ReLU activations
        '''

        with tf.variable_scope(name) as scope:
            
            shape = prev_layer.get_shape().as_list()
            w = tf.get_variable(name='weights', shape=[shape[1], n_output], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='bias', shape=[n_output], dtype=tf.float32,
                               initializer=tf.random_normal_initializer())
            out = tf.nn.relu(tf.matmul(prev_layer, w) + b)
            return out
        
    def _create_dropout(self, prev_layer, name):
        
        '''
        Dropout layer for minimizing variance of the network.
        Nodes are dropped with 50 percent probability.
        '''

        with tf.variable_scope(name) as scope:
                      
            dropout = tf.nn.dropout(prev_layer, self.model['keep_prob'])
            return dropout
        
    def _create_softmax(self, prev_layer, n_output=None, name='softmax'):
        
        '''
        Softmax layer for normalized probabilities of output
        '''

        with tf.variable_scope(name) as scope:
            
            if not n_output:
                n_output = self.params['num_classes']
            
            w = tf.get_variable(name='weights', shape=[prev_layer.shape[1], n_output],
                                dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='bias', shape=[n_output], dtype=tf.float32,
                               initializer=tf.random_normal_initializer())
            softmax = tf.nn.softmax(tf.matmul(prev_layer, w) + b)
            return softmax
    
    def _create_loss(self, predicted_labels):
        
        '''
        Categorical Cross Entropy Loss function
        '''
        
        with tf.variable_scope('loss') as scope:
            
            self.model['loss'] = tf.reduce_mean(-tf.reduce_sum(self.model['y']*tf.log(predicted_labels + 1e-8),
                                                               reduction_indices=[1]))
            
    
    def _create_summaries(self):
        
        '''
        Loss summary for plotting
        '''
        
        tf.summary.scalar("loss curve", self.model['loss'])
        tf.summary.histogram("loss histogram", self.model['loss'])
        self.model['summary_op'] = tf.summary.merge_all()
        
    
    def _accuracy(self, predicted_labels):
        
        '''
        Predicting accuracy of the model by comparing
        predicted labels with true labels for a mini
        batch
        '''
        
        with tf.variable_scope('accuracy') as scope:
            
            correct_preds = tf.equal(tf.argmax(self.model['y'], 1), tf.argmax(predicted_labels, 1))
            self.model['accuracy'] = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
            
    
    def _create_optimizer(self):
        
        '''
        Creating AdaGrad Optimizer to minimize the CE loss
        '''
        
        self.model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.model['optimizer'] = tf.train.AdamOptimizer(self.params['learning_rate']).minimize(self.model['loss'],
                                                                                             global_step = self.model['global_step'])
    
    def _train(self, X, y, X_val, y_val):
        
        '''
        Training the model using Stochastic Gradient Descent
        '''
        
        num_train = X.shape[0]
        init = tf.global_variables_initializer()
        n_batches = num_train // self.params['batch_size']
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            sess.run(init)
            writer = tf.summary.FileWriter('graphs/nn', sess.graph)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/nn/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                
                print('Restoring checkpoint')
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            point_loss = []
            
            for i in range(1, self.params['num_epochs']+1):
                epoch_loss, epoch_accuracy = 0, 0
                
                for j in range(n_batches):
                    
                    cur_batch = j*self.params['batch_size']
                    next_batch = (j+1)*self.params['batch_size']
                    
                    x_batch, y_batch = X[cur_batch:next_batch, :], y[cur_batch:next_batch, :]
                    
                    feeder = {self.model['X']:X, self.model['y']:y,
                              self.model['keep_prob']:self.params['keep_prob']}
                    
                    _, l, acc, summary = sess.run([self.model['optimizer'], self.model['loss'],
                                                 self.model['accuracy'], self.model['summary_op']],
                                                 feed_dict=feeder)
                    point_loss.append(l)
                    epoch_loss += l
                    epoch_accuracy += acc
                    
                writer.add_summary(summary, global_step = i)
                
                print('Epoch: {}\n Loss: {}\tAccuracy: {}'.format(i, epoch_loss, epoch_accuracy/n_batches))
                
                if i % self.params['train_step'] == 0:
                    
                    print("Saving checkpoint...")
                    
                    if not os.path.exists('checkpoints'):
                        os.makedirs('checkpoints')
                        os.makedirs('checkpoints/nn')
                    saver.save(sess, 'checkpoints/nn/nn', i)
                    cross_val_loss, cross_val_score = sess.run([self.model['loss'], self.model['accuracy']],
                                                               feed_dict={self.model['X']:X_val,
                                                                          self.model['y']:y_val,
                                                                          self.model['keep_prob']:self.params['keep_prob']})
                    
                    print("Cross Val Loss: {}\t Cross Val Score: {}".format(cross_val_loss, cross_val_score))
                
                print("*"*100)
                
            writer.close()
        return point_loss
