##########################################################
# @author: Najeeb Khan                                   #
# @github: najeeb97khan                                  #
# @email: najeeb.khan96@gmail.com                        #
# @version: 1.0.0, 20/01/2018                            #
# @license: MIT                                          #
##########################################################

# Importing Libraries
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

class ResNet(object):
    
    '''
    ResNet Model Class
    '''
    def __init__(self, params):
        
        '''
        Initialising the parameters
        '''
        self.params = params
        self.model = {}
        
    def _create_placeholders(self):
        
        '''
        Placeholders for input data and labels
        '''
        self.model['X'] = tf.placeholder(dtype=tf.float32, shape=[None, self.params['image_width'], 
                                                                 self.params['image_height'],
                                                                 self.params['color_channels']], name='X')
        self.model['y'] = tf.placeholder(dtype=tf.float32, shape=[None, self.params['num_classes']], name='y')
        return self.model['X'], self.model['y']
        
    def _create_conv_layer(self, prev_layer, kernel_size, num_filters, name):
        
        '''
        Create a convolution layer with `kernel_size` kernel and
        `num_filter` number of activation maps. The activation 
        function is relu and stride-length is 1
        '''

        with tf.variable_scope(name) as scope:
            
            shape = kernel_size + [num_filters]
            w = tf.get_variable(name='weights', shape=shape, dtype=tf.float32, 
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable(name='bias', shape=[num_filters], dtype=tf.float32,
                               initializer=tf.random_normal_initializer())
            conv2d = tf.nn.conv2d(input=prev_layer, filter=w, strides=[1,1,1,1], padding="SAME")
            out = tf.nn.relu(conv2d + b)
            return out
    
    def _create_maxpool(self, prev_layer, name):
        
        '''
        Maxpooling layer with 2 stride length.
        '''

        with tf.variable_scope(name) as scope:
            
            pool = tf.nn.max_pool(value=prev_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            return pool
    
    def _create_avgpool(self, prev_layer, name):
        
        '''
        Average pooling layer
        '''

        with tf.variable_scope(name) as scope:
            
            pool = tf.nn.avg_pool(value=prev_layer, ksize=[1,2,2,1], strides=[1,1,1,1], padding="SAME")
            return pool
         
    def _create_residual_layer(self, prev_layer, residual_layer, name):
        
        '''
        Residual layer for providing the skip connection
        '''

        with tf.variable_scope(name) as scope:
            
            out = tf.add(prev_layer, residual_layer)
            return out
        
    def _create_fully_connected(self, prev_layer, num_output, name):
        
        '''
        Fully connected layer with relu activation layer
        '''

        with tf.variable_scope(name) as scope:
            
            shape = prev_layer.get_shape().as_list()
            try:
                flat = shape[1]*shape[2]*shape[3]
            except:
                flat = shape[1]
            
            w = tf.get_variable(name='weights', shape=[flat, num_output], dtype=tf.float32, 
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='bias', shape=[num_output], dtype=tf.float32,
                               initializer=tf.random_normal_initializer())
            out = tf.nn.relu(tf.matmul(tf.reshape(prev_layer, shape=[-1, flat]), w) + b)
            return out
    
    def _create_softmax(self, prev_layer, num_output=None, name='softmax'):
        
        '''
        Softmax activation layer
        '''

        if not num_output:
            num_output = self.params['num_classes']
        
        with tf.variable_scope(name) as scope:
            
            w = tf.get_variable(name='weights', shape=[prev_layer.get_shape().as_list()[1], num_output],
                               dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='bias', shape=[num_output], dtype=tf.float32,
                               initializer=tf.random_normal_initializer())
            out = tf.nn.softmax(tf.matmul(prev_layer, w) + b)
            return out
    
    def _create_loss(self, predicted_labels, name='loss'):
        
        '''
        Categorical cross entropy loss function
        '''

        with tf.variable_scope(name) as scope:
            
            self.model['loss'] = tf.reduce_mean(-tf.reduce_sum(self.model['y']*tf.log(predicted_labels + 1e-8),
                                                               reduction_indices=[1]))
            
    
    def _create_optimizer(self, name='optimizer'):
        
        '''
        Adaptive Moment (AdaM) Optimizer to minimize the CE Loss
        '''
        
        with tf.variable_scope(name) as scope:
            
            self.model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.model['optimizer'] = tf.train.AdamOptimizer(self.params['learning_rate']).minimize(
                loss=self.model['loss'], global_step=self.model['global_step'])
    
    def _create_summary(self):
        
        '''
        Plot summary of the losses as histogram and loss curve
        '''

        tf.summary.scalar(name='loss', tensor=self.model['loss'])
        tf.summary.histogram(name='loss', values=self.model['loss'])
        self.model['summary_op'] = tf.summary.merge_all()
    
    def _calculate_accuracy(self, predicted_labels, name='accuracy'):
        
        '''
        Calculating accuracy of the labels predicted for a mini-batch
        '''

        with tf.variable_scope(name) as scope:
            
            correct_labels = tf.equal(tf.argmax(predicted_labels, 1), 
                                     tf.argmax(self.model['y'], 1))
            self.model['accuracy'] = tf.reduce_mean(tf.cast(correct_labels, dtype=tf.float32))
            
    def _train(self, X, y, X_val, y_val):
        
        '''
        Training method. Applying SGD to minimze the loss
        '''

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        batch_size = self.params['batch_size']
        n_batches = X.shape[0]/batch_size
        
        with tf.Session() as sess:
            
            sess.run(init)
            writer = tf.summary.FileWriter('graphs/resnet', sess.graph)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/resnet/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                print('Restoring Session..')
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            point_loss = []
            for i in range(1, self.params['num_epochs']+1):
                
                epoch_loss, epoch_accuracy = 0, 0
                
                for j in range(n_batches):
                    
                    x_batch, y_batch = X[j*batch_size:(j+1)*batch_size, :], y[j*batch_size:(j+1)*batch_size, :]
                    _, l, accuracy, summary = sess.run([self.model['optimizer'], self.model['loss'],
                                                       self.model['accuracy'], self.model['summary_op']], 
                                                       feed_dict={self.model['X']:x_batch, 
                                                                  self.model['y']:y_batch})
                    point_loss.append(l)
                    epoch_loss += l
                    epoch_accuracy += accuracy
                
                writer.add_summary(summary, global_step=i)
                
                print('Epoch: {}\nLoss: {}\tAccuracy: {}'.format(i,epoch_loss,epoch_accuracy/n_batches))
                
                if i % self.params['train_step'] == 0:
                    
                    print('Saving Checkpoint...')
                    saver.save(sess, 'checkpoints/resnet/resnet', i)
                    cross_val_score = sess.run([self.model['accuracy']],
                                               feed_dict={self.model['X']:X_val,
                                                        self.model['y']:y_val})
                    print('Cross Validation Score: {}').format(np.round(cross_val_score, 2))
                
            
            writer.close()
        
        return point_loss