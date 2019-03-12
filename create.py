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
from models.neural_network import Neural_Network
from models.resnet import ResNet

def create_resnet(params):
    
    '''
    Creating model for the resnet network. The model consists of 'n' resnet block.
    Each resnet block consists of 2 convolution units and a skip connection from
    one of the previous blocks. The model starts with a combination of convolution
    - maxpool layer and is succeeded by 'n' residual blocks. The last layer consist
    of an average pooling layer followed by fully connected and softmax activation.
    '''

    tf.reset_default_graph()
    layers, list_layers = {}, []
    resnet = ResNet(params)
    X, y = resnet._create_placeholders()

    layers['conv1'] = resnet._create_conv_layer(prev_layer=X, kernel_size=[7,7,1], num_filters=32, name='conv1')
    layers['pool1']= resnet._create_maxpool(layers['conv1'], name='pool1')
    list_layers += ['conv1', 'pool1']

    l, k, r = 1, 2, 1
    residual_layer = layers['pool1']
    prev_layer = layers['pool1']
    
    while l <= params['num_blocks']:
        
        layers['conv' + str(k)] = resnet._create_conv_layer(prev_layer=prev_layer, kernel_size=[3,3,32], num_filters=32, name='conv' + str(k))
        prev_layer = layers['conv' + str(k)]
        layers['conv' + str(k+1)] = resnet._create_conv_layer(prev_layer=prev_layer, kernel_size=[3,3,32], num_filters=32, name='conv' + str(k+1))
        prev_layer = layers['conv' + str(k+1)]
        layers['res' + str(r)] = resnet._create_residual_layer(prev_layer=prev_layer, residual_layer=residual_layer, name='res'+str(r))
        
        prev_layer = layers['res' + str(r)]
        residual_layer = layers['res' + str(r)]
        list_layers += ['conv' + str(k), 'conv' + str(k+1), 'res' + str(r)]

        k += 2
        r += 1
        l += 1

    layers['avgpool'] = resnet._create_avgpool(prev_layer=prev_layer, name='avgpool')
    layers['full'] = resnet._create_fully_connected(prev_layer=layers['avgpool'], num_output=100, name='full')
    layers['softmax'] = resnet._create_softmax(prev_layer=layers['full'])
    resnet._create_loss(layers['softmax'])
    resnet._create_optimizer()
    resnet._create_summary()
    resnet._calculate_accuracy(layers['softmax'])

    list_layers += ['avgpool', 'full', 'softmax']

    return resnet, layers, list_layers

def create_neural_network(params):
    
    '''
    Creating an L layered neural network. Each layer consist
    of a fully connected layer with relu activation and a dropout
     layer. The final layer consist of softmax activation units.    
    '''
    tf.reset_default_graph()
    layers, list_layers = {}, []
    
    nn = Neural_Network(params)
    X, y = nn.create_placeholders()
    prev_layer = X

    for l in range(1, params['num_layers']):
        
        layers['hidden' + str(l)] = nn._fully_connected(prev_layer, params['num_hidden'][l-1], 'hidden' + str(l))
        layers['dropout' + str(l)] = nn._create_dropout(layers['hidden' + str(l)], 'dropout' + str(l))
        prev_layer = layers['dropout' + str(l)]
        list_layers += ['hidden' + str(l), 'dropout' + str(l)]

    layers['softmax'] = nn._create_softmax(prev_layer)
    list_layers += ['softmax']
    nn._create_loss(layers['softmax'])
    nn._create_optimizer()
    nn._create_summaries()
    nn._accuracy(layers['softmax'])

    return nn, layers, list_layers