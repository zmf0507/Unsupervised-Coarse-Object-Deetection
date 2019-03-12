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
import sys
import argparse
import os
import json

from models.neural_network import Neural_Network
from models.resnet import ResNet
from create import create_resnet
from create import create_neural_network
from input_data import process_data
from utils import print_layers

def main():
    
    # Adding command line arguments to get training information
    parser = argparse.ArgumentParser(description='Training a neural network based on some configurations')
    parser.add_argument("-c","--config", help="model configuration",type=str)
    parser.add_argument("-n","--name", help="model name", type=str)
    parser.add_argument("dir", help="input data directory", type=str)
    parser.add_argument("-s", "--size", help="size of training set to be trained on", type=int)
    args = parser.parse_args()

    # Loading data
    print('='*25 + ' Loading Data ' + '='*25)
    if args.size:
        num_train, num_val = args.size, int(args.size*0.3)
        print(type(num_val))
    else:
        num_train, num_val = X_train.shape[0], X_val.shape[0]
    X_train, y_train, X_val, y_val = process_data(args.dir, num_train, num_val)
    # Loading configuration
    if args.config:
        config_file = os.path.join('configs', args.config)
        if not os.path.exists(config_file):
            print('File {} does not exists'.format(config_file))
            exit()
        
        with open(config_file, 'r') as fp:
            params = json.load(fp)
    else:
        print('Enter a configuration file')
        
    # Loading network
    if args.name == 'neural_network':
	#try:
        nn, layers, list_layers = create_neural_network(params)
        print_layers(layers, list_layers)
        loss = nn._train(X_train, y_train, X_val, y_val)
        #except:
            #print('Model or the configuration file is broken!')
    
    elif args.name == 'resnet':
        try:
            resnet, layers, list_layers = create_resnet(params)
            print_layers(layers, list_layers)
            loss = resnet._train(X_train.reshape((num_train,28,28,1)), y_train,
                                X_val.reshape((num_val, 28, 28, 1)), y_val)
        except:
            print('Model or the configuration file is broken!')
    else:
        print('Enter a model name')
    
if __name__ == "__main__":

    main()
