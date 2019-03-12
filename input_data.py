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
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from random import shuffle



def process_data(path_to_file, num_train, num_val):

    mnist = input_data.read_data_sets(path_to_file, one_hot=True)
    X_train, X_val, y_train, y_val = train_test_split(mnist.train.images, mnist.train.labels, random_state=7, test_size=0.3)
    
    # Shuffling the data
    idx_list = range(num_train)
    shuffle(idx_list)
    X_train, y_train = X_train[idx_list, :], y_train[idx_list, :]
    idx_list = range(num_val)
    shuffle(idx_list)
    X_val, y_val = X_val[idx_list, :], y_val[idx_list, :]
    print('Shape of Training Data: {}'.format(X_train.shape))
    print('Shape of Validation Data: {}'.format(X_val.shape))

    # Normalising the Data
    print('='*25+' Data Characteristics '+'='*25)
    print('Mean Before Normalisation: Training: {}, Validation: {}'.format(np.round(np.mean(X_train), 2), np.round(np.mean(X_val), 2)))
    print('Standard Deviation Before Normalisation: Training: {}, Validation: {}'.format(np.round(np.std(X_train), 2), np.round(np.std(X_val), 2)))

    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train -= mean
    X_train /= std

    X_val -= mean
    X_val /= std

    print('Mean After Normalisation: Training: {}, Validation: {}'.format(np.mean(X_train), np.mean(X_val)))
    print('Standard Deviation After Normalisation: Training: {}, Validation: {}'.format(np.std(X_train), np.std(X_val)))

    return X_train, y_train, X_val, y_val
