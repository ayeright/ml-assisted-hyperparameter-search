from __future__ import print_function
import numpy as np
import pandas as pd 
import sys
from os.path import join
from datetime import datetime
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from scipy import stats
    
def validation_split(X_train, y_train):   
    # split the data into training and validation sets
    nb_img = len(X_train)
    train_size = 0.8
    nb_train = int(np.floor(nb_img * train_size))
    X_val = X_train[nb_train:, :, :, :]
    y_val = y_train[nb_train:, :]
    X_train = X_train[0:nb_train, :, :, :]
    y_train = y_train[0:nb_train, :]  
    
    return X_train, y_train, X_val, y_val
    
def normalise_data(X_train, X_val, X_test):
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    # calculate the mean and standard deviation of each feature 
    # using the training data ONLY, not validation or test data
    shp = X_train.shape
    X_train = np.reshape(X_train, (shp[0], shp[1] * shp[2] * shp[3]))
    feature_mean = np.mean(X_train, axis=0)
    feature_std = np.std(X_train, axis=0)
    
    # normalise training data
    X_train -= feature_mean
    X_train /= feature_std
    X_train = np.reshape(X_train, shp) 
    
    # normalise validation data
    shp = X_val.shape
    X_val = np.reshape(X_val, (shp[0], shp[1] * shp[2] * shp[3]))
    X_val -= feature_mean
    X_val /= feature_std
    X_val = np.reshape(X_val, shp)
    
    # normalise test data
    shp = X_test.shape
    X_test = np.reshape(X_test, (shp[0], shp[1] * shp[2] * shp[3]))
    X_test -= feature_mean
    X_test /= feature_std
    X_test = np.reshape(X_test, shp)
    
    return X_train, X_val, X_test
    
def hyperparameter_dists():
    # define hyperparameter distributions
    param_dists = {}
    
    # learning rate
    # drawn exponentially from 1e-5 to 1 
    param_dists['sgd_lr'] = stats.uniform(np.log(1e-5), np.log(1) - np.log(1e-5))
    
    # learning rate decay
    # drawn exponentially from 1e-5 to 1 
    param_dists['sgd_lr_decay'] = stats.uniform(np.log(1e-5), np.log(1) - np.log(1e-5))
    
    # momentum
    # drawn uniformly from {0.5, 0.9, 0.95, 0.99}
    param_dists['sgd_momentum'] = [0.5, 0.9, 0.95, 0.99] 
    
    # nesterov momentum
    # drawn uniformly form {True, False}
    param_dists['sgd_nesterov'] = [True, False] 
    
    # number of convolutional layers: 
    # draw uniformly from {1, 2}
    param_dists['nb_conv_layers'] = stats.randint(1,3) 
    
    # number of fully-connected layers:
    # drawn uniformly from {1, 2}
    param_dists['nb_fc_layers'] = stats.randint(1,3)  
    
    # number of filters in first convolutional layer
    # drawn exponentially from 32 to 128
    param_dists['nb_conv_filters_1'] = stats.uniform(np.log(32), np.log(128) - np.log(32))
    
    # ratio of number of filters in last convolutional layer 
    # over number of filters in first convolutional layer
    # drawn uniformly from 1 to 3
    param_dists['conv_filter_ratio'] = stats.uniform(1, 2)
    
    # number of units in first fully-connected layer
    # drawn exponentially from 256 to 4096
    param_dists['fc_size'] = stats.uniform(np.log(256), np.log(4096) - np.log(256))
    
    # dropout after convolutional layers and after last fully-connected layer
    # drawn uniformly from (0, 1)
    param_dists['dropout_conv'] = stats.uniform(0, 1)
    param_dists['dropout_fc'] = stats.uniform(0, 1)    

    return param_dists
    
def sample_hyperparameters(param_dists):
    # sample hyperparameters
    params = {}
    params['nb_conv_layers'] = param_dists['nb_conv_layers'].rvs(size=1)[0]
    params['nb_conv_filters_1'] = param_dists['nb_conv_filters_1'].rvs(size=1)[0]
    params['conv_filter_ratio'] = param_dists['conv_filter_ratio'].rvs(size=1)[0]
    params['nb_fc_layers'] = param_dists['nb_fc_layers'].rvs(size=1)[0]
    params['fc_size'] = param_dists['fc_size'].rvs(size=1)[0]
    params['dropout_conv'] = float(param_dists['dropout_conv'].rvs(size=1)[0])
    params['dropout_fc'] = float(param_dists['dropout_fc'].rvs(size=1)[0])
    params['sgd_lr'] = param_dists['sgd_lr'].rvs(size=1)[0]
    params['sgd_momentum'] = param_dists['sgd_momentum'][stats.randint(0,4).rvs(size=1)]
    params['sgd_lr_decay'] = param_dists['sgd_lr_decay'].rvs(size=1)[0]
    params['sgd_nesterov'] = param_dists['sgd_nesterov'][stats.randint(0,2).rvs(size=1)] 
    
    # exponentiate samples from log domain and round to nearest integer if necessary
    params['sgd_lr'] = np.exp(params['sgd_lr'])
    params['sgd_lr_decay'] = np.exp(params['sgd_lr_decay'])
    params['nb_conv_filters_1'] = int(np.round(np.exp(params['nb_conv_filters_1'])))
    params['fc_size'] = int(np.round(np.exp(params['fc_size'])))
    
    # calculate number of filters in second convolutional block
    params['nb_conv_filters_2'] = int(np.round(params['nb_conv_filters_1'] * params['conv_filter_ratio']))    
            
    return params
    
def make_model(nb_conv_layers, nb_conv_filters_1, nb_conv_filters_2,
               nb_fc_layers, fc_size, 
               dropout_conv, dropout_fc,
               sgd_lr, sgd_momentum, sgd_lr_decay, sgd_nesterov):

    '''    
    Creates model comprised of up to 4 convolutional layers followed by up to 2 dense layers

    nb_conv_layers: value of N in INPUT -> [[CONV -> RELU]*N -> POOL]*M -> [FC -> RELU]*K -> FC
    nb_conv_filters_1: number of convolutional filters in first convolutional layer
    nb_conv_filters_2: number of convolutional filters in second convolutional layer
    nb_fc_layers: value of K in INPUT -> [[CONV -> RELU]*N -> POOL]*M -> [FC -> RELU]*K -> FC
    fc_size: number of neurons in fully connected layers
    dropout_conv: dropout after convolutional layers
    droupout_fc: dropout after fully-connected layers
    sgd_lr: learning rate for SGD
    sgd_momentum: momentum for SGD
    sgd_lr_decay: learning rate decay for SGD
    sgd_nesterov: boolean indicating whether to use Nesterov momentum for SGD
    '''   
    
    # the CIFAR-10 images are 32x32 RGB
    img_rows, img_cols, img_channels = 32, 32, 3
    # there are 10 classes in the data
    nb_classes = 10    
    
    model = Sequential()

    # add first convolutional layer followed by ReLu  
    model.add(Convolution2D(nb_conv_filters_1, 3, 3,
                            border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))    
    if nb_conv_layers > 1:
        # add another convolutional layer followed by ReLu
        model.add(Convolution2D(nb_conv_filters_1, 3, 3, border_mode='same'))
        model.add(Activation('relu'))        
    # add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
    # add dropout layer
    model.add(Dropout(float(dropout_conv)))
    
    # add another convolutional layer followed by ReLu 
    model.add(Convolution2D(nb_conv_filters_2, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    if nb_conv_layers > 1:
        # add another convolutional layer followed by ReLu
        model.add(Convolution2D(nb_conv_filters_2, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
    # add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
    # add dropout layer
    model.add(Dropout(float(dropout_conv)))  
                            
    # add first fully connected layer followed by ReLu
    model.add(Flatten())
    model.add(Dense(fc_size))
    model.add(Activation('relu')) 
 
    if nb_fc_layers > 1: 
        # add another fully connected layer followed by ReLu               
        model.add(Dense(fc_size))
        model.add(Activation('relu'))                 
    model.add(Dropout(float(dropout_fc)))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))   
     
    # define SCG optimiser
    optimiser = SGD(lr=sgd_lr, momentum=sgd_momentum, decay=sgd_lr_decay, nesterov=sgd_nesterov)  
      
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])   

    return model
    
def train_cnn(X_train, y_train, X_val, y_val, params, nb_epoch):
    # make model
    model = make_model(params['nb_conv_layers'], params['nb_conv_filters_1'], 
                       params['nb_conv_filters_2'],
                       params['nb_fc_layers'], params['fc_size'], 
                       params['dropout_conv'], params['dropout_fc'],
                       params['sgd_lr'], params['sgd_momentum'], 
                       params['sgd_lr_decay'], params['sgd_nesterov'])
                       
    # define early stopping                           
    early_stopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')                           
    
    # fit model
    model.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epoch, 
                      verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])
                      
    return model
                      
def eval_model(model, X, y):
    loss_and_accuracy = model.evaluate(X, y)
    loss = loss_and_accuracy[0]
    acc = loss_and_accuracy[1] 
    return loss, acc
    
def save_results(results_df, results_dir):
    # save results to csv with timestamp as filename                            
    time_now = str(datetime.now())
    time_now = time_now.replace(':', '')
    time_now = time_now.replace('.', '')
    results_file = join(results_dir, time_now)
    results_df.to_csv(results_file)    
    
def random_search(X_train, y_train, X_val, y_val, X_test, y_test, param_dists):
    # define columns of DataFrame where we will save the results
    columns = ['nb_conv_layers', 'nb_conv_filters_1', 'nb_conv_filters_2',
               'nb_fc_layers', 'fc_size', 'dropout_conv', 'dropout_fc', 
               'sgd_lr', 'sgd_momentum', 'sgd_lr_decay', 'sgd_nesterov',
               'val_logloss', 'val_acc', 'test_logloss', 'test_acc']
               
    for _ in range(nb_experiments):        
        results_df = pd.DataFrame(columns=columns)        
        for _ in range(nb_trials):  
            # sample hyperparameters and train resultant CNN                                                                
            params = sample_hyperparameters(param_dists)           
            model = train_cnn(X_train, y_train, X_val, y_val, params, nb_epoch)

            # evaluate model on validation and test sets
            val_logloss, val_acc = eval_model(model, X_val, y_val) 
            test_logloss, test_acc = eval_model(model, X_test, y_test)                                                                    
            
            # append hyperparameters and scores to dataframe
            results_dict = {'nb_conv_layers': params['nb_conv_layers'], 
                            'nb_conv_filters_1': params['nb_conv_filters_1'], 
                            'nb_conv_filters_2': params['nb_conv_filters_2'],
                            'nb_fc_layers': params['nb_fc_layers'],
                            'fc_size': params['fc_size'],
                            'dropout_conv': params['dropout_conv'],
                            'dropout_fc': params['dropout_fc'], 
                            'sgd_lr': params['sgd_lr'],
                            'sgd_momentum': params['sgd_momentum'],
                            'sgd_lr_decay': params['sgd_lr_decay'],
                            'sgd_nesterov': params['sgd_nesterov'],
                            'val_logloss': val_logloss,
                            'val_acc': val_acc,
                            'test_logloss': test_logloss,
                            'test_acc': test_acc}                            
            results_df = results_df.append(results_dict, ignore_index=True)
        
        save_results(results_df, results_dir)                                             
    
def main():   
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() 
    
    X_train, y_train, X_val, y_val = validation_split(X_train, y_train)
    
    X_train, X_val, X_test = normalise_data(X_train, X_val, X_test)
    
    param_dists = hyperparameter_dists()

    random_search(X_train, y_train, X_val, y_val, X_test, y_test, param_dists)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 4:
        nb_experiments = int(args[0])
        nb_trials = int(args[1])
        nb_epoch = int(args[2])
        results_dir = args[3]                                    
        main()
    else:
        print("Four input arguments required. "
                "Aborting.")
        sys.exit(-1)    
