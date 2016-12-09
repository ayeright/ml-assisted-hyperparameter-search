from __future__ import print_function
import numpy as np
import pandas as pd
import sys
from os import listdir
from os.path import join, isfile
from cnn_randomsearch_cifar10 import validation_split, normalise_data
from cnn_randomsearch_cifar10 import save_results, hyperparameter_dists
from cnn_randomsearch_cifar10 import train_cnn, eval_model
from sklearn.ensemble import RandomForestRegressor
from keras.datasets import cifar10

def draw_candidates(param_dists, feature_columns):    
    # draw hyperparameter candidate sets from same distributions as used in random search
    hp_sets = pd.DataFrame(columns=feature_columns)
    nb_obs = 1000000

    # learning rate
    hp_sets['sgd_lr'] = np.exp(param_dists['sgd_lr'].rvs(size=nb_obs))
    
    # learning rate decay
    hp_sets['sgd_lr_decay'] = np.exp(param_dists['sgd_lr_decay'].rvs(size=nb_obs))
    
    # momentum
    hp_sets['sgd_momentum'] = np.random.choice(param_dists['sgd_momentum'], size=nb_obs, replace=True)
    
    # nesterov
    hp_sets['sgd_nesterov'] = np.random.choice(param_dists['sgd_nesterov'], size=nb_obs, replace=True)
    
    # number of convolutional layers: 
    hp_sets['nb_conv_layers'] = param_dists['nb_conv_layers'].rvs(size=nb_obs)
    
    # number of fully-connected layers
    hp_sets['nb_fc_layers'] = param_dists['nb_fc_layers'].rvs(size=nb_obs)
    
    # number of filters in first convolutional layer
    hp_sets['nb_conv_filters_1'] = np.round(np.exp(param_dists['nb_conv_filters_1'].rvs(size=nb_obs)))
    hp_sets['nb_conv_filters_1'] = hp_sets['nb_conv_filters_1'].apply(int)
    
    # number of filters in last convolutional layer
    hp_sets['conv_filter_ratio'] = param_dists['conv_filter_ratio'].rvs(size=nb_obs)
    hp_sets['nb_conv_filters_2'] = np.round(hp_sets['nb_conv_filters_1'] * hp_sets['conv_filter_ratio'])
    hp_sets['nb_conv_filters_2'] = hp_sets['nb_conv_filters_2'].apply(int)
    hp_sets.drop(['conv_filter_ratio'], axis=1, inplace=True)
    
    # number of units in first fully-connected layer
    hp_sets['fc_size'] = np.round(np.exp(param_dists['fc_size'].rvs(size=nb_obs)))
    hp_sets['fc_size'] = hp_sets['fc_size'].apply(int)
    
    # dropout after convolutional layers and after last fully-connected layer:
    hp_sets['dropout_conv'] = param_dists['dropout_conv'].rvs(size=nb_obs)
    hp_sets['dropout_fc'] = param_dists['dropout_fc'].rvs(size=nb_obs)
    
    return hp_sets

def ml_search(X_train, y_train, X_val, y_val, X_test, y_test, param_dists):
    # get list of all files in results directory
    file_names = [f for f in listdir(results_dir) if isfile(join(results_dir, f))]
    
    # define feature and target columns in results
    feature_columns = ['nb_conv_layers', 'nb_conv_filters_1', 'nb_conv_filters_2',
                   'nb_fc_layers', 'fc_size', 
                   'dropout_conv', 'dropout_fc',
                   'sgd_lr', 'sgd_momentum', 'sgd_lr_decay', 'sgd_nesterov']
    target_column = ['val_logloss']
    columns = feature_columns + target_column
    
    for _ in range(nb_experiments):              
        # loop through all files and append results into a single data frame
        for i, f in enumerate(file_names):
            df_temp = pd.read_csv(join(results_dir, f))  
            df_temp = df_temp[columns]
            if i == 0:
                val_results = df_temp
            else:
                val_results = val_results.append(df_temp, ignore_index=True)
                
        # train a random forest to predict validation score based on hyperparameter values
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5) 
        rf.fit(val_results[feature_columns], np.ravel(val_results[target_column])) 
        
        # draw hyperparameter candidate sets
        hp_sets = draw_candidates(param_dists, feature_columns)        
        
        # predict validation scores for hyperparameter candidate sets
        hp_sets['pred_val_logloss'] = rf.predict(hp_sets)
        
        # keep the best
        hp_sets.sort_values(by='pred_val_logloss', inplace=True)
        hp_sets.reset_index(drop=True, inplace=True)  
        hp_sets = hp_sets.iloc[0:nb_trials,:]
                       
        # low through best hyperparameter sets and train CNN for each one
        results_df = pd.DataFrame(columns=columns) 
        for i, params in hp_sets.iterrows(): 
            # train CNN
            model = train_cnn(X_train, y_train, X_val, y_val, params, nb_epoch)

            # evaluate model on validation and test sets
            val_logloss, val_acc = eval_model(model, X_val, y_val) 
            test_logloss, test_acc = eval_model(model, X_test, y_test)                          
            
            # append hyperparameters and scores to results DataFrame
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
                            'pred_val_logloss': params['pred_val_logloss'] ,
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

    ml_search(X_train, y_train, X_val, y_val, X_test, y_test, param_dists)

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