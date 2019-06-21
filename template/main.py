#!/usr/bin/env python
# coding: utf-8

'''
Putting everything together
Running models and getting results

Author: Rachel Ker
'''

import os
import config
import pipeline as pp


def go():
    '''
    Putting everything together
    '''
    run_models()


def preprocess(df):
    '''
    Clean and preprocess data
    '''
    # change to correct types
    # missing and imputation
    # discretize
    # dummy
    return df
    

def run_models(labels=config.LABELS, results_dir=config.RESULTS_DIR, graph_dir=config.GRAPH_FOLDER,
               results_file=config.RESULTS_FILE, data_dir=config.DATA_DIR, variables=config.VARIABLES, 
               models=config.MODELS, eval_metrics=config.EVAL_METRICS,eval_metrics_by_level=config.EVAL_METRICS_BY_LEVEL, 
               grid=config.define_clfs_params(config.GRIDSIZE), period=config.TRAIN_TEST_NUM, threshold=config.POP_THRESHOLD, 
               plot_pr=config.PLOT_PR, compute_bias=config.BIAS, visualize_dt=config.VISUALIZE_DT, 
               ft_impt=config.FEATURE_IMP, score_hist=config.SCOREHIST, save_pred=config.SAVE_PRED):
    '''
    Build and test models for all train test sets

    Inputs:
        labels: list of labels
        results_dir: (str) directory to place results
        graphs_dir: (str) directory to place graphs
        results_file: (str) file for results of the models
        data_dir: (str) directory for train test sets
        variables: dictionary of variables defined in config for cleaning
        models: list of models to run
        eval_metrics: list of eval metrics
        eval_metrics_by_level: list of eval metrics with thresholds
        grid: grid of models and parameters to run
        period: period of train test sets we care about
        plot_pr: (Boolean) save PRC or not
        compute_bias: (Boolean) compute bias or not
        visualize_dt: (Boolean) visualize dt or not
        ft_impt: (Boolean) save feature imp or not
        score_hist: (Boolean) save score histogram or not
        save_pred: (Boolean) save predictions or not
    
    Returns None
    '''
    
    # check if necessary data and results directories exist
    print(os.getcwd())
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)

    # initialize variables
    bias_lst = variables['BIAS']
    bias_dict = variables['BIAS_METRICS']

    for label in labels:
        tt = period[0]
        
        while tt <= period[1]:            
            # check if training/test data exists, create it if not
            test_csv = os.path.join(data_dir, "{}_test.csv".format(tt))
            train_csv = os.path.join(data_dir, "{}_train.csv".format(tt))
            
            if not os.path.exists(test_csv) or not os.path.exists(train_csv):
                print('Creating training and test sets')
                pass

            # read in training and test data 
            df_test = pp.get_csv(test_csv)
            df_train = pp.get_csv(train_csv)

            print('Running train-test set: {}, Label: {}'.format(tt, label))

            # Pre-process data, except scaling
            print('Pre-processing data')
            df_test = preprocess(df_test)
            df_train = preprocess(df_train)

            # define list of features
            attributes_lst = [x for x in df_train.columns if x not in variables['VARS_TO_EXCLUDE']]
            for attr in attributes_lst:
                if attr not in df_test.columns:
                    df_test.loc[:,attr] = 0
            print('Training set has {} features'.format(len(attributes_lst)))
        
            # run models
            pp.classify(
                df_train, df_test, label, models, eval_metrics, eval_metrics_by_level, grid, 
                attributes_lst, bias_lst, bias_dict, tt, variables, results_dir, results_file, graph_dir,  
                threshold, plot_pr, compute_bias, visualize_dt, ft_impt, score_hist, save_pred
            )

            tt += 1
    

if __name__ == '__main__':
    go()