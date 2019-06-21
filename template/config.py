#!/usr/bin/env python
# coding: utf-8

'''
Configuration File

Author: Rachel Ker
'''

## DATA FILE / LABEL

CSVFOLDER="../data/"
#TODO: Include filenames here
FILENAME=CSVFOLDER + ".csv"
DATA_DIR = CSVFOLDER + "traintest"


## MODELS AND RESULTS
GRIDSIZE = 'small'
#MODELS = ['LR', 'RF', 'GB', 'DT']
MODELS = ['RF', 'ET', 'GB', 'XG', 'AB', 'BAG', 'KNN', 'LR', 'SVM', 'NB', 'DT']
EVAL_METRICS_BY_LEVEL = (['accuracy', 'precision', 'recall', 'f1'],\
                         [1,2,5,10,20,30,50])
EVAL_METRICS = ['auc']
ID = 'ID'

#TODO: change the train test cols and nums
TRAIN_TEST_COL = 'year'
TRAIN_TEST_NUM = [2010, 2017]

RESULTS_DIR = "results"
GRAPH_FOLDER = "graphs"
RESULTS_FILE = "results_" + GRIDSIZE
SEED = 0


## GRAPH OPTIONS
PLOT_PR = True
VISUALIZE_DT = True
FEATURE_IMP = True
BIAS = True
SAVE_PRED = True
SCOREHIST = True
POP_THRESHOLD = 10


## VARIABLES
#TODO: change the labels and variables accordingly

LABELS = []
VARIABLES = {
             'TO_DISCRETIZE' : [{'var': (['bins', 'bins'], ['labels','labels'])}],
             'DATES' : [],
             'TIMES' : [],
             'MISSING' : {'MISSING_CAT': [],
                          'IMPUTE_ZERO': []
                           },
             'INDICATOR': {'missing': []
                           },
             'CONTINUOUS_VARS_MINMAX' : [],
             'CATEGORICAL_VARS' : [],
             'SPECIAL_DUMMY': [],
             'VARS_TO_EXCLUDE' : ['ID'] + LABELS,
             'NO_CLEANING_REQ': [],
             'BIAS': ['gender', 'label', 'id', 'score'],
             'BIAS_METRICS': {'metrics':['ppr','pprev','fnr','fpr', 'for'], 'min_group_size': None}
             }


def define_clfs_params(grid_size):
    """
    This functions defines parameter grid for all the classifiers
    Inputs:
       grid_size: how big of a grid do you want. it can be test, small, or large
    Returns a set of model and parameters
    Raises KeyError: Raises an exception.
    """

    large_grid = { 
    'RF':   {'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],
             'min_samples_split': [2,5,10,50,100], 'n_jobs': [-1], 'random_state': [SEED]},
    'ET':   {'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100],
             'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10,50,100], 'n_jobs': [-1], 'random_state': [SEED]},
    'AB':   {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000], 'random_state': [SEED]},
    'GB':   {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0],
             'max_depth': [1,5,10,20,50,100], 'random_state': [SEED]},
    'KNN':  {'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'DT':   {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'min_samples_split': [2,5,10,50,100], 'random_state': [SEED]},
    'SVM':  {'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10], 'random_state': [SEED]},
    'LR':   {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'random_state': [SEED]},
    'BAG':  {'n_estimators': [1,10,100,1000,10000], 'n_jobs': [-1], 'random_state': [SEED]},
    'NB':   {'alpha': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'fit_prior': [True, False]},
    'XG':   {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0],
             'max_depth': [1,5,10,20,50,100], 'random_state': [SEED]}
           }
    
    small_grid = {
    'RF':   {'n_estimators': [10,100,1000], 'max_depth': [1,5,10,20], 'max_features': ['sqrt','log2'],
             'min_samples_split': [2,10,50], 'n_jobs': [-1], 'random_state': [SEED]},
    'ET':   {'n_estimators': [10,100,1000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20],
             'max_features': ['sqrt','log2'],'min_samples_split': [2,10,50], 'n_jobs': [-1], 'random_state': [SEED]},
    'AB':   {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10,100], 'random_state': [SEED]},
    'GB':   {'n_estimators': [10,100], 'learning_rate' : [0.1],'subsample' : [0.5],
             'max_depth': [5,10], 'random_state': [SEED]},
    'KNN':  {'n_neighbors': [10,25],'weights': ['uniform','distance'],'algorithm': ['auto']},
    'DT':   {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20], 'min_samples_split': [2,10,50], 'random_state': [SEED]},
    'SVM':  {'C' :[0.01,0.1,1,10], 'random_state': [SEED]},
    'LR':   {'penalty': ['l1','l2'], 'C': [0.01,0.1,1,10], 'random_state': [SEED]},
    'BAG':  {'n_estimators': [10,100], 'n_jobs': [None], 'random_state': [SEED]},
    'NB':   {'alpha': [0.01,0.1,1,10], 'fit_prior': [True, False]},
    'XG':   {'n_estimators': [10,100], 'learning_rate' : [0.1],'subsample' : [0.5],
             'max_depth': [5,10], 'random_state': [SEED]}
            }
       
    test_grid = {
    'RF':   {'n_estimators': [1], 'max_depth': [5], 'max_features': ['sqrt'], 'min_samples_split': [10], 'n_jobs': [-1], 'random_state': [SEED]},
    'ET':   {'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [5],
             'max_features': ['sqrt'],'min_samples_split': [10], 'n_jobs': [-1], 'random_state': [SEED]},
    'AB':   {'algorithm': ['SAMME.R'], 'n_estimators': [5], 'random_state': [SEED]},
    'GB':   {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [5], 'random_state': [SEED]},
    'KNN':  {'n_neighbors': [1],'weights': ['uniform'],'algorithm': ['auto']},
    'DT':   {'criterion': ['gini'], 'max_depth': [5], 'min_samples_split': [10], 'random_state': [SEED]},
    'SVM':  {'C' :[10], 'random_state': [SEED]},
    'LR':   {'penalty': ['l1'], 'C': [10], 'random_state': [SEED]},
    'BAG':  {'n_estimators': [1], 'n_jobs': [-1], 'random_state': [SEED]},
    'NB':   {'alpha': [1], 'fit_prior': [True]},
    'XG':   {'n_estimators': [1], 'learning_rate' : [0.1], 'subsample' : [0.5], 'max_depth': [5], 'random_state': [SEED]}  
            }

    
    if (grid_size == 'large'):
        return large_grid
    elif (grid_size == 'small'):
        return small_grid
    elif (grid_size == 'test'):
        return test_grid
    else:
        return 0, 0
