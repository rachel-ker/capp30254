'''
Homework 3
Applying the Pipeline to DonorsChoose Data

Rachel Ker
'''
import datetime
import pandas as pd

import etl
from iteratemodels import *

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


metrics = ['precision_at_1','precision_at_2','precision_at_5','precision_at_10','precision_at_20','precision_at_30',
           'precision_at_50',' recall_at_1','recall_at_2','recall_at_5','recall_at_10','recall_at_20','recall_at_30','recall_at_50','auc-roc']
labels = ["jan12-jun12/jul12-dec12", "jan12-dec12/jan13-jun13", "jan12-jun13/jul13-dec13"]


def hw3(gridsize, outfile, models_to_run):
    '''
    Code to apply machine learning pipeline to DonorChoose Data
    Returns a table with all the models run and their relevant metrics
    '''
    filename = "data/projects_2012_2013.csv"
    df = etl.read_csvfile(filename)

    # Create Label
    dates = ["date_posted", "datefullyfunded"]
    df = etl.replace_dates_with_datetime(df, dates)

    y_col = "notfullyfundedin60days"
    df = create_label(df, y_col)

    # Create discrete variables
    df = etl.discretize(df, 'total_price_including_optional_support', [92, 245, 510, 753])
    df = etl.discretize(df, 'students_reached', [1, 31])

    # Train and evaluate models
    all_models = iterate_models(df, gridsize, y_col, outfile, models_to_run)
    print("-- done with script -- ")
        
    return all_models    


def get_best_models(df, labels, parameters, metrics):
    '''
    Find best models for each train test set based on metrics
    Inputs:
        df: pandas dataframe of all results
        labels: list of the label indicating sets
        parameters: list of parameters we care about
        metrics: list of metrics
    Returns a dictionary of best model for each metric
    '''
    all_best = []
    for lab in labels:
        data = df[df['trainsetlabel']==lab]
        best = pipeline.best_model(data, metrics, parameters)
        all_best.append(best)
    return all_best


def create_label(df, label_name):
    '''
    Creates a label column
    Inputs:
        df: pandas dataframe
        label_name: (str) label column name
    Returns a dataframe with label col
    '''
    df.loc[:,'60daysafterpost'] = df['date_posted'] + datetime.timedelta(days=60)
    df.loc[:, label_name] = df['datefullyfunded'] > df['60daysafterpost']
    df[label_name] = df[label_name].astype(int)
    return df

if __name__ == '__main__':
    gridsize = 'small'
    outfile = "results_"+gridsize+".csv"
    models_to_run=['RF', 'ET', 'GB', 'AB', 'BAG', 'DT', 'KNN', 'LR', 'SVM']
    hw3(gridsize, outfile, models_to_run)
    


    
    
