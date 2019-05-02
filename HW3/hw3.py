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


def hw3(gridsize, outfile):
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
    models_to_run=['RF', 'ET', 'GB', 'AB', 'BAG', 'DT', 'KNN', 'LR', 'SVM']
    all_models = iterate_models(df, gridsize, y_col, outfile, models_to_run) 
    return all_models    


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
    gridsize = 'test'
    outfile = "results_"+gridsize+".csv"
    hw3(gridsize, outfile)
    


    
    
