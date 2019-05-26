'''
Applying the Pipeline to DonorsChoose Data
Rachel Ker
'''
import datetime

import etl
import pipeline
import config

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def go(gridsize, outfile, models_to_run):
    '''
    Code to apply machine learning pipeline to DonorChoose Data
    Returns a table with all the models run and their relevant metrics

    Inputs:
        gridsize: 'test', 'small', 'large'
        outfile: (str) name of output file
        models_to_run: (list) list of models
    '''
    df = etl.read_csvfile(config.DATAFILE)

    # Create Label
    df = etl.replace_dates_with_datetime(df, config.TO_DATE)
    df = create_label(df, config.OUTCOME, config.PREDICTION_PERIOD)

    # Train and evaluate models
    all_models = pipeline.iterate_models(df, gridsize, config.OUTCOME, config.FEATURES, config.MISSING,
                                         config.TO_DISCRETIZE, config.DISCRETE_LEVELS,  
                                         config.METRICS_THRESHOLD, config.OTHER_METRICS, config.DATE_TO_SPLIT,
                                         config.TRAIN_TEST_DATES, config.TRAIN_TEST_LABELS,
                                         config.PREDICTION_PERIOD, config.TEST_PERIOD, 
                                         outfile, models_to_run)
    print("-- done with script -- ")
        
    return all_models


def create_label(df, label_name, time_period):
    '''
    Creates a label column
    Inputs:
        df: pandas dataframe
        label_name: (str) label column name
    Returns a dataframe with label col
    '''
    df.loc[:,'ndaysafterpost'] = df['date_posted'] + datetime.timedelta(days=time_period)
    df.loc[:, label_name] = df['datefullyfunded'] > df['ndaysafterpost']
    df[label_name] = df[label_name].astype(int)
    return df


if __name__ == '__main__':
    gridsize = config.GRIDSIZE
    outfile = config.OUTFILE
    models_to_run= config.MODELS
    go(gridsize, outfile, models_to_run)


    
    
