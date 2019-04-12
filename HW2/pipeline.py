'''
Homework 2
Machine Learning Pipeline Functions
This python file contains general functions to read, explore,
preprocss data, generate features, build classifer, and evaluate classifier

Rachel Ker
'''

import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn import tree


###################
#    Read Data    #
###################

def read_csvfile(csvfile):
    '''
    Reads in csv and returns a pandas dataframe
    Inputs: csv file path
    Returns a pandas dataframe
    '''
    return pd.read_csv(csvfile)


###################
#  Explore Data   #
###################

def scatterplot(df, x, y):
    '''
    Creates a scatter plot
    Inputs:
        df: pandas dataframe
        x,y: columne names
    Returns a matplotlib.axes.Axes
    '''
    return df.plot.scatter(x, y)


def descriptive_stats(df, var=None):
    '''
    Generates simple descriptive statistics e.g. count,
    mean, standard deviation, min, max, 25%, 50%, 70%
    Inputs:
        df: pandas dataframe
        var: (optional) list of variables of interest
    '''
    if var:
        rv = df.describe()[var]
    rv = df.describe()
    return rv


def detect_outliers(df):
    pass


def get_corr_table(df, y):
    '''
    Get a table of pairwise correlation coefficient of all features
    with respect to the dependent variable
    Inputs:
        df: pandas dataframe
        y: column name (dependent variable of the model)
    Returns a pandas dataframe
    '''
    table = df.corr().loc[:,y]
    table = table.to_frame('correlation coefficient wrt' + y).reset_index()
    return table


def get_corr_coeff(df, var1, var2):
    '''
    Get a pairwise correlation coefficient of a pair of variables
    Inputs:
        df: panda dataframe
        var1, var2: column names
    Returns a float
    '''
    corr = df.corr()
    return corr.loc[var1, var2]


def check_missing(df):
    '''
    Identify the number of missing values for each column
    Input:
        df: pandas dataframe
    Returns a pandas dataframe of the columns and its missing count
    '''
    return df.isnull().sum().to_frame('missing count').reset_index()


###################
#  Preprocessing  #
###################

def replace_missing(df):
    pass


######################
# Feature Generation #
######################

def discretize(continous_var):
    pass

def create_dummies(categorical_var):
    pass

def standardize(var):
    pass


######################
#  Build Classifier  #
######################

def build_decision_tree(df):
    '''
    Build a decision tree classifier
    Inputs:
        df: pandas dataframe
    Returns Decision Tree Classifier object
    '''
    dt_model = tree.DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)
    return dt_model
# https://scikit-learn.org/stable/modules/tree.html#


def predict(df):
    '''
    Get predictions using the Decision Tree Model
    Inputs:
        df: pandas dataframe
    Returns an array of predicted values
    '''
    dt = build_decision_tree(df)
    x_values = df # get features to predict

    return dt.predict(x_values)


#######################
# Evaluate Classifier #
#######################

def get_accuracy(df, y_col, features):
    '''
    Get the fraction of the correctly classified instances

    Inputs:
        df: dataframe
        y_col: column name of target variable
        features: list of column names of features
    Returns a float between 0 to 1
    '''
    x_values = df.loc[:,features] #get features
    y_values = df.loc[:,y_col] #get y_values
    dt = build_decision_tree(df)
    
    return dt.score(x_values, y_values)
