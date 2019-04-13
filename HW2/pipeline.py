'''
Homework 2
Machine Learning Pipeline Functions
This python file contains general functions to read, explore,
preprocss data, generate features, build classifer, and evaluate classifier

Rachel Ker
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import tree
#from sklearn.cross_validation import train_test_split


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

# Univariate exploration

def descriptive_stats(df, continuous_var):
    '''
    Generates simple descriptive statistics e.g. count,
    mean, standard deviation, min, max, 25%, 50%, 70%
    Inputs:
        df: pandas dataframe
        var: list of continuous of interest
    '''
    return df.describe()[continuous_var]

def plot_linegraph(df, continuous_var):
    '''
    Plot linegraphs
    Inputs:
        df: pandas dataframe
        continuous_var: column name
    '''
    data = df.groupby(continuous_var).size()
    data.plot.line()
    plt.show()


def tabulate_counts(df, categorical_var):
    '''
    Generate counts for categorical variables
    Inputs:
        df: pandas dataframe
        categorical_var: column name
    Returns pandas dataframe of counts
    '''
    count= df.groupby(categorical_var).size()
    count= count.to_frame('count')
    return count


def plot_barcharts_counts(df, categorical_var):
    '''
    Plot barcharts by tabulated counts
    Inputs:
        df: pandas dataframe
        categorical_var: column name
    '''
    data = tabulate_counts(df, categorical_var)
    data.plot.bar()
    plt.show()


def boxplot(df, var=None):
    '''
    Creates a box plot
    Inputs:
        df: pandas dataframe
        var: (optional) list of column names
    Returns a matplotlib.axes.Axes
    '''
    if var:
        df.boxplot(grid=False, column=var)
    else:
        df.boxplot(grid=False)
    plt.show() 


def detect_outliers(df, var, threshold=1.5):
    '''
    Detecting outliers mathematically using interquartile range
    Inputs:
        df: pandas dataframe
        threshold: (float) indicates the threshold for
            defining an outlier e.g. if 1.5, a value outside 1.5
            times of the interquartile range is an outlier
    
    Returns a panda series indicating count of outliers
    and non-outliers for the variable
    '''
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    filtr = (df < (q1 - threshold*iqr)) | (df > (q3 + threshold*iqr))
    return filtr.groupby(var).size()
    # (Source: https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)


def check_missing(df):
    '''
    Identify the number of missing values for each column
    Input:
        df: pandas dataframe
    Returns a pandas dataframe of the columns and its missing count
    '''
    return df.isnull().sum().to_frame('missing count').reset_index()


# Bivariate Exploration and Relationships

def scatterplot(df, x, y):
    '''
    Creates a scatter plot
    Inputs:
        df: pandas dataframe
        x,y: column names
    Returns a matplotlib.axes.Axes
    '''
    df.plot.scatter(x, y)
    plt.show()


def plot_corr_heatmap(df):
    '''
    Plots heatmap of the pairwise correlation coefficient of all features
    with respect to the dependent variable
    
    Inputs:
        df: pandas dataframe
    '''
    corr_table = df.corr()
    sns.heatmap(corr_table,
                xticklabels=corr_table.columns,
                yticklabels=corr_table.columns,
                cmap=sns.diverging_palette(220, 20, as_cmap=True))
    plt.show()
    # (Source: https://towardsdatascience.com/a-guide-to-pandas-and-matplotlib-for-data-exploration-56fad95f951c)
    
    
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
    # (Source: https://scikit-learn.org/stable/modules/tree.html#)


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
