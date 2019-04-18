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

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

from IPython.display import SVG
from IPython.display import display


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

def replace_missing_with_mean(df):
    '''
    Replaces null values in dataframe with the mean of the col
    Inputs: pandas dataframe
    Returns a pandas dataframe with missing values replaced
    '''
    values = {}
    for col in df.columns:
        values[col] = df[col].mean()
    df.fillna(value=values, inplace=True)
    return df


######################
# Feature Generation #
######################

def discretize(df, continuous_var, lower_bounds):
    '''
    Discretize continuous variable by creating new var in df
    Inputs:
        df: pandas dataframe
        continuous_var: column name
        bounds: list of lowerbound inclusive for discretization

    New discretized variable added to dataframe
    Returns df
    '''
    min_val = df[continuous_var].min()
    assert lower_bounds[0] == min_val
    max_val = df[continuous_var].max()

    lower_bounds = lower_bounds + [max_val+1]

    replace_dict = {}
    
    for i in range(len(lower_bounds)-1):
        key = str(lower_bounds[i]) + "_to_" + str(lower_bounds[i+1])
        replace_dict[key] = lower_bounds[i]

    df[continuous_var + "_discrete"] = pd.cut(df[continuous_var],
                                              right=False,
                                              bins=list(replace_dict.values()) + [max_val],
                                              labels=list(replace_dict.keys()),
                                              include_lowest=True)
    return df
        

def create_dummies(df, categorical_var):
    '''
    Creates dummy variables from categorical var
    Inputs:
        df: pandas dataframe
        categorical: column name
    Drops the categorical column
    Returns a new dataframe with dummy variables added
    '''
    dummy = pd.get_dummies(df[categorical_var], prefix=categorical_var)
    return df.join(dummy)


######################
#  Build Classifier  #
######################

def select_features(df, y_col, features):
    '''
    Keep only the relevant columns for model building
    Inputs:
        df: pandas dataframe
        y_col: col name for target variable
        features: list of features
    Return a data frame that only contains these relevant col
    '''
    return df.loc[:, features + [y_col]]


def split_training_testing(df, y_col, test_size):
    '''
    Splits the dataset into training and testing sets
    Inputs:
        df: pandas dataframe
        y_col: col name for target variable
        test_size: (between 0 and 1) percentage of data to use for testing set
    Returns x_train, x_test, y_train, y_test
    '''
    x = df.loc[:,df.columns != y_col]
    y = df.loc[:,y_col]
    return train_test_split(x, y, test_size=test_size, random_state=100)
    

def build_decision_tree(x, y, max_depth, min_leaf):
    '''
    Build a decision tree classifier
    Inputs:
        x, y: training sets for features and labels
        max_depth: (int) max depth of decision tree
        min_leaf: (int) min sample in the leaf of decision tree
        
    Returns Decision Tree Classifier object
    '''
    dt_model = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=max_depth, min_samples_leaf=min_leaf)
    dt_model.fit(x, y)
    return dt_model
    # (Source: https://scikit-learn.org/stable/modules/tree.html#)



def visualize_tree(dt, feature_labels, class_labels, file=None):
    '''
    Visualization of the decision tree
    Inputs:
        dt_model: DecisionTreeClassifier object
        feature_labels: a list of labels for features
        class_labels: a list of labels for target class
        file: (optional) filepath for visualization
    Returns a graphviz objects
    '''
    graph = graphviz.Source(tree.export_graphviz(dt, out_file=file,
                                                 feature_names=feature_labels,
                                                 class_names=class_labels,
                                                 filled=True))
    return graph
    # (Source: https://towardsdatascience.com/interactive-visualization-of-decision-trees-with-jupyter-widgets-ca15dd312084)


def get_labels(df, y_col):
    '''
    Get feature labels
    Inputs:
        df: pandas dataframe
        y_col: (str) column name of target variable
    Return a list of feature labels
    '''
    return df.loc[:,df.columns != y_col].columns


#######################
# Evaluate Classifier #
#######################

def predict_prob(dt, x_test):
    '''
    Get predicted probabilities from the model

    Inputs:
        dt: decision tree model
        x_test: testing set for the features
    Returns an array of predicted probability
    '''
    return dt.predict_proba(x_test)[:,1]


def accuracy_prediction(dt, x_test, y_test, threshold):
    '''
    Builds decision tree and gets the prediction accuracy from
    the model according to specified threshold

    Inputs:
        dt: decision tree model
        x_test, y_test: testing sets from the data
        threshold: specified probability threshold to
            classify obs as positive
    Returns a float between 0 and 1
    '''
    pred_prob = predict_prob(dt, x_test)
    calc_threshold = lambda x,y: 0 if x < y else 1 
    predicted_test = np.array( [calc_threshold(score, threshold) for score in pred_prob] )                    
    return accuracy_score(y_test, predicted_test)


def build_and_test_classifier(df, y_col, test_size, max_depth, min_leaf):
    '''
    Split test and train sets, build and visualize classifier and returns
    accuracy score
    
    Inputs:
        df: pandas dataframe with selected features
        y_col: (str) column name of target variable
        test_size: (between 0 and 1) percentage of data to use for testing set
        max_depth: (int) max depth of decision tree
        min_leaf: (int) min sample in the leaf of decision tree
        
    Returns accuracy score between 0 and 1
    '''
    x_train, x_test, y_train, y_test = split_training_testing(df, y_col, test_size)
    dt = build_decision_tree(x_train, y_train, max_depth=max_depth, min_leaf=min_leaf)
    graph = visualize_tree(dt, get_labels(df, y_col), ['No', 'Yes'])
    display(SVG(graph.pipe(format='svg')))    
    return accuracy_prediction(dt, x_test, y_test, 0.5)
