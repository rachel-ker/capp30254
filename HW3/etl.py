'''
ETL - Explore, Transform and Load Code

Rachel Ker
'''
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


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

def scatterplot(df, x, y, y_col):
    '''
    Creates a scatter plot
    Inputs:
        df: pandas dataframe
        x,y: column names
        y_col: (str) column name of target variable
    Returns a matplotlib.axes.Axes
    '''
    df.plot.scatter(x, y, c=y_col, colormap='PiYG')
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

def replace_dates_with_datetime(df, date_cols):
    '''
    Replace date columns with datetime
    Inputs:
        df: dataframe
        date_cols: list of date columns
    Returns dataframe with datetime columns
    '''
    df[date_cols] = df[date_cols].apply(pd.to_datetime)
    return df


def replace_missing_value(df, col, val):
    '''
    Replace null values with val specified
    '''
    values = {col: val}
    df.fillna(value=values, inplace=True)
    return df



def replace_missing_with_mode(df, cols):
    '''
    Replaces null values in dataframe with the mean of the col
    Inputs:
        df: pandas dataframe
        cols: list of col
    Returns a pandas dataframe with missing values replaced
    '''
    values = {}
    for col in cols:
        values[col] = df[col].mode().loc[0]
    df.fillna(value=values, inplace=True)
    return df


def replace_missing_with_mean(df, cols):
    '''
    Replaces null values in dataframe with the mean of the col
    Inputs:
        df: pandas dataframe
        cols: list of col
    Returns a pandas dataframe with missing values replaced
    '''
    values = {}
    for col in cols:
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
    Returns a new dataframe with dummy variables added
    '''
    dummy = pd.get_dummies(df[categorical_var],
                           prefix=categorical_var,
                           drop_first=True)
    return df.join(dummy)
