'''
Homework 2
Applying Machine Learning to Credit Data 

Rachel Ker
'''
import pipeline as pl


def hw2():
    '''
    Putting the code together for assignment 2
    (Reading in Data, Data Exploration, Pre-processing,
    Feature Generation, Build Classifier and Evaluate)
    
    Returns accuracy score of classifier built
    '''
    data = pl.read_csvfile("data/credit-data.csv")
    univariate_analysis(data)
    multivariate_analysis(data)
    
    data = replace_missing(data)
    data = feature_generation(data)
    data_features = feature_selection(data)
    accuracy = pl.build_and_test_decision_tree(data_features, 'SeriousDlqin2yrs', 0.2, max_depth=4, min_leaf=100)
    return accuracy


def univariate_analysis(data):
    '''
    Plots and Tables for univariable data explorations
    Inputs:
        data: pandas dataframe
    Returns None
    '''
    pl.plot_barcharts_counts(data, 'SeriousDlqin2yrs')
    pl.plot_linegraph(data, 'age')
    table = pl.descriptive_stats(data, ['age'])
    print(table)
    
    pl.boxplot(data, 'age')
    pl.detect_outliers(data, 'age')
    
    pl.plot_barcharts_counts(data, 'NumberOfTime30-59DaysPastDueNotWorse')
    table = pl.tabulate_counts(data,'NumberOfTime30-59DaysPastDueNotWorse')
    print(table)
    pl.boxplot(data, 'NumberOfTime30-59DaysPastDueNotWorse')
    pl.detect_outliers(data, 'NumberOfTime30-59DaysPastDueNotWorse')
    
    pl.plot_barcharts_counts(data, 'NumberOfTime60-89DaysPastDueNotWorse')
    table = pl.tabulate_counts(data,'NumberOfTime60-89DaysPastDueNotWorse')
    print(table)
    pl.boxplot(data, 'NumberOfTime60-89DaysPastDueNotWorse')
    pl.detect_outliers(data, 'NumberOfTime60-89DaysPastDueNotWorse')
    
    pl.plot_barcharts_counts(data,  'NumberOfTimes90DaysLate')
    table = pl.tabulate_counts(data,'NumberOfTimes90DaysLate')
    print(table)
    pl.boxplot(data, 'NumberOfTimes90DaysLate')
    pl.detect_outliers(data, 'NumberOfTimes90DaysLate')
    
    pl.plot_barcharts_counts(data, 'NumberOfOpenCreditLinesAndLoans')
    table = pl.descriptive_stats(data, ['NumberOfOpenCreditLinesAndLoans'])
    print(table)
    pl.boxplot(data, 'NumberOfOpenCreditLinesAndLoans')
    pl.detect_outliers(data, 'NumberOfOpenCreditLinesAndLoans')
    
    pl.plot_barcharts_counts(data, 'NumberRealEstateLoansOrLines')
    table = pl.descriptive_stats(data, ['NumberRealEstateLoansOrLines'])
    print(table)
    pl.boxplot(data, 'NumberRealEstateLoansOrLines')
    pl.detect_outliers(data, 'NumberRealEstateLoansOrLines')
    
    pl.plot_barcharts_counts(data, 'NumberOfDependents')
    pl.boxplot(data, 'NumberOfDependents')
    pl.detect_outliers(data, 'NumberOfDependents')
    
    pl.descriptive_stats(data, ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome'])


def multivariate_analysis(data):
    '''
    Correlational Heatmap and scatterplots for multivariate data exploration
    Inputs:
        data: pandas dataframe
    Returns None
    '''
    pl.plot_corr_heatmap(data)

    pl.scatterplot(data, 'age', 'SeriousDlqin2yrs')
    pl.scatterplot(data, 'DebtRatio', 'SeriousDlqin2yrs')
    pl.scatterplot(data, 'MonthlyIncome', 'SeriousDlqin2yrs')
    pl.scatterplot(data, 'NumberOfOpenCreditLinesAndLoans', 'SeriousDlqin2yrs')
    pl.scatterplot(data, 'NumberRealEstateLoansOrLines', 'SeriousDlqin2yrs')


def replace_missing(data):
    '''
    Replace missing variables with mean and
    check number of missing variables
    Inputs:
        data: pandas dataframe
    Returns dataframe with missing replaced with mean
    '''
    data = pl.replace_missing_with_mean(data)
    print(pl.check_missing(data))
    return data


def feature_generation(data):
    '''
    Discretize continuous variables
    and create dummies for categorical variables
    Inputs:
        data: pandas dataframe
    Returns data with added variables
    '''
    print("Discretization")
    data = pl.discretize(data, 'RevolvingUtilizationOfUnsecuredLines', [0,1])
    data = pl.discretize(data, 'DebtRatio', [0,1])
    print(data.columns)
    print()

    print("Create Dummies")
    data = pl.create_dummies(data, 'RevolvingUtilizationOfUnsecuredLines_discrete')
    data = pl.create_dummies(data, 'DebtRatio_discrete')
    data = pl.create_dummies(data, 'zipcode')
    print(data.columns)
    print()
    return data


def feature_selection(data):
    '''
    Selects only relevant features for model building
    Inputs:
        data: pandas dataframe
    Returns dataframe with only selected features
    '''
    data_features = pl.select_features(data, 'SeriousDlqin2yrs', ['age', 
                                  'NumberOfTime30-59DaysPastDueNotWorse',
                                  'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                                  'NumberRealEstateLoansOrLines', 'NumberOfTimes90DaysLate',
                                  'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
                                  'RevolvingUtilizationOfUnsecuredLines_discrete_1_to_22001.0',
                                  'DebtRatio_discrete_1_to_106886.0', 'zipcode_60618', 
                                  'zipcode_60625', 'zipcode_60629', 'zipcode_60637', 'zipcode_60644'])
    return data_features

