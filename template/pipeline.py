#!/usr/bin/env python
# coding: utf-8

'''
Helper Functions

Author: Rachel Ker
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score as accuracy, confusion_matrix, f1_score, auc, roc_auc_score, precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import ParameterGrid 
from sklearn.preprocessing import MinMaxScaler

from sklearn.externals.six import StringIO
import os
import csv
import graphviz
import pydotplus
import config
from aequitas.group import Group
from aequitas.plotting import Plot
from textwrap import wrap


## READ IN DATA

def get_csv(file):
    '''
    Takes in a csv file and returns a pandas dataframe

    Input:
        file: (str) filepath

    Returns pandas data frame
    '''
    df = pd.read_csv(file)
    return df


## PREPROCESSING

# Change Types

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


# Dealing with Missing Values

def check_missing(df):
    '''
    Identify the number of missing values for each column

    Input:
        df: pandas dataframe

    Returns a pandas dataframe of the columns and its missing count
    '''
    return df.isnull().sum().to_frame('missing count').reset_index()


def fill_na(df, col, fill_method = np.mean):
    '''
    Fills NA values in a df column by given method

    Input:
        df: pandas dataframe
        col: column name
        fill_method: function specifying how to fill the NA values
            e.g. np.mean, np.median, np.max, np.min 
    
    Returns pandas dataframe after imputation
    '''
    cp = df.copy()

    try:
        cp.loc[cp[col].isna(), col] = fill_method(cp[col])
    except:
        raise Exception("No such fill method")
    
    df[col] = cp[col]
    return df


def create_indicator(df, col, indicator_name='missing'):
    '''
    This function creates missing indicator feature
    
    Input:
        df: pandas dataframe
        col: column name
        indicator_name: by default, 'missing'

    Returns df with added col
    '''
    missingcol = col + '_' + indicator_name
    df[missingcol] = [1 if x else 0 for x in df[col].isna()]
    return df


def impute_missing_cat(df, col, fill_cat = 'MISSING'):
    '''
    This function creates a missing binary column and imputes fill_cat

    Input:
        df: dataframe
        fill_cat: impute value

    Returns dataframe with imputed data
    '''
    cp = df.copy()
    cp.loc[cp[col].isna(), col] = fill_cat
    df[col] = cp[col]
    return df


# Creating Dummies

def categorical_to_dummy(df, categorical_var):
    '''
    Creates dummy variables from categorical var
    
    Inputs:
        df: pandas dataframe
        categorical_var: column name
    
    Returns df with dummy variables added
    '''
    dummy = pd.get_dummies(df[categorical_var], prefix=categorical_var)
    return df.join(dummy)


def categorical_to_dummy_with_groupconcat(df, var):
    '''
    Convert categorical (group concat) variables to dummies

    Input:
        df: pandas dataframe
        var: categorical var column name
    
    Returns df with added dummies
    '''
    dummy = df[var].str.get_dummies(sep=',')
    for d in dummy.columns:
        col_name = "{}_{}".format(var, d)
        df[col_name] = dummy[d]
    return df 


# Discretize Variables

def discretize(df, continuous_var, n, labels, showbins=True):
    '''
    Discretize continuous variable
    Inputs:
        df: pandas dataframe
        continuous_var: column name
        n: number of equal bins / list of array inclusive of upperbound
        labels: list of labels
        showbins: return bins (Boolean, default True)

    Returns df, bins if bins=True
    Returns df if bins=False
    '''
    cat, bin_ = pd.cut(df[continuous_var], bins=n, labels=labels, retbins=True)
    df[continuous_var] = cat
    
    if showbins:
        return df, bin_
    else:
        return df


## TEMPORAL VALIDATION

##TODO: change accordingly

def to_timestamp(time):
    '''
    Convert time to timestamp

    Input:
        time: tuple of (year, month, date)
    
    Returns timestamp
    '''
    yy, mm, dd = time
    return pd.Timestamp(yy, mm, dd)


def temporal_split(df, date_col, train_start_date, test_start_date, 
                   test_end_date, time_period):
    '''
    Splits dataset by time
    Inputs:
        df: pandas dataframe
        date_col: column of dates
        train_start_date, test_start_date, test_end_date: tuple of (year, month, date) inclusive
        time_period: (str) prediction period in days e.g. '60 days'
    Returns x_train, x_test, y_train, y_test
    '''
    train_start = to_timestamp(train_start_date)
    test_start = to_timestamp(test_start_date)
    test_end = to_timestamp(test_end_date)

    train = df[(df[date_col] >= train_start) & (df[date_col] < test_start - pd.Timedelta(time_period))]
    test = df[(df[date_col] >= test_start) & (df[date_col] <= test_end)]
        
    return train, test


## EVALUATION METRICS

def accuracy_at_threshold(y_true, y_predicted):
    '''
    Calculates accuracy given the arrays of true y and predicted y
    
    Input:
        y_true: np.array with the observed Ys 
        y_predicted: np.array with the predicted Ys 
    
    Returns accuracy
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    return 1.0 * (tp + tn) / (tn + fp + fn + tp )


def precision_at_threshold(y_true, y_predicted):
    '''
    Calculates precision given the arrays of true y and predicted y
    
    Input:
        y_true: np.array with the observed Ys 
        y_predicted: np.array with the predicted Ys 
    
    Returns precision
    '''
    _, fp, _, tp = confusion_matrix(y_true, y_predicted).ravel()
    return 1.0 * tp / (tp + fp)


def recall_at_threshold(y_true, y_predicted):
    '''
    Calculates recall given the arrays of true y and predicted y
    
    Input:
        y_true: np.array with the observed Ys 
        y_predicted: np.array with the predicted Ys 
    
    Returns recall
    '''
    _, _, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    return 1.0 * tp / (tp + fn)


def f1_at_threshold(y_true, y_predicted):
    '''
    Calculates F1 score given the arrays of true y and predicted y
    
    Input:
        y_true: np.array with the observed Ys 
        y_predicted: np.array with the predicted Ys 
    
    Returns F1 score
    '''
    return  f1_score(y_true, y_predicted)


def scores_pctpop(pred_scores, threshold):
    '''
    Identifies the units to be given 1 and 0 based on the predicted scores
    and threshold
    
    Inputs:
        pred_scores: array of predicted scores
        threshold: percentage of population to be identified

    Returns an array of 1s and 0s
    '''
    #identify number of positives to have given target percent of population
    num_pos = int(round(len(pred_scores)*(threshold/100),0))
    tmp = pred_scores.copy()
    pred_df = pd.Series(tmp)
    idx = pred_df.sort_values(ascending=False)[0:num_pos].index 
    
    #set all observations to 0
    pred_df.iloc[:] = 0
    #set observations by index (the ones ranked high enough) to 1
    pred_df.iloc[idx] = 1
    
    return list(pred_df)


## GRAPHS - Score histogram, PRC, Decision Tree, Bias Graphs

def plot_scores_hist(df,col_score, model_name, graph_dir, save=True):
    '''
    Plots histogram of scores.

    Inputs:
        df: dataframe
        col_score: column with scores
        model_name: name of model
        save: boolean
    '''
    plt.clf()
    df[col_score].hist()
    plt.title(model_name)
    if save:
        f = os.path.join(graph_dir, model_name)
        plt.savefig(f)


def plot_precision_recall_n(precision, recall, pct_pop, model_name, text, graph_dir, save):
    '''
    Plot precision and recall for each threshold
    
    Inputs:
        y_true: real labels for testing set
        y_prob: array of predicted scores from model
        model_name: (str) title for plot
        text: additional text for the graph
        save: boolean
    '''
    plt.clf()
    fig, ax1 = plt.subplots()

    #plot precision
    color = 'blue'
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color=color)
    ax1.plot(pct_pop, precision, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.yticks(np.arange(0, 1.2, step=0.2))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #plot recall
    color = 'orange'
    ax2.set_ylabel('recall', color=color)  # we already handled the x-label with ax1
    ax2.plot(pct_pop, recall, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.yticks(np.arange(0, 1.2, step=0.2))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(top=0.85)
    
    # plot vertical line for threshold
    ax1.axvline(x=10, ymin=0, ymax=1, color = 'gray')
    # set titles
    ax1.set_title("\n".join(wrap(model_name, 60)))
    plt.text(55, 0.4, text)

    if save:
        pltfile = os.path.join(graph_dir,model_name)
        plt.savefig(pltfile)


# Visualize DT
def visualize_tree(dt, feature_labels, class_labels, filename, graph_dir, save):
    '''
    Visualization of the decision tree

    Inputs:
        dt_model: DecisionTreeClassifier object
        feature_labels: a list of labels for features
        class_labels: a list of labels for target class
        filename
        save: True or False
    
    Saves a png
    '''
    dot_data = StringIO()
    graphviz.Source(tree.export_graphviz(dt, out_file=dot_data,
                                         feature_names=feature_labels,
                                         class_names=class_labels,
                                         filled=True))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    if save:  
        f = os.path.join(graph_dir,filename)
        graph.write_png(f)
    

def plot_bias(model_name, bias_df, bias_metrics = ['ppr','pprev','fnr','fpr', 'for'], 
              min_group_size=None, graph_dir=config.GRAPH_FOLDER,save=True):
    '''
    Creates bar charts for bias metrics given.

    Inputs:
        model_name: (str) filename
        bias_df: pandas df
        bias_metrics: list of metrics we care about
        min_group_size: integer
        save: boolean
    '''
    g = Group()
    xtab, _ = g.get_crosstabs(bias_df)
    aqp = Plot()
    n = len(bias_metrics)
    p = aqp.plot_group_metric_all(xtab, metrics=bias_metrics, ncols=n, min_group_size=min_group_size)
    if save:
        pltfile = os.path.join(graph_dir,model_name)
        p.savefig(pltfile)


## RESULTS - Feature list txt, Feature Importance, Predictions, Results

# Get Feature Importance
def get_feature_importance(clfr, features, filename, results_dir, save):
    '''
    Get the feature importance of each feature

    Inputs:
        clfr: classifier
        features: list of features
        filename
        save: True or False
    
    Return a dataframe of feature importance
    '''
    d = {'Features': features}
    if (isinstance(clfr, LogisticRegression) or
        isinstance(clfr, LinearSVC) or 
        isinstance(clfr, MultinomialNB)):
        # Logistic Regression, SVM, Naive Bayes
        d['Importance'] = clfr.coef_[0]
    else:
        try:
        # Works for Random forest, Decision trees, Extra trees,
        # Gradient boosting, Adaboost , XGBoost
            d['Importance'] = clfr.feature_importances_
        except:
        # K-nearest neighbors, bagging
            print('feature importance not found for {}'.format(clfr))
            return
    
    feature_importance = pd.DataFrame(data=d)
    feature_importance = feature_importance.sort_values(by=['Importance'],
                                                        ascending=False)
    if save:
        f = os.path.join(results_dir,filename) 
        feature_importance.to_csv(f)

    return feature_importance


metrics = { 'accuracy': accuracy_at_threshold,
            'precision': precision_at_threshold,
            'recall': recall_at_threshold,
            'f1': f1_at_threshold,
            'auc': roc_auc_score}

classifiers = { 'RF': RandomForestClassifier(n_jobs=-1, random_state=config.SEED),
                'ET': ExtraTreesClassifier(n_jobs=-1, criterion='entropy', random_state=config.SEED),
                'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=config.SEED),
                'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6,
                                                n_estimators=10, random_state=config.SEED),
                'KNN': KNeighborsClassifier(n_neighbors=3),
                'DT': DecisionTreeClassifier(max_depth=5, random_state=config.SEED),
                'SVM': LinearSVC(random_state=config.SEED),
                'LR': LogisticRegression(penalty='l1', C=1e5, random_state=config.SEED),
                'BAG': BaggingClassifier(random_state=config.SEED),
                'NB': MultinomialNB(alpha=1.0),
                'XG': XGBClassifier()
        }

def classify(train_set, test_set, label, models, eval_metrics, eval_metrics_by_level, custom_grid, 
             attributes_lst, bias_lst, bias_dict, train_test_num, variables, results_dir, results_file, graph_dir,  
             threshold, plot_pr, compute_bias, visualize_dt, ft_impt, score_hist, save_pred):
    '''
    Builds classifier and writes out results in csv
    
    Input:
        train_set, test_set: dataframe for training and testing the models
        label: name of the Y variable
        models: classifier models to fit
        eval_metrics: list of threshold-independent metrics.
        eval_metrics_by_level: tuple containing a list of threshold-dependent metrics as first element and 
            a list of thresholds as second element
        custom_grid: grid of parameters
        attributes_lst: list containing the names of the features (i.e. X variables) to be used.
        bias_lst: list of column names for bias 
        bias_dict: dictionary of metrics for bias computation
        train_test_num: train test sets
        variables: variable dictionary we care about
        results_dir/results_file/graph_dir
        threshold: population threshold we care about
        plot_pr/compute_bias/visualize_dt/ft_impt/score_hist/save_pred: boolean whether to save or not
    
    Returns None
    '''
    #initialize results
    results_columns = (['train_test_num', 'model', 'classifiers', 'parameters', 'train_set_size', 
                        'num_features', 'validation_set_size', 'baseline'] + eval_metrics + 
                      [metric + '_' + str(level) for level in eval_metrics_by_level[1] for metric in eval_metrics_by_level[0]])
    
    # Write header for the csv
    outfile = os.path.join(results_dir, "{}_{}.csv".format(results_file, train_test_num))
    with open(outfile, "w") as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(results_columns)

    # subset training and test sets 
    y_train = train_set[label]
    y_test = test_set[label]

    n_target = sum(test_set[label])
    n_observations = len(test_set[label])
    baseline = n_target/n_observations
        
    features_txt = os.path.join(results_dir, "features_{}.txt".format(train_test_num))
    if not os.path.exists(features_txt):
        with open(features_txt, "w") as f:
            print(attributes_lst, file=f)
    print('features_{}.txt file created'.format(train_test_num))


    # iterate through models
    for model in models:
        X_train = train_set.loc[:, attributes_lst]
        X_test = test_set.loc[:, attributes_lst]
        
        # Scaling continuous variable if not DT
        cont = variables['CONTINUOUS_VARS_MINMAX']
        scaler = MinMaxScaler()
        data_for_fitting = X_train[cont]
        scaler.fit(data_for_fitting)
        
        if model != 'DT':
            print('Scaling data for {}'.format(model))
            X_train[cont] = scaler.transform(X_train[cont])
            X_test[cont] = scaler.transform(X_test[cont])            

        #iterate through parameters grids
        grid = ParameterGrid(custom_grid[model])

        for parameters in grid:
            classifier = classifiers[model]
            print('Running model: {}, param: {}'.format(model, parameters))
            clfr = classifier.set_params(**parameters)
            clfr.fit(X_train, y_train)

            # visualize decision tree
            if isinstance(clfr, DecisionTreeClassifier):
                filename = '{}_{}_{}.png'.format(train_test_num, model, str(parameters).replace(':','-'))
                visualize_tree(clfr, attributes_lst, ['No','Yes'], filename, graph_dir, visualize_dt)
            
            # Get feature importance
            filename = 'FIMPORTANCE_{}_{}_{}.csv'.format(train_test_num, model, str(parameters).replace(':','-'))
            get_feature_importance(clfr, attributes_lst, filename, results_dir, ft_impt)

            # calculate scores
            if isinstance(clfr, LinearSVC):
                y_pred_prob = clfr.decision_function(X_test)
            else:    
                y_pred_prob = clfr.predict_proba(X_test)[:,1]
            test_set['SCORE'] = y_pred_prob
            # plot and save score distributions
            model_name = 'HIST_{}_{}_{}.png'.format(train_test_num, model, str(parameters).replace(':','-'))
            plot_scores_hist(test_set, 'SCORE', model_name, graph_dir, score_hist)

            # Calculate and save predictions
            test_set['PREDICTION'] = scores_pctpop(y_pred_prob, threshold)
            if save_pred:
                filename = 'PRED_{}_{}_{}.csv'.format(train_test_num, model, str(parameters).replace(':','-'))
                f = os.path.join(results_dir, filename)
                final_pred = test_set.loc[:, ['ID', 'PREFIX', 'START_DATE', 'END_DATE', 'LABEL', 'SCORE','PREDICTION']]
                final_pred.to_csv(f, index=False)
            
            # plot bias metrics if desired
            if compute_bias:
                tmp = test_set.copy()
                tmp['id'] = tmp['ID']
                tmp['score'] = tmp['PREDICTION']
                tmp['label'] = tmp[label]
                bias_df = tmp.loc[:, bias_lst]
                # plot and save bias
                model_name = 'BIAS_{}_{}_{}.png'.format(train_test_num, model, str(parameters).replace(':','-'))
                plot_bias(model_name, bias_df, bias_dict['metrics'], 
                          bias_dict['min_group_size'], graph_dir, compute_bias)

            eval_result = [train_test_num, model, classifier, parameters, len(X_train), len(attributes_lst), len(X_test), baseline]

            # evaluate metrics
            if eval_metrics:
                eval_result += [metrics[metric](y_test, y_pred_prob) for metric in eval_metrics]
            
            if eval_metrics_by_level[0]:
                precision = []
                recall = []
                for level in eval_metrics_by_level[1]:
                    y_pred = scores_pctpop(y_pred_prob, level)
                    for metric in eval_metrics_by_level[0]:
                        score = metrics[metric](y_test, y_pred)
                        if metric == 'precision':
                            precision.append(score)
                        elif metric == 'recall':
                            recall.append(score)
                            if level == threshold:
                                rec = score
                        eval_result += [score]
            
            # plot precision and recall if desired
            if plot_pr:    
                model_name = 'PRC_{}_{}_{}.png'.format(train_test_num, model, str(parameters).replace(':','-'))
                print('plotting precision recall for {}'.format(model_name))
                text = 'Recall at {} is {}'.format(threshold, round(rec,3))
                # add last point on curve at 100%
                y_pred = scores_pctpop(y_pred_prob,100)
                thresholds = eval_metrics_by_level[1] + [100]
                precision.append(precision_at_threshold(y_test, y_pred))
                recall.append(recall_at_threshold(y_test, y_pred))
                # plot graph
                plot_precision_recall_n(precision, recall, thresholds, model_name, text, graph_dir, plot_pr)

            # writing out results in csv file
            with open(outfile, "a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(eval_result)


## Understanding best models

def get_best_models(df, time_col, test_years, cols, metric):
    '''
    Identify best models for the specified metric
    Inputs:
        df: pandas dataframe of results
        time_col: (str) name of the col that identifies different traintest sets 
        test_years: list of years in results
        cols: list of cols for the table
        metric: (str) precision, recall, accuracy, f1_score, auc
    
    Return dataframe of best model for the specified metric
    '''
    best_models = pd.DataFrame(columns= cols)

    for year in test_years:
        year_data = df[df[time_col]==year]
        highest = year_data[metric].max()
        model = year_data[year_data[metric] == highest]
        print("For train-test set {}, highest {} attained is {}".format(year, metric, highest))
        best_models = best_models.append(model[cols])

    return best_models

def sort_models(data, metric, top_k, cols):
    '''
    Get top k models
    '''
    sort = data.sort_values(metric, ascending=False)
    results = sort[cols][:top_k]
    return results


def get_stability_score(trainsets, metric, cols):
    '''
    Identify models that are top ranking in all train test sets
    Inputs:
        trainsets: list of dataframes that correspond to each traintest set
    '''
    result = pd.DataFrame()
    for sets in trainsets:
        sort = sets.sort_values(metric, ascending=False)
        sort['rank'] = range(len(sort))
        result = result.append(sort)
    result = result[cols + ['rank']]
    return result.groupby(['classifiers','parameters']).mean().sort_values('rank')


def get_metric_graph(df, metric, model_and_para, baseline, train_test_col, 
                     train_test_val, title, filename, save=False):
    '''
    Inputs:
        df: pandas dataframe of results
        metric: str e.g. 'precision_at_5'
        model_and_para: list of tuples containing models and paras in str 
            e.g. [('model', 'parameters'), ('model','parameters')]
        train_test_col: column name for train test sets (str)
        train_test_val: list of values in train test sets
        baseline: list of baselines over the train test sets
        title: title of graph
    '''
    def get_data(df, dic, model, para, metric, train_test_col, train_test_val):
        '''
        Getting the data points to plot
        '''
        col = []
        for yr in train_test_val:
            trainset = df[df[train_test_col]==yr]
            temp = trainset[trainset['parameters']==para][[metric]]
            col.extend(temp[metric].values)
        dic[model + ' ' + para] = col
        return dic
    
    def plot_graph(df, metric, train_test_val, title, filename, save=False):
        '''
        Plot metric over different traintest sets
        '''
        df.plot.line()
        plt.title(title)
        plt.ylabel(metric)
        tick = list(range(len(train_test_val)))
        plt.xticks(tick, train_test_val)
        plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
        if save:
            plt.savefig(filename)
        plt.show()

    full_dict = {}

    for m in model_and_para:
        model, para = m
        dic = get_data(df, full_dict, model, para, metric, train_test_col, train_test_val)
        full_dict.update(dic)
    full_dict['baseline'] = baseline
    new_df = pd.DataFrame(full_dict)
    
    plot_graph(new_df, metric, train_test_val, title, filename, save)

