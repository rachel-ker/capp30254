'''
Machine Learning Pipeline Functions
This python file contains general functions to do train-test splits,
get predictions, evaluate classifier

etl.py contains functions to read, explore, clean data

Rachel Ker
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import csv

import config
import etl

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier)

SEED = 0

###########################
#  Train and Test Splits  #
###########################

def split_training_testing(df, y_col, features, test_size):
    '''
    Splits the dataset into training and testing sets randomly
    Inputs:
        df: pandas dataframe
        y_col: col name for target variable
        features: list of features
        test_size: (between 0 and 1) percentage of data to use for testing set
    Returns x_train, x_test, y_train, y_test
    '''
    x = df.loc[:, features]
    y = df.loc[:,y_col]
    return train_test_split(x, y, test_size=test_size, random_state=SEED)


def temporal_split(df, y_col, features, date_col,
                   train_start_date, test_start_date, 
                   time_period, test_period):
    '''
    Splits dataset by time
    Inputs:
        df: pandas dataframe
        y_col: col name for target variable
        features: list of features
        train_start_date: tuple of (year, month, date)
        test_start_date: tuple of (year, month, date)
        time_period: (str) prediction period in days e.g. '60 days'
        test_period: (str) size of test set
    Returns x_train, x_test, y_train, y_test
    '''
    train_start_yy, train_start_mm, train_start_dd = train_start_date
    test_start_yy, test_start_mm, test_start_dd = test_start_date

    train_start = pd.Timestamp(train_start_yy, train_start_mm, train_start_dd)
    test_start = pd.Timestamp(test_start_yy, test_start_mm, test_start_dd)

    train = df[(df[date_col] >= train_start) 
               & (df[date_col] <= test_start - pd.Timedelta(time_period))]
    test = df[(df[date_col] >= test_start) 
               & (df[date_col] <= test_start + pd.Timedelta(test_period))]

    x_train = train.loc[:, features]
    x_test = test.loc[:, features]
    y_train = train.loc[:, y_col]
    y_test = test.loc[:, y_col]
        
    return x_train, x_test, y_train, y_test



#######################
#   Get Prediction    #
#######################


def get_predicted_scores(model, x_test, svm):
    '''
    Get prediction scores according to specified threshold

    Inputs:
        model: classifier object
        x_test: testing set features
        svm: If this is an svm model, set to True. else False.
    Returns an array of predicted scores
    '''
    if svm:
        scores = model.decision_function(x_test)
    else:
        scores = model.predict_proba(x_test)[:,1]
    return scores


def get_score_distribution(pred_score):
    '''
    Plot predicted score distributions
    '''
    sns.distplot(pred_score, kde=False, rug=False)
    plt.show()



#######################
# Evaluate Classifier #
#######################

# General evaluation metrics

def joint_sort_descending(l1, l2):
    '''
    Sort arrays descending 
    Inputs:
        l1 and l2 - numpy arrays
    Returns sorted arrays
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    '''
    Generate binary predictions at k percentage threshold
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def get_accuracy_score(y_test, predicted_score, threshold):
    '''
    Get accuracy score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        threshold: specified % threshold to classify obs as positive
    Returns accuracy score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(predicted_score), np.array(y_test))
    preds_at_k = generate_binary_at_k(y_scores_sorted, threshold)
    return sklearn.metrics.accuracy_score(y_true_sorted, preds_at_k)


def get_precision(y_test, predicted_score, threshold):
    '''
    Get precision score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        threshold: specified % threshold to classify obs as positive    
    Returns precision score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(predicted_score), np.array(y_test))
    preds_at_k = generate_binary_at_k(y_scores_sorted, threshold)
    precision = sklearn.metrics.precision_score(y_true_sorted, preds_at_k)
    return precision
    

def get_recall(y_test, predicted_score, threshold):
    '''
    Get recall score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        threshold: specified % threshold to classify obs as positive        
    Returns recall score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(predicted_score), np.array(y_test))
    preds_at_k = generate_binary_at_k(y_scores_sorted, threshold)
    recall = sklearn.metrics.recall_score(y_true_sorted, preds_at_k)
    return recall


def get_f1(y_test, predicted_score, threshold):
    '''
    Get f1 score score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        threshold: specified % threshold to classify obs as positive        
    Returns f1 score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(predicted_score), np.array(y_test))
    preds_at_k = generate_binary_at_k(y_scores_sorted, threshold)
    return sklearn.metrics.f1_score(y_true_sorted, preds_at_k)


def get_auc(y_test, predicted_score):
    '''
    Get area under the curve score score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_score: predicted score from predict proba/decision fn
        
    Returns area under the ROC
    '''
    return sklearn.metrics.roc_auc_score(y_test, predicted_score)


def plot_roc_curve(y_test, predicted_scores, labels):
    '''
    Plot the ROC curves
    Inputs:
        y_test: testing set
        predicted_scores: list of predicted score from predict proba/decision fn
        labels: (list of str) title for plots
    '''
    plt.figure()

    for scores in predicted_scores:
        for label in labels:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, scores)
            plt.plot(fpr, tpr, label=label + ' (area = {:.2f})'.format(get_auc(y_test, scores)))

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('graphs/ROC curves')
    plt.show()
    # (Source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)

 
def plot_precision_recall_curve(y_test, predicted_scores, model_names):
    '''
    Plot precision and recall curves for the predicted labels
    
    Inputs:
        y_test: testing set
        predicted_scores: list of predicted score from predict proba/decision fn
        model_names: (list of str) title for plots
    '''
    for scores in predicted_scores:
        for m_name in model_names:
            precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, scores)
            plt.plot(recall, precision, marker='.', label=m_name)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('graphs/precision recall curve')
    plt.show()


def plot_precision_recall_n(y_true, y_prob, model_name, output_type):
    '''
    Plot precision and recall for each threshold

    Inputs:
        y_true: real labels for testing set
        y_prob: array of predicted scores from model
        model_name: (str) title for plot
        output_type: (str) save or show
    '''
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]

    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)

    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])

    plt.title(model_name)

    if (output_type == 'save'):
        plt.savefig("graphs/" + model_name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()
    # (Source: https://github.com/dssg/hitchhikers-guide/blob/master/
    # sources/curriculum/3_modeling_and_machine_learning/machine-learning/machine_learning_clean.ipynb)


# Getting Best models #

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


#######################
#   Building Models   #
#######################


eval_metric_dic = { 'accuracy': get_accuracy_score,
                    'precision': get_precision,
                    'recall': get_recall,
                    'f1': get_f1,
                    'auc': get_auc }


clfs = {'RF': RandomForestClassifier(n_jobs=-1, random_state=SEED),
        'ET': ExtraTreesClassifier(n_jobs=-1, criterion='entropy', random_state=SEED),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=SEED),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6,
                                         n_estimators=10, random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'DT': DecisionTreeClassifier(max_depth=5, random_state=SEED),
        'SVM': LinearSVC(random_state=SEED),
        'LR': LogisticRegression(penalty='l1', C=1e5, random_state=SEED),
        'BAG': BaggingClassifier(random_state=SEED),
        'NB': MultinomialNB(alpha=1.0)
        }


def iterate_models(data, grid_size, outcomes, features, missing, to_discretize, levels, 
                   metric_w_threshold, other_metrics, date_to_split, train_test_dates, 
                   labels, time_period, test_period, outfile, models_to_run):

    grid = config.define_clfs_params(grid_size)
    
    # getting header columns for results
    metrics, thresholds = metric_w_threshold
    INIT_COLUMNS = ['model_type', 'clf', 'parameters', 'outcome', 'traintestset',
                    'train_set_size', 'validation_set_size','features', 'baseline']
    metric_threshold = [m + '_at_' + str(t) for t in thresholds for m in metrics]
    columns = INIT_COLUMNS + metric_threshold + other_metrics

    # Write header for the csv
    with open(outfile, "w") as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(columns)

    # Define dataframe to write results to
    results_df =  pd.DataFrame(columns=columns)

    # Loop over models, parameters, outcomes, validation_Dates
    # and store several evaluation metrics

    for i,dates in enumerate(train_test_dates):
        # split train test sets
        x_train, x_test, y_train, y_test = temporal_split(data, outcomes, features, date_to_split,
                                                          dates[0], dates[1], 
                                                          str(time_period) + ' days',
                                                          str(test_period) + ' days')
        print('train test split done - {}'.format(labels[i]))

        # data cleaning on train test splits separately
        for j, var in enumerate(to_discretize):
            n = levels[j][0]
            bin_label = levels[j][1]
            print(var, n, bin_label)
            x_train, bins = etl.discretize(x_train, var, n, bin_label)
            print(bins)
            x_test = etl.discretize(x_test, var, bins, bin_label, False)
        print('discretization done')

        for m in missing:
            x_train.loc[:,m + '_missing'] = x_train[m].isna().astype(int)
            x_test.loc[:,m + '_missing'] = x_test[m].isna().astype(int)
        print('missing indicators created')

        x_train = etl.replace_missing_with_mode(x_train, x_train, missing)
        x_test = etl.replace_missing_with_mode(x_test, x_test, missing)
        print('imputation done')

        for d in config.CATEGORICAL:
            x_train = etl.create_dummies(x_train, d)
            x_test = etl.create_dummies(x_test, d)

        col = list(x_train.columns)
        for c in col:
            if c not in x_test.columns:
                x_test.loc[:,c] = 0
        x_test = x_test[col]
        # only consider the dummies that is created in the train set
        
        print('dummies created')


        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                print(models_to_run[index], labels[i])
                clf.set_params(**p) 

                # build model on train set
                model = clf.fit(x_train, y_train)
                print(clf)

                # get predictions from model on test only for the columns that appear in train
                if isinstance(clf, LinearSVC):
                    score = get_predicted_scores(clf, x_test, svm=True)
                else:
                    score = get_predicted_scores(clf, x_test, svm=False)

                # writing out results in pandas df
                strp = str(p)
                strp.replace('\n', '')
                strclf = str(clf)
                strclf.replace('\n', '')
                write_results = [ models_to_run[index], strclf, strp, outcomes,
                                    labels[i], len(x_train), len(x_test), features, 
                                    get_precision(y_test, score, 100) ]
                
                if metrics:
                    write_results += [eval_metric_dic[m](y_test, score, t) for t in thresholds for m in metrics]  
                if other_metrics:
                    write_results += [eval_metric_dic[m](y_test, score) for m in other_metrics]

                results_df.loc[len(results_df)] = write_results

                # plot precision recall graph
                name = str(p).replace(':','-')
                model_name = models_to_run[index] + ' ' + name + ' ' + labels[i]
                plot_precision_recall_n(y_test, score, model_name+'.png', 'save')
                print("saved graph", model_name)

                # writing out results in csv file
                with open(outfile, "a") as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow(write_results)

    # write final dataframe to csv
    dfoutfile = 'df_' + outfile
    results_df.to_csv(dfoutfile, index=False)
    print(dfoutfile, ' created')
    return results_df
