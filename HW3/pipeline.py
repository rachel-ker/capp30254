'''
Homework 3
Machine Learning Pipeline Functions
This python file contains general functions to do train-test splits,
get predictions, and evaluate classifier

etl.py contains functions to read, explore, clean data
classifiers.py contains functions to build classifers

Rachel Ker
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn.metrics

from IPython.display import SVG
from IPython.display import display

import classifiers

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
                   train_start_date, train_end_date,
                   test_start_date, test_end_date):
    '''
    Splits dataset by time
    Inputs:
        df: pandas dataframe
        y_col: col name for target variable
        features: list of features
        train_start_date, train_end_date: tuple of (year, month, date) - training set start and end dates inclusive
        test_start_date, test_end_date: tuple of (year, month, date) - testing set start and end dates inclusive
    Returns x_train, x_test, y_train, y_test
    '''
    train_start_yy, train_start_mm, train_start_dd = train_start_date
    train_end_yy, train_end_mm, train_end_dd = train_end_date
    test_start_yy, test_start_mm, test_start_dd = test_start_date
    test_end_yy, test_end_mm, test_end_dd = test_end_date

    train = df[(df[date_col] >= pd.Timestamp(train_start_yy, train_start_mm, train_start_dd)) 
               & (df[date_col] <= pd.Timestamp(train_end_yy, train_end_mm, train_end_dd))]
    test = df[(df[date_col] >= pd.Timestamp(test_start_yy, test_start_mm, test_start_dd)) 
               & (df[date_col] <= pd.Timestamp(test_end_yy, test_end_mm, test_end_dd))]

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

    
def get_prediction_labels(predicted_scores, threshold):
    '''
    Get prediction labels according to specified threshold

    Inputs:
        predicted_scores: array of predicted scores from model
        threshold: specified probability threshold to classify obs as positive
    Returns an array of predicted labels
    '''
    calc_threshold = lambda x, y: 0 if x < y else 1 
    predict_label = np.array( [calc_threshold(score, threshold) for score in predicted_scores] )                    
    return predict_label



#######################
# Evaluate Classifier #
#######################

# General evaluation metrics

def get_accuracy_score(y_test, predicted_score, threshold):
    '''
    Get accuracy score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        threshold: specified % threshold to classify obs as positive
    Returns accuracy score
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
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
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
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
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
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
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
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
    plt.savefig('ROC curves')
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
    plt.savefig('precision recall curve')
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
        plt.savefig(model_name + 'precision_recall_threshold')
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()
    # (Source: https://github.com/dssg/hitchhikers-guide/blob/master/sources/curriculum/3_modeling_and_machine_learning/machine-learning/machine_learning_clean.ipynb)


# Getting Best models #

def best_model(df, metric):
    '''
    Identify best models for the specified metric
    Inputs:
        df: pandas dataframe
        metric: (str) precision, recall, accuracy, f1_score, auc
    
    Return dataframe with best models
    '''
    return df[df[metric] == df[metric].max()]
    


# Visualize Tree #

def selected_decision_tree(df, x_train, y_train, x_test, y_test,
                           y_col, threshold, 
                           max_depth, min_leaf, criterion,
                           feature_labels, class_labels):
    '''
    Visualize and get feature importance for the selected decision tree
    
    Inputs:
        df: pandas dataframe
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        max_depth: (int) max depth of decision tree
        min_leaf: (int) min sample in the leaf of decision tree
        criterion: (str) 'entropy' or 'gini'

    Returns None
    '''
    dt = classifiers.build_decision_tree(x_train, y_train,
                                          max_depth=max_depth, min_leaf=min_leaf,
                                          criterion=criterion)
    
    print("max_depth: {}, min_leaf: {}, criterion: {}".format(max_depth, min_leaf, criterion))
    graph = classifiers.visualize_tree(dt, feature_labels, class_labels)
    display(SVG(graph.pipe(format='svg')))
    print(classifiers.feature_importance(dt, y_col, feature_labels))






