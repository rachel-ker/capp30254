'''
Homework 3
Machine Learning Pipeline Functions
This python file contains general functions to read, explore,
preprocss data, generate features, build classifer, and evaluate classifier

Rachel Ker
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import sklearn.metrics

from IPython.display import SVG
from IPython.display import display

import classifiers


######################
#  Build Classifier  #
######################

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
    return train_test_split(x, y, test_size=test_size, random_state=100)


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


def build_decision_trees(x_train, y_train, x_test, y_test,
                         y_col, threshold,
                         max_depth, min_leaf, criterion=['entropy','gini']):
    '''
    Build and compare different decision tree models
    
    Inputs:
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        threshold: specified probability threshold to classify obs as positive
        max_depth: (list of int) max depth of decision tree
        min_leaf: (list of int) min sample in the leaf of decision tree
        criterion: (list of str) optional, defaults to ['entropy','gini']

    Returns None
    '''
    for d in max_depth:
        for l in min_leaf:
            for c in criterion:
                dt = classifiers.build_decision_tree(x_train, y_train,
                                                      max_depth=d, min_leaf=l, criterion=c)
                
                scores = get_predicted_scores(dt, x_test)
                pred_label = get_prediction_labels(scores, threshold)
                
                print("max_depth: {}, min_leaf: {}, criterion: {}".format(d, l, c))
                evaluate_model(y_test, scores, threshold)



#######################
#   Get Prediction    #
#######################


def get_predicted_scores(model, x_test):
    '''
    Get prediction scores according to specified threshold

    Inputs:
        model: classifier object
        x_test: testing set features
    Returns an array of predicted scores
    '''
    scores = model.predict_proba(x_test)[:,1]
    return scores


def get_predicted_scores_for_svm(svm_model, x_test):
    '''
    Get prediction scores for svm

    Inputs:
        svm_model: svm object
        x_test: testing set features
    Returns an array of predicted scores
    '''
    scores = model.decision_function(x_test)
    return scores

    
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


def vary_threshold(model, x_test, y_test, threshold, svm=False):
    '''
    Runs through a list of threshold to pick a threshold to use for predicted labels
    Prints the number of predicted positives and accuracy score for each threshold
    to compare with actual number of positives in labels

    Inputs:
        model: 
        x_test, y_test: testing sets from data
        threshold: (list of threshold to test)
        svm: if the model is svm (defaults to False)    
    ''' 
    for t in threshold:
        if svm:
            scores = get_predicted_scores_for_svm(model, x_test)
        else:
            scores = get_predicted_scores(model, x_test)
        pred_label = get_prediction_labels(scores, t)
        evaluate_model(y_test, scores, t)



#######################
# Evaluate Classifier #
#######################

def get_accuracy_score(y_test, predicted_label):
    '''
    Get accuracy score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_label: predicted labels from the model

    Returns accuracy score
    '''
    return sklearn.metrics.accuracy_score(y_test, predicted_label)


def get_confusion_matrix(y_test, predicted_label):
    '''
    Get True Negatives, False Positives, False Negatives and True Positives

    Inputs:
        y_test: real labels for testing set
        predicted_label: predicted labels from the model
        threshold: specified probability threshold to classify obs as positive

    Returns (true_negatives, false_positives, false_negatives, true_positives)
    '''
    c = sklearn.metrics.confusion_matrix(y_test, predicted_label)
    return c.ravel()


def get_precision(y_test, predicted_label):
    '''
    Get precision score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_label: predicted labels from the model
    
    Returns precision score
    '''
    return sklearn.metrics.precision_score(y_test, predicted_label)
    


def get_recall(y_test, predicted_label):
    '''
    Get recall score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_label: predicted labels from the model
        
    Returns recall score
    '''
    return sklearn.metrics.recall_score(y_test, predicted_label)


def get_f1(y_test, predicted_label):
    '''
    Get f1 score score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_label: predicted labels from the model
        
    Returns f1 score
    '''
    return sklearn.metrics.f1_score(y_test, predicted_label)


def get_auc(y_test, predicted_score):
    '''
    Get area under the curve score score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_score: predicted score from predict proba/decision fn
        
    Returns area under the ROC
    '''
    return sklearn.metrics.roc_auc_score(y_test, predicted_score)


def plot_roc_curve(y_test, predicted_score, label):
    '''
    Plot the ROC curve
    Inputs:
        y_test: real labels for testing set
        predicted_score: predicted score from predict proba/decision fn
        label: (str) model specifications
    '''
    plt.figure()
    plt.plot(fpr, tpr, label=label + ' (area = {:.2f})'.format(get_auc(y_test, predicted_score)))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(label+'_ROC')
    plt.show()
    # (Source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)


def evaluate_model(y_test, predicted_score, threshold):
    '''
    Prints accuracy, precision and recall scores

    Inputs:
        y_test: real labels for testing set
        predicted_score: predicted score from predict proba/decision fn
        threshold: specified probability threshold to classify obs as positive
    '''
    print("The true number positives is {}/{} from the data, with percentage {:.2f}%\n".format(
          sum(y_test), len(y_test), 100.*sum(y_test)/len(y_test)))
    print("Threshold: {}".format(threshold))
    
    pred_label = get_prediction_labels(predicted_score, threshold)
    
    print("    The total number of predicted positives is {}".format(sum(pred_label)))
    print("    The accuracy is {:.2f}".format(get_accuracy_score(y_test, pred_label)))
    print("    The precision is {:.2f}".format(get_precision(y_test, pred_label)))
    print("    The recall is {:.2f}".format(get_recall(y_test, pred_label)))
    print("    The f1 score is {:.2f}".format(get_f1(y_test, pred_label)))
    print("    The area under the ROC is {:.2f}".format(get_auc(y_test, predicted_score)))
    print()
    
 
def plot_precision_recall_curve(y_test, predicted_scores):
    '''
    Plot precision and recall curves for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model

    '''
    precision, recall, thresholds = precision_recall_curve(y_test, predicted_scores)
    plt.plot(recall, precision, marker='.')
    plt.show()



# Evaluate Specific Chosen Classifiers #


def selected_decision_tree(df, x_train, y_train, x_test, y_test,
                           y_col, threshold,
                           max_depth, min_leaf, criterion):
    '''
    Visualize and get feature importance for the selected decision tree
    
    Inputs:
        df: pandas dataframe with selected features
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

    scores = get_predicted_scores(dt, x_test)
    pred_label = get_prediction_labels(scores, threshold)
    
    print("max_depth: {}, min_leaf: {}, criterion: {}".format(d, l, c))
    evaluate_model(y_test, pred_label, threshold)

    graph = classifiers.visualize_tree(dt, get_labels(df, y_col), ['No', 'Yes'])
    display(SVG(graph.pipe(format='svg')))
    print(classifiers.feature_importance(df, y_col, dt))
    
                         
