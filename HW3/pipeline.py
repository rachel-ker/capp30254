'''
Homework 3
Machine Learning Pipeline Functions
This python file contains general functions to read, explore,
preprocss data, generate features, build classifer, and evaluate classifier

Rachel Ker
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

from IPython.display import SVG
from IPython.display import display

import etl
import classifiers


######################
#  Build Classifier  #
######################

# To include:
# Logistic Regression, K-Nearest Neighbor, SVM
# Random Forests, Boosting, and Bagging


def split_training_testing(df, y_col, features, test_size):
    '''
    Splits the dataset into training and testing sets
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


def build_decision_trees(df, x_train, y_train, x_test, y_test,
                         y_col, threshold,
                         max_depth, min_leaf, criterion=['entropy','gini']):
    '''
    Build and compare different decision tree models
    
    Inputs:
        df: pandas dataframe with selected features
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
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
                evaluate_model(y_test, pred_label)


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
    scores = model.predict_proba(x_test)[:,1])
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


def pick_threshold(model, x_test, y_test, threshold, svm=False):
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
    print("The true number positives is {}/{} from the data, with percentage {:.2f}%\n".format(
          sum(y_test), len(y_test), 100.*sum(y_test)/len(y_test)))
    
    for t in threshold:
        if svm:
            scores = get_predicted_scores_for_svm(model, x_test)
        else:
            scores = get_predicted_scores(model, x_test)
        pred_label = get_prediction_labels(scores, t)
        print("(Threshold: {}), the total number of predicted positives is {}".format(
              threshold, sum(pred_label)))
        evaluate_model(y_test, pred_label)



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
    return accuracy_score(y_test, predicted_label)


def get_confusion_matrix(y_test, predicted_label):
    '''
    Get True Negatives, False Positives, False Negatives and True Positives

    Inputs:
        y_test: real labels for testing set
        predicted_label: predicted labels from the model

    Returns (true_negatives, false_positives, false_negatives, true_positives)
    '''
    c = confusion_matrix(y_test, predicted_label
    return c.ravel()


def get_precision(y_test, predicted_label):
    '''
    Get precision score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_label: predicted labels from the model
    
    Returns precision score
    '''
    _, false_positives, _, true_positives = get_confusion_matrix(y_test, predicted_label)
    return 1.0 * true_positives / (false_positives + true_positives)
    


def get_recall(y_test, predicted_label):
    '''
    Get recall score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_label: predicted labels from the model
        
    Returns recall score
    '''
    _, _, false_negatives, true_positives = get_confusion_matrix(y_test, predicted_label)
    return 1.0 * true_positives / (false_negatives + true_positives)


def evaluate_model(y_test, predicted_label):
    '''
    Prints accuracy, precision and recall scores

    Inputs:
        y_test: real labels for testing set
        predicted_label: predicted labels from the model
    '''
    print("    The accuracy is {:.2f}".format(get_accuracy_score(y_test, pred_label)))
    print("    The precision is {:.2f}".format(get_precision_score(y_test, pred_label)))
    print("    The recall is {:.2f}".format(get_recall_score(y_test, pred_label)))
    print()
    
 
def plot_precision_recall_curve(y_test, predicted_scores):
    '''
    Plot precision and recall curves for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model

    '''
    precision, recall, thresholds = precision_recall_curve(y_test, predicted scores))
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
    evaluate_model(y_test, pred_label)

    graph = classifiers.visualize_tree(dt, get_labels(df, y_col), ['No', 'Yes'])
    display(SVG(graph.pipe(format='svg')))
    print(classifiers.feature_importance(df, y_col, dt))
    
                         
