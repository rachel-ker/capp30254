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
    scores = svm_model.decision_function(x_test)
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

    Returns a dictionary
    ''' 
    table = {"true_num_positives": [],
             "total_obs": [],
             "threshold": [],
             "predicted_num_positives": [],
             "accuracy": [],
             "precision": [],
             "recall": [],
             "f1_score": [],
             "auc": []}
    
    for t in threshold:
        if svm:
            scores = get_predicted_scores_for_svm(model, x_test)
        else:
            scores = get_predicted_scores(model, x_test)

        table["true_num_positives"] += [sum(y_test)]
        table["total_obs"] += [len(y_test)]
        table["threshold"] += [t]
        table["predicted_num_positives"] += [sum(get_prediction_labels(scores, t))]
        table["accuracy"] += [get_accuracy_score(y_test, scores, t)]
        table["precision"] += [get_precision(y_test, scores, t)]
        table["recall"] += [get_recall(y_test, scores, t)]
        table["f1_score"] += [get_f1(y_test, scores, t)]
        table["auc"] += [get_auc(y_test, scores)]

    return table



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
        threshold: specified probability threshold to classify obs as positive
    Returns accuracy score
    '''
    predicted_label = get_prediction_labels(predicted_score, threshold)
    return sklearn.metrics.accuracy_score(y_test, predicted_label)


def get_confusion_matrix(y_test, predicted_score, threshold):
    '''
    Get True Negatives, False Positives, False Negatives and True Positives

    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        threshold: specified probability threshold to classify obs as positive
    Returns (true_negatives, false_positives, false_negatives, true_positives)
    '''
    predicted_label = get_prediction_labels(predicted_score, threshold)
    c = sklearn.metrics.confusion_matrix(y_test, predicted_label)
    return c.ravel()


def get_precision(y_test, predicted_score, threshold):
    '''
    Get precision score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        threshold: specified probability threshold to classify obs as positive    
    Returns precision score
    '''
    predicted_label = get_prediction_labels(predicted_score, threshold)
    return sklearn.metrics.precision_score(y_test, predicted_label)
    


def get_recall(y_test, predicted_score, threshold):
    '''
    Get recall score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        threshold: specified probability threshold to classify obs as positive        
    Returns recall score
    '''
    predicted_label = get_prediction_labels(predicted_score, threshold)
    return sklearn.metrics.recall_score(y_test, predicted_label)


def get_f1(y_test, predicted_score, threshold):
    '''
    Get f1 score score for the predicted labels
    
    Inputs:
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        threshold: specified probability threshold to classify obs as positive        
    Returns f1 score
    '''
    predicted_label = get_prediction_labels(predicted_score, threshold)
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

def random_baseline(x_test):
    '''
    Get random predictions
    '''
    random_score = [random.uniform(0,1) for i in enumerate(x_test)]
    calc_threshold = lambda x, y: 0 if x < y else 1 
    random_predicted = np.array( [calc_threshold(score, 0.5) for score in random_score] )
    return random_predicted


def plot_roc_curve(y_test, predicted_score, label):
    '''
    Plot the ROC curve
    Inputs:
        y_test: real labels for testing set
        predicted_score: predicted score from predict proba/decision fn
        label: (str) model specifications
    '''
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, predicted_score)
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

 
def plot_precision_recall_curve(model, y_test, predicted_scores, model_name):
    '''
    Plot precision and recall curves for the predicted labels
    
    Inputs:
        model: classifier object
        y_test: real labels for testing set
        predicted_scores: array of predicted scores from model
        model_name: (str) title for plot

    '''
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test,
                                                                           predicted_scores)
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()


def plot_precision_recall_n(model, y_true, y_prob, model_name):
    '''
    Plot precision and recall for each threshold

    Inputs:
        model: classifier object
        y_true: real labels for testing set
        y_prob: array of predicted scores from model
        model_name: (str) title for plot
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
    l1,=ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylim(0,1.05)
    ax2 = ax1.twinx().twiny()
    l2,=ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_xlabel('threshold')
    ax2.set_xticks(pr_thresholds, ['0.0','0.1','0.2','0.3','0.4','0.5',
                                   '0.6','0.7','0.8','0.9','1.0'])
    ax2.set_ylim(0,1.05)
    plt.legend([l1,l2],['precision','recall'], loc=2)

    plt.title(model_name)
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
    

# Evaluate Specific Classifiers #

def build_knn_models(x_train, y_train, x_test, y_test,
                     y_col, threshold, train_test_split,
                     k, weight=['uniform', 'distance'], p=[1,2]):
    '''
    Build and compare different knn models
    
    Inputs:
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        threshold: (list of threshold to test)
        train_test_split: (str) label for the different splits
        k: (list of int) number of neighbors
        weight: (list of str) optional
        p: (list of int) optional

    Returns a pandas dataframe summary of models and metrics
    '''
    print("building k-nn models...")
    df = pd.DataFrame()
 
    for n in k:
        for w in weight:
            for num in p:
                knn = classifiers.build_knn(x_train, y_train, k=n, weight=w, p=num)

                if num==1:
                    dist = "Manhatten"
                elif num==2:
                    dist = "Euclidean"
                
                dic = vary_threshold(knn, x_test, y_test, threshold)
                dic["model"] = "k-nn - k: {}, weight: {}, dist: {}".format(n, w, dist)
                dic["train_test_split"] = train_test_split

                table = pd.DataFrame(data=dic)
                df = df.append(table)
    return df    
    


def build_decision_trees(x_train, y_train, x_test, y_test,
                         y_col, threshold, train_test_split,
                         max_depth, min_leaf, criterion=['entropy','gini']):
    '''
    Build and compare different decision tree models
    
    Inputs:
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        threshold: (list of threshold to test)
        train_test_split: (str) label for the different splits
        max_depth: (list of int) max depth of decision tree
        min_leaf: (list of int) min sample in the leaf of decision tree
        criterion: (list of str) optional, defaults to ['entropy','gini']

    Returns a pandas dataframe summary of models and metrics
    '''
    print("building decision tree models...")

    df = pd.DataFrame()
    
    for d in max_depth:
        for l in min_leaf:
            for cr in criterion:
                dt = classifiers.build_decision_tree(x_train, y_train, cr, d, l)                                
                dic = vary_threshold(dt, x_test, y_test, threshold)
                dic["model"] = "decision tree - max_depth: {}, min_leaf: {}, criterion: {}".format(d, l, cr)
                dic["train_test_split"] = train_test_split
                table = pd.DataFrame(data=dic)
                df = df.append(table)
    return df


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

    scores = get_predicted_scores(dt, x_test)
    pred_label = get_prediction_labels(scores, threshold)
    
    print("max_depth: {}, min_leaf: {}, criterion: {}".format(max_depth, min_leaf, criterion))
    graph = classifiers.visualize_tree(dt, feature_labels, class_labels)
    display(SVG(graph.pipe(format='svg')))
    print(classifiers.feature_importance(dt, y_col, feature_labels))



def build_logistic_regressions(x_train, y_train, x_test, y_test,
                               y_col, threshold, train_test_split,
                               c, penalty=['l1','l2']):
    '''
    Build and compare different logistic regression models
    
    Inputs:
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        threshold: (list of threshold to test)
        train_test_split: (str) label for the different splits
        c: (list of positive floats) strength of regularization;
        smaller values are stronger regularization      
        penalty: (list of str) optional

    Returns a pandas dataframe summary of models and metrics
    '''
    print("building logistic regressions...")

    df = pd.DataFrame()
    
    for flt in c:
        for p in penalty:
            lr = classifiers.build_logistic_regression(x_train, y_train,
                                                       penalty=p, c=flt)                                
            dic = vary_threshold(lr, x_test, y_test, threshold)
            dic["model"] = "logistic regression - penalty: {}, c: {}".format(p, flt)
            dic["train_test_split"] = train_test_split
            table = pd.DataFrame(data=dic)
            df = df.append(table)
    return df    


def build_svm_models(x_train, y_train, x_test, y_test,
                    y_col, threshold, train_test_split, c):
    '''
    Build and compare different SVM models
    
    Inputs:
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        threshold: (list of threshold to test)
        train_test_split: (str) label for the different splits
        c: (list of positive floats) strength of regularization;
        smaller values are stronger regularization         

    Returns a pandas dataframe summary of models and metrics
    '''
    print("building svm models...")

    df = pd.DataFrame()

    for flt in c:
        svm = classifiers.build_svm(x_train, y_train, c=flt)                                
        dic = vary_threshold(svm, x_test, y_test, threshold, svm=True)
        # think about how to vary threshold for svm
        
        dic["model"] = "SVM - c: {}".format(flt)
        dic["train_test_split"] = train_test_split
        table = pd.DataFrame(data=dic)
        df = df.append(table)
    return df


def build_ada_boostings(x_train, y_train, x_test, y_test,
                       y_col, threshold, train_test_split,
                       base_model, n_estimators):
    '''
    Build and compare different Ada boosting models
    
    Inputs:
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        threshold: (list of threshold to test)
        train_test_split: (str) label for the different splits
        base_model: (list of models)
        n_estimators: (list of int) number of base estimators
    

    Returns a pandas dataframe summary of models and metrics
    '''
    print("building adaboost models...")

    df = pd.DataFrame()
    
    for m in base_model:
        for n in n_estimators:
            ada = classifiers.build_ada_boosting(x_train, y_train, base=m, n=n)                                
            dic = vary_threshold(ada, x_test, y_test, threshold)
            dic["model"] = "AdaBoosting - base_model: {}, n_estimators: {}".format(m,n)
            dic["train_test_split"] = train_test_split
            table = pd.DataFrame(data=dic)
            df = df.append(table)
    return df 


def build_gradient_boostings(x_train, y_train, x_test, y_test,
                            y_col, threshold, train_test_split,
                            n_estimators):
    '''
    Build and compare different Gradient boosting models
    
    Inputs:
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        threshold: (list of threshold to test)
        train_test_split: (str) label for the different splits
        n_estimators: (list of int) number of base estimators    
    
    Returns a pandas dataframe summary of models and metrics
    '''
    print("building gradient boosting models...")

    df = pd.DataFrame()
    
    for n in n_estimators:
        gbm = classifiers.build_gradient_boosting(x_train, y_train, n=n)                                
        dic = vary_threshold(gbm, x_test, y_test, threshold)
        dic["model"] = "Gradient Boosting - n_estimators: {}".format(n)
        dic["train_test_split"] = train_test_split
        table = pd.DataFrame(data=dic)
        df = df.append(table)
    return df 


def build_bagging_models(x_train, y_train, x_test, y_test,
                        y_col, threshold, train_test_split,
                        base_model, n_estimators, n_jobs):
    '''
    Build and compare different Bagging Classifiers
    
    Inputs:
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        threshold: (list of threshold to test)
        train_test_split: (str) label for the different splits
        base_model: (list of models)
        n_estimators: (list of int) number of trees in the forest
        n_jobs: (int) number of jobs to run in parallel

    Returns a pandas dataframe summary of models and metrics
    '''
    print("building bagging models...")

    df = pd.DataFrame()

    for m in base_model:
        for n in n_estimators:
            for job in n_jobs:
                bag = classifiers.build_bagging(x_train, y_train, m, n, job)                                
                dic = vary_threshold(bag, x_test, y_test, threshold)
                dic["model"] = "bagging - base_model: {}, n_estimator: {}, \
                                n_jobs: {}".format(m, n, job)
                dic["train_test_split"] = train_test_split
                table = pd.DataFrame(data=dic)
                df = df.append(table)
    return df



def build_random_forests(x_train, y_train, x_test, y_test,
                        y_col, threshold, train_test_split,
                        n_estimators, max_depth, min_leaf,
                        criterion=['entropy', 'gini']):
    '''
    Build and compare different Random Forests
    
    Inputs:
        x_train, y_train: training sets from the data
        x_test, y_test: testing sets from data
        y_col: (str) column name of target variable
        threshold: (list of threshold to test)
        train_test_split: (str) label for the different splits
        n_estimators: (list of int) number of trees in the forest
        max_depth: (list of int) max depth of decision tree
        min_leaf: (list of int) min sample in the leaf of decision tree
        criterion: (list of str) optional, defaults to ['entropy','gini'] 

    Returns a pandas dataframe summary of models and metrics
    '''
    print("building random forest models...")

    df = pd.DataFrame()

    for n in n_estimators:
        for d in max_depth:
            for l in min_leaf:
                for cr in criterion:
                    rf = classifiers.build_random_forest(x_train, y_train, n, cr, d, l)                                
                    dic = vary_threshold(rf, x_test, y_test, threshold)
                    dic["model"] = "random forest - n_estimator: {}, max_depth: {}, \
                                    min_leaf: {}, criterion: {}".format(n, d, l, cr)
                    dic["train_test_split"] = train_test_split
                    table = pd.DataFrame(data=dic)
                    df = df.append(table)
    return df


def build_all_models(x_train, y_train, x_test, y_test,
                     y_col, threshold, train_test_split,
                     k, max_depth, min_leaf, c, n_estimators,
                     base_model_bag, base_model_ada, n_jobs):
    '''
    Build Decision Trees, K-nearest neighbors, logistic regression, SVM
    for the train and test set at the different parameters
    Returns a pandas dataframe summary of models and metrics
    '''
    models = pd.DataFrame()
    dt = build_decision_trees(x_train, y_train, x_test, y_test, y_col,
                              threshold, train_test_split, max_depth, min_leaf)
    
    knn = build_knn_models(x_train, y_train, x_test, y_test, y_col,
                           threshold, train_test_split, k)
    
    lr = build_logistic_regressions(x_train, y_train, x_test, y_test, y_col,
                                    threshold, train_test_split, c)
    
    svm = build_svm_models(x_train, y_train, x_test, y_test, y_col,
                           threshold, train_test_split, c)

    rf = build_random_forests(x_train, y_train, x_test, y_test, y_col,
                              threshold, train_test_split, n_estimators, max_depth, min_leaf)

    bag = build_bagging_models(x_train, y_train, x_test, y_test, y_col,
                              threshold, train_test_split, base_model_bag,
                              n_estimators, n_jobs)

    ada = build_ada_boostings(x_train, y_train, x_test, y_test, y_col,
                              threshold, train_test_split, base_model_ada,
                              n_estimators)

    gbm = build_gradient_boostings(x_train, y_train, x_test, y_test, y_col,
                              threshold, train_test_split, n_estimators)

    print('done with model building')
    tables = [dt, knn, lr, svm, rf, bag, ada, gbm]
    
    for table in tables:
        models= models.append(table)
    
    return models
    

