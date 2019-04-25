'''
Code for Building Different Classifiers
Rachel Ker
'''

import pandas as pd

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import graphviz


######################
#   Decision Tree    #
######################

def build_decision_tree(x_train, y_train, criterion, max_depth, min_leaf):
    '''
    Build a decision tree classifier
    Inputs:
        x_train, y_train: training sets for features and labels
        max_depth: (int) max depth of decision tree
        min_leaf: (int) min sample in the leaf of decision tree
        
    Returns Decision Tree Classifier object
    '''
    dt_model = tree.DecisionTreeClassifier(criterion=criterion,
                                           splitter='best',
                                           max_depth=max_depth,
                                           min_samples_leaf=min_leaf,
                                           random_state=0)
    dt_model.fit(x_train, y_train)
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


def feature_importance(df, y_col, dt):
    '''
    Get the feature importance of each feature
    Inputs:
        df: pandas dataframe
        y_col: (str) column name of target variable
        dt: decision tree
    Return a dataframe of feature importance
    '''
    d = {'Features': get_labels(df, y_col),
         'Importance': dt.feature_importances_}
    feature_importance = pd.DataFrame(data=d)
    feature_importance = feature_importance.sort_values(by=['Importance'],
                                                        ascending=False)
    return feature_importance
    
    
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
# Logistic Regression #
#######################


def logistic_regression(x_train, y_train, penalty, c):
    '''
    Build a logistic regression classifier
    Inputs:
        x_train, y_train: training sets for features and labels
        penalty: (str) regularization 'l1' or 'l2'
        c: (positive float) strength of regularization
            smaller values specify stronger regularization
        
    Returns classifier object
    '''
    lr = LogisticRegression(penalty=penalty,
                            C=c,
                            solver='liblinear',
                            random_state=0)
    lr.fit(x_train, y_train)
    return lr

# lr.coeff_


##########################
# Support Vector Machine #
##########################

def build_svm(x_train, y_train, c):
    '''
    Build a SVM classifier
    Inputs:
        x_train, y_train: training sets for features and labels
        c: (positive float) strength of regularization
            smaller values specify stronger regularization        
    Returns classifier object
    '''
    svm = LinearSVC(random_state=0, tol=1e-5, C=c)
    svm.fit(x_train, y_train)
    return svm


