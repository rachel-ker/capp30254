'''
Code for Building Different Classifiers
Rachel Ker
'''

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, 
AdaBoostClassifier)

import graphviz
from mlxtend.plotting import plot_decision_regions

SEED = 0


#######################
# K-nearest neighbors #
#######################

def build_knn(x_train, y_train, k, weight, p=2, dist='minkowski'):
    '''
    Build a k-nearest neighbors model
    Inputs:
        x_train, y_train: training sets for features and labels
        k: (int) number of neighbors
        weight: (str) 'uniform' or 'distance'
        p: (int) optional, default=2 (Euclidian distance).
            also, p=1 (manhatten distance)
        dist: (str) optional, distance metric. default is 'minkowski'
        
    '''
    knn = KNeighborsClassifier(n_neighbors=k,
                               weights=weight,
                               p=p,
                               metric=dist)
    knn.fit(x_train, y_train)
    return knn


def find_closest_neighbors(knn, n, point):
    '''
    Identifiest the closest neighbors
    Inputs:
        knn: knn model object
        n: (int) number of closest neighbors
        point: (array) array that represents that instance
    
    Returns an array of (distance, index)
    '''
    return knn.kneighbors(X=point, n_neighbors=n)


######################
#   Decision Tree    #
######################

def build_decision_tree(x_train, y_train, criterion, max_depth, min_leaf):
    '''
    Build a decision tree classifier
    Inputs:
        x_train, y_train: training sets for features and labels
        criterion: (str) gini or entropy
        max_depth: (int) max depth of decision tree
        min_leaf: (int) min sample in the leaf of decision tree
        
    Returns Decision Tree Classifier object
    '''
    dt_model = tree.DecisionTreeClassifier(criterion=criterion,
                                           splitter='best',
                                           max_depth=max_depth,
                                           min_samples_leaf=min_leaf,
                                           random_state=SEED)
    dt_model.fit(x_train, y_train)
    return dt_model



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


def feature_importance(dt, y_col, features):
    '''
    Get the feature importance of each feature
    Inputs:
        dt: decision tree
        y_col: (str) column name of target variable
        feature: a list of labels for features
    Return a dataframe of feature importance
    '''
    d = {'Features': features,
         'Importance': dt.feature_importances_}
    feature_importance = pd.DataFrame(data=d)
    feature_importance = feature_importance.sort_values(by=['Importance'],
                                                        ascending=False)
    return feature_importance
    


#######################
# Logistic Regression #
#######################


def build_logistic_regression(x_train, y_train, penalty, c):
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
                            random_state=SEED)
    lr.fit(x_train, y_train)
    return lr


def plot_regression_line(lr, x_train):
    plt.lr()
    plt.scatter(x_train.ravel(), y, color='black', zorder=20)
    X_test = np.linspace(-5, 10, 300)

    def model(x):
        return 1 / (1 + np.exp(-x))

    loss = model(X_test * lr.coef_ + lr.intercept_).ravel()
    plt.plot(X_test, loss, color='red', linewidth=3)
    plt.show()
    # (Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html)


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
    svm = LinearSVC(tol=1e-5, C=c, random_state=SEED)
    svm.fit(x_train, y_train)
    return svm


def plot_svm_boundaries(svm, x_train, y_train, y_col):
    '''
    Plot decision boundaries for svm

    Inputs:
        svm: svm classifier object
        x_train, y_train: training sets
        y_col: (str) column name of target variable
    '''
    plot_decision_regions(X=x_train, 
                          y=y_train,
                          clf=svm, 
                          legend=2)

    plt.xlabel('Features', size=14)
    plt.ylabel(y_col, size=14)
    plt.title('SVM Decision Region Boundary', size=16)
    plt.show()
    # (Source: http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions)



##########################
#     Random Forests     #
##########################

def build_random_forest(x_train, y_train, n, criterion, max_depth, min_leaf):
    '''
    Build a Random Forest classifier
    Inputs:
        x_train, y_train: training sets for features and labels
        n: number of trees in the forest
        criterion: (str) gini or entropy
        max_depth: (int) max depth of decision tree
        min_leaf: (int) min sample in the leaf of decision tree
    Returns classifier object
    '''
    rf = RandomForestClassifier(n_estimators=n,
                                criterion=criterion,
                                max_depth=max_depth,
                                min_samples_leaf=min_leaf,
                                random_state=SEED)
    rf.fit(x_train, y_train)
    return rf
    


##########################
#       Bagging          #
##########################

def build_bagging(x_train, y_train, base_model, n, n_jobs):
    '''
    Build a Bagging classifier
    Inputs:
        x_train, y_train: training sets for features and labels
        base_model: classifier object
        n: number of base estimator in the ensemble
        n_jobs: (int) number of jobs to run in parallel
    Returns classifier object
    '''
    bag = BaggingClassifier(base_estimator=base_model,
                            n_estimators=n,
                            n_jobs=n_jobs,
                            random_state=SEED)
    bag.fit(x_train, y_train)
    return bag
                            

##########################
#       Boosting         #
##########################

def build_ada_boosting(x_train, y_train, base, n):
    '''
    Build a gradient boosting classifier
    Inputs:
        x_train, y_train: training sets for features and labels
        base: classifier object
        n: number of base estimator in the ensemble
    Returns classifier object  
    '''
    ada = AdaBoostClassifier(base_estimator=base,
                             n_estimators=n,
                             random_state=SEED)
    ada.fit(x_train, y_train)
    return ada


def build_gradient_boosting(x_train, y_train, n):
    '''
    Build a gradient boosting classifier
    Inputs:
        x_train, y_train: training sets for features and labels
        n: number of base estimator in the ensemble
    Returns classifier object  
    '''
    gbc = GradientBoostingClassifier(n_estimators=n,
                                     random_state=SEED)
    gbc.fit(x_train, y_train)
    return gbc

# Good read: https://medium.com/@rrfd/boosting-bagging-and-stacking-ensemble-methods-with-sklearn-and-mlens-a455c0c982de
