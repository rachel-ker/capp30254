'''
Homework 2
Machine Learning Pipeline Functions
This python file contains general functions to read, explore,
preprocss data, generate features, build classifer, and evaluate classifier

Rachel Ker
'''

import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score


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

def scatterplot(df):
	pass

def descriptive_stats(df):
	pass

def detect_outliers(df):
	pass

def corr_coeff(df):
	pass


###################
#  Preprocessing  #
###################

def replace_missing(df):
	pass


######################
# Feature Generation #
######################

def discretize(continous_var):
	pass

def create_dummies(categorical_var):
	pass

def standardize(var):
	pass

######################
#  Build Classifier  #
######################

def build_decision_tree(df):
	dt_model = tree.DecisionTreeClassifier()
	dt_model.fit(x_train, y_train)
	return dt_model
# https://scikit-learn.org/stable/modules/tree.html#


def predict(model, df):
	values = pass
	predictions = []
	for val in values:
		pred = model.predict(values)
		predictions.append(pred)
	return predictions


#######################
# Evaluate Classifier #
#######################

def get_accuracy_score(y_pred, y_true):
	'''
	Get the fraction of the correctly classified instances

	Inputs:
		y_pred: predictions of label using model
		y_true: true labels

	Returns a float between 0 to 1
	'''
	return accuracy_score(y_pred, y_true)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
