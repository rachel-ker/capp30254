'''
Homework 3
Applying the Pipeline to DonorsChoose Data

Rachel Ker
'''

import etl

def go():
    filename = "data/projects_2012_2013.csv"
    y_col = "fullyfundedin60days"
    selected_features = [''] # to include after cleaning
    test_size = 0.2
    
    x_train, x_test, y_train, y_test = get_training_testing_data(
                                       filename, y_col, features, test_size)

    # clean training set - e.g. processing missing variables
    
    # build models
    # evaluate models
    

def get_training_testing_data(filepath, y_col, features, test_size):
    df = etl.read_csvfile(filepath)
    
    # create the 'fullyfundedin60days' col
    # clean data - changing categoricals to dummies
    
    return split_training_testing(df, y_col, features, test_size)
    


    
    
