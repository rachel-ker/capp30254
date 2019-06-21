#!/usr/bin/env python
# coding: utf-8

'''
Feature Generation

Author: Rachel Ker
'''

import pandas as pd
import config


def get_raw_data(file_list):
    '''
    Read in data
    Return a combined dataset
    '''
    df = pd.DataFrame()
    return df        


def get_labels():
    pass


def add_all_features(df):
    '''
    Generate features to add to the data
    Return a dataset with features generated
    '''
    return df


def full_dataset():
    df = get_raw_data([])
    df = add_all_features(df)
    return df


def get_train_test_splits():
    pass


if __name__ == '__main__':
    full_dataset()