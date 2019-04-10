'''
Homework 1 - Diagnostic
Rachel Ker

This code contains generic functions to deal with data
'''

import pandas as pd
import requests
import matplotlib.pyplot as plt


'''
GETTING DATASETS
'''

def get_data(url):
    '''
    Get Data from URL using requests library
    Inputs: url
    Returns a pandas dataframe
    '''
    req = requests.get(url)
    data = req.json()
    df = pd.DataFrame(data)
    return df


def get_crime_data(years):
    '''
    Get Crime data from City of Chicago Data Portal
    Inputs: list of years
    Returns a pandas dataframe
    '''
    df = pd.DataFrame()
    for yr in years:
        url = ('https://data.cityofchicago.org/resource/6zsd-86xi.json?year='
               + str(yr) + '&$limit=300000')
        data = get_data(url)
        df = df.append(data)
    return df


def arrest_data(df):
    '''
    Get subset of data where there is an arrest

    Inputs: dataframe
    Returns a dataframe of rows where arrest occured
    '''
    arrested = df[df['arrest']==True]
    return arrested



'''
GET BASIC STATS
'''

def crimes_by_type(df, groupby_cols):
    '''
    Get a sorted panda series from data by types of crime
    Inputs:
        panda df
        groupby_cols: list of cols to groupby
    Returns a dataframe in descending order of values
    '''
    series = df.groupby(groupby_cols).size()
    data = series.to_frame('count').reset_index()
    return data


def get_descriptive_stats(df, var):
    '''
    Get basic descriptive statisitics for census data on zipcodes
    Inputs:
        df: pandas dataframe
        var: list of variables of interest
    Returns a pandas dataframe
    '''
    return df.describe()[var]


def prob_crime_type_by_address(data, address):
    '''
    Get probability of each crime type by address
    Inputs: 
        data: pandas dataframe
        address string
    Return a pandas datafrane with probabilities for each crime type
    '''
    address = data[data['block'] == address]
    num = address.groupby('primary_type').size().sort_values(ascending=False)
    perc = num.apply(lambda x: (x/len(address))*100)
    return perc


def prob_of_crimetype(data, crime_type, community_areas):
    '''
    Calculating the probability that of crime in the community areas 
    given a certain crime type
    Inputs:
        crime_type: str in CAPS
        community_areas: list of community area number
    Returns a float
    '''
    crimetype = data[data['primary_type']==crime_type]

    df = pd.DataFrame()
    for ca in community_areas:
        data = crimetype[(crimetype['community_area'] == str(ca))]
        df = df.append(data)
    
    return len(df) / len(crimetype)


'''
DATA VISUALIZATION/TABLES
'''

def get_table(data, groupby_cols):
    '''
    Returns a table of crime by type and year
    Inputs:
        pandas dataframe
        groupby_cols: list of cols to groupby
    Returns a pandas dataframe of the top 10 values in descending order
    '''
    types_year = crimes_by_type(data, groupby_cols)
    df = types_year.pivot(index=groupby_cols[0], columns='year', values='count')
    df = df.sort_values(by='2018', ascending=False)
    df_10 = df.iloc[0:10]
    return df_10


def get_table_and_graph(df, groupby_cols, filename, title):
    '''
    Saves tables and plots based on the dataset and variables
    into the relevant folders 
    Inputs:
        df: pandas dataframe
        groupby_cols: list of cols to groupby
        filename: str
        title: str
    '''
    table = get_table(df, groupby_cols)
    table.to_csv("tables/" + filename + ".csv")
    table.plot.bar(figsize=(20,10), rot=0)
    plt.title(title)
    plt.savefig('bar_charts/' + filename)


