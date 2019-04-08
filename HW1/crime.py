'''
Homework 1 - Diagnostic
Rachel Ker
'''

import pandas as pd
import matplotlib.pyplot as plt
import requests
import censusgeocode as cg


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


# Problem 1: Data Acquistion and Analysis

def get_crime_data(year):
    '''
    Get Crime data from City of Chicago Data Portal
    Inputs: year
    Returns a pandas dataframe
    '''
    url = ('https://data.cityofchicago.org/resource/6zsd-86xi.json?year='
          + str(year) + '&$limit=300000')
    df = get_data(url)
    return df

def get_both_years():
    data_2017 = get_crime_data(2017)
    data_2018 = get_crime_data(2018)
    data = data_2017.append(data_2018)
    return data

def arrest_data(df):
    '''
    Get subset of data where there is an arrest

    Inputs: dataframe
    Returns a dataframe of rows where arrest occured
    '''
    arrested = df[df['arrest']==True]
    return arrested


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


def get_table(data, groupby_col):
    '''
    Returns a table of crime by type and year

    Inputs:
        pandas dataframe
        groupby_cols: list of cols to groupby
    '''
    types_year = crimes_by_type(data, groupby_col)
    sortby = data[groupby_col[1]].max()
    df = types_year.pivot(index=groupby_col[0], columns='year', values='count').sort_values(by='2018', ascending=False)
    df_10 = df.iloc[0:10]
    return df_10


def crime_summary():
    '''
    Gets summary statistics of crime data
    '''
    data = get_both_years()

    print("Number of reported incidents of crime by year")
    num_reported = data.groupby('year').size().to_frame('count').reset_index()
    print(num_reported)
    print()

    print("Top 10 Reported crimes by type by year")
    table1 = get_table(data, ['primary_type', 'year'])
    print(table1)
    table1.plot.bar(figsize=(20,10), rot=0)
    plt.title('Theft is the most reported crime in Chicago in 2017 and 2018')
    plt.savefig('bar_charts/Top 10 Crime reported by type by year')
    print()
    
    print("Top 10 Arrested crimes by type by year")
    table2 = get_table(arrest_data(data), ['primary_type', 'year'])
    print(table2)
    table2.plot.bar(figsize=(20,10), rot=0)
    plt.title('Narcotics results in an arrest outcome the most in Chicago in 2017 and 2018')
    plt.savefig('bar_charts/Top 10 Crime arrested by type by year')
    print()

    print("Top 10 Neighborhoods with most reported crime by year")
    table3 = get_table(data, ['community_area', 'year'])
    table3.plot.bar(figsize=(20,10), rot=0)
    plt.title('Community Area 25, 8, 32 has the most reported crime in Chicago in 2017 and 2018')
    plt.savefig('bar_charts/Top 10 neighborhoods with most reported crime by year')
    print()
    
    print("Top 10 Neighborhoods with most arrests by year")
    table4 = get_table(arrest_data(data), ['community_area', 'year'])
    table4.plot.bar(figsize=(20,10), rot=0)
    plt.title('Community Area 25, 29, 23 has the most arrests of reported crime in Chicago in 2017 and 2018')
    plt.savefig('bar_charts/Top 10 neighborhoods with most arrests by year')
    print()

    print("Top 10 Wards with most reported crime by year")
    table3 = get_table(data, ['ward', 'year'])
    table3.plot.bar(figsize=(20,10), rot=0)
    plt.title('Wards 42, 24, 28 has the most reported crime in Chicago in 2017 and 2018')
    plt.savefig('bar_charts/Top 10 wards with most reported crime by year')
    print()
    
    print("Top 10 Wards with most arrests by year")
    table4 = get_table(arrest_data(data), ['ward', 'year'])
    table4.plot.bar(figsize=(20,10), rot=0)
    plt.title('Wards 24, 28, 27 has the most arrests of reported crime in Chicago in 2017 and 2018')
    plt.savefig('bar_charts/Top 10 wards with most arrests by year')
    print()



# Problem 2: Data Augmentation and APIS


def get_fipscode(row):
    '''
    Get census block from latitude and longitude
    
    Inputs: lat, lon
    Returns census block
    '''
    result= cg.coordinates(row['longitude'], row['latitude'])
    fipscode = result['2010 Census Blocks'][0]['GEOID']
    return fipscode

#  (Source: https://pypi.org/project/censusgeocode/)


def adding_fipscode_to_df(df):
    '''
    Adding FIPS Code column to the dataframe

    Inputs: Pandas dataframe
    Returns a dataframe with the corresponding zipcode added
    '''
    df = df[~df['latitude'].isna() & ~df['longitude'].isna()]
    fips = df.apply(get_fipscode, axis=1)

    df['fips_code'] = fips
    return df


def get_census_data():
    '''
    Get Data from the 5-year ACS Census Estimates
    '''
    pass


def augment():
    '''
    Augments crime data with ACS data
    '''
    df = get_both_years()

    df_homicide = df[df['primary_type'] == 'BATTERY']
    df = adding_fipscode_to_df(df_homicide)

    df_homicide = df[df['primary_type'] == 'HOMICIDE']
    df = adding_fipscode_to_df(df_homicide)
    return df


# (From https://jtleider.github.io/censusdata/)
# census


# Problem 3: Analysis and Communication



