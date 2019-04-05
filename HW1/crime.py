'''
Homework 1 - Diagnostic
Rachel Ker
'''

import pandas as pd
import matplotlib.pyplot as plt


def get_data(csv2017, csv2018):
    '''
    Function to read data from csv and append into one dataframe
    
    Inputs: file path for 2017 crime and 2017 crime data
    Returns a pd data frame with both datasets
    '''
    crimes2017 = pd.read_csv(csv2017)
    crimes2018 = pd.read_csv(csv2018)
    data = crimes2017.append(crimes2018)
    return data


def arrest_data(df):
    '''
    Get subset of data where there is an arrest

    Inputs: dataframe
    Returns a dataframe of rows where arrest occured
    '''
    arrested = df[df['Arrest']==True]
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
    data = series.to_frame('Count').reset_index()
    return data


def get_table_and_charts(data, groupby_col):
    '''
    Prints a table of crime by type and year and
    returns a plot (matplotlib.axes.Axes)

    Inputs:
        pandas dataframe
        groupby_cols: list of 2 cols to groupby
    '''
    types_year = crimes_by_type(data, groupby_col)
    sortby = data[groupby_col[1]].max()
    df = types_year.pivot(index=groupby_col[0], columns='Year', values='Count').sort_values(by=2017, ascending=False)
    df_10 = df.iloc[0:10]
    print(df_10)
    return df_10.plot.bar(figsize=(20,10), rot=0)


def crime_summary():
    '''
    Gets summary statistics of crime data
    '''
    data = get_data("data/Crimes2017.csv", "data/Crimes2018.csv")

    print("Number of reported incidents of crime by year")
    num_reported = data.groupby('Year').size().to_frame('Count').reset_index()
    print(num_reported)
    print()

    print("Top 10 Reported crimes by type by year")
    fig = get_table_and_charts(data, ['Primary Type', 'Year'])
    plt.title('Theft is the most reported crime in Chicago in 2017 and 2018')
    plt.savefig('Top 10 Crime reported by type by year')
    print()
    
    print("Top 10 Arrested crimes by type by year")
    fig = get_table_and_charts(arrest_data(data), ['Primary Type', 'Year'])
    plt.title('Narcotics results in an arrest outcome the most in Chicago in 2017 and 2018')
    plt.savefig('Top 10 Crime arrested by type by year')
    print()

    print("Top 10 Neighborhoods with reported crime by year")
    fig = get_table_and_charts(data, ['Community Area', 'Year'])
    plt.title('Community Area 25, 8, 32 has the most crime reported in Chicago in 2017 and 2018')
    plt.savefig("Top 10 Neighborhoods with reported crime by year")
    print()
    
    print("Top 10 Neighborhoods with arrests by year")
    fig = get_table_and_charts(arrest_data(data), ['Community Area', 'Year'])
    plt.title('Community Area 25, 29, 23 has the most crime arrests in Chicago in 2017 and 2018')
    plt.savefig("Top 10 Neighborhoods with reported crime by year")
    print()

    # Find out more about these communities (what type of neighborhoods?)


