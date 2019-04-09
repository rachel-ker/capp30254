'''
Homework 1 - Diagnostic
Rachel Ker
'''

import pandas as pd
import matplotlib.pyplot as plt
import requests
import geopandas as gpd
from shapely.geometry import Point



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
    '''
    Get crime data from 2017 and 2018
    Returns a pandas dataframe
    '''
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
    Returns a pandas dataframe of the top 10 values in descending orderß
    '''
    types_year = crimes_by_type(data, groupby_col)
    df = types_year.pivot(index=groupby_col[0], columns='year', values='count')
    df = df.sort_values(by='2018', ascending=False)
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

def get_geometry(row):
    '''
    Get shapely point object from latitude and longitude
    
    Inputs: row
    Returns shapely point objects
    '''
    return Point(float(row['longitude']), float(row['latitude']))
#  (Source: https://shapely.readthedocs.io/en/stable/manual.html#points)


def df_to_geodataframe(df):
    '''
    Adding geometry column to the dataframe

    Inputs: Pandas dataframe
    Returns a geoDataFrame with geometry
    '''
    df = df[~df['latitude'].isna() & ~df['longitude'].isna()]
    geometry = df.apply(get_geometry, axis=1)

    df['geometry'] = geometry
    crs = {'init': 'epsg:4326'}
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
#   (Source: https://gis.stackexchange.com/questions/174159/convert-a-pandas-dataframe-to-a-geodataframe)
    return gdf


def get_census_data():
    '''
    Get Data from the 5-year ACS Census Estimates

    Variables include total population, race - white only,
    no internet access, average household size, income poverty ratio below 0.5,
    median income in the past 12 months

    Data is by zip code area

    Returns pandas dataframe
    '''
    url = ("https://api.census.gov/data/2017/acs/acs5?" +
           "get=GEO_ID,B01003_001E,B02001_002E,B28002_013E,B25010_001E," +
           "C17002_002E,B19013_001E,NAME&for=zip%20code%20tabulation%20area:*")
    data = get_data(url)
    
    header = data.iloc[0]
    data = data[1:]
    data = data.rename(columns = header)

    return data
#   (Source: https://www.census.gov/data/developers/data-sets/acs-5year.html)


def get_shape_data():
    '''
    Gets the geodataframe representing the polygons of each zipcode
    '''
    fname = "https://data.cityofchicago.org/resource/unjd-c2ca.geojson"
    df = gpd.read_file(fname)
    return df
#   (Source: https://ocefpaf.github.io/python4oceanographers/blog/2015/03/30/geo_pandas/)


def augment():
    '''
    Augments crime data with ACS data

    Saves and Returns dataframe that combines acs and crime data with zipcode
    '''
    # Use spatial join to get the zipcodes from lat and lon
    crime_df = get_both_years()
    crime_df = df_to_geodataframe(crime_df)

    zipcodedata = get_shape_data()

    crime_with_zip = gpd.sjoin(crime_df, zipcodedata,
                               how="inner", op='intersects')
    crime_with_zip = crime_with_zip.set_index('zip')
#   (Source: http://geopandas.org/mergingdata.html#spatial-joins)

    acs_detailed = get_census_data()
    rename_dict = {'B01003_001E': 'totalpop',
                   'B02001_002E': 'white',
                   'B28002_013E': 'nointernetaccess',
                   'B25010_001E': 'avghhsize',
                   'C17002_002E': 'incpovratiobelow0.5',
                   'B19013_001E': 'medianinc'}
    acs_detailed.rename(columns = rename_dict, inplace=True)
    acs_detailed = acs_detailed.set_index('zip code tabulation area')

    # Left join on zipcode
    crime_with_acs = crime_with_zip.join(acs_detailed).reset_index()

    crime_with_acs.to_csv("crime_with_acs.csv")
    return crime_with_acs


def get_descriptive_stats(df):
    '''
    Get basic descriptive statisitics
    Returns a pandas dataframe
    '''
    var_of_interest = ['perc_white', 'perc_nointernet', 'perc_poverty',
                       'avghhsize','medianinc']
    return df.describe()[var_of_interest]


def data_analysis():
    '''
    Analyzes data to answer policy questions
    Returns a Geodataframe
    '''
    data = augment()
 
    vars = ['white', 'nointernetaccess', 'totalpop', 'avghhsize', 
            'medianinc', 'incpovratiobelow0.5']
    for var in vars:
        data[var] = data[var].apply(float)

    data['perc_white'] = (data['white']/data['totalpop'])
    data['perc_nointernet'] = (data['nointernetaccess']/data['totalpop'])
    data['perc_poverty'] = (data['incpovratiobelow0.5']/data['totalpop'])
    overall_stats = get_descriptive_stats(data)

    print("\nOverall\n")
    print(overall_stats)

    battery_2017 = data[(data['year']=='2017') & (data['primary_type']=='BATTERY')]
    print("\nBattery 2017\n")
    print(get_descriptive_stats(battery_2017))

    battery_2018 = data[(data['year']=='2018') & (data['primary_type']=='BATTERY')]
    print("\nBattery 2018\n")
    print(get_descriptive_stats(battery_2018))

    homicide_2017 = data[(data['year']=='2017') & (data['primary_type']=='HOMICIDE')]
    print("\nHomicide 2017\n")
    print(get_descriptive_stats(homicide_2017))

    homicide_2018 = data[(data['year']=='2018') & (data['primary_type']=='HOMICIDE')]
    print("\nHomicide 2018\n")
    print(get_descriptive_stats(homicide_2018))

    dp = get_descriptive_stats(data[data['primary_type']=='DECEPTIVE PRACTICE'])
    print("\nDeceptive Practice\n")
    print(dp)

    sex_offense = get_descriptive_stats(data[data['primary_type']=='SEX OFFENSE'])
    print("\nSex Offenses\n")
    print(sex_offense)

    return data

#    To try doing plots in future
#    data.plot(column='medianinc', cmap='OrRd', scheme='quantiles')
#    plt.show()

#    print('First plot')

#    fig, ax = plt.subplots()
#    ax.set_aspect('equal')
#    data.plot(ax=ax, color='white', edgecolor='black')
#    battery_2017 = data[data['year']=='2017' & data['primary_type']=='BATTERY']
#    battery_2017.plot(ax=ax, marker='o', color='red', markersize=5)
#    plt.show()
#   (Source: http://geopandas.org/mapping.html)



# Problem 3: Analysis and Communication

def transform_dates(row):
    '''
    Function to get date-time object
    '''
    return pd.to_datetime(row['date'])


def get_dates(df, start_date, end_date):
    '''
    Filter a dataframe by the start and end date

    Inputs:
        df: pandas dataframe
        start_date / end_date: date time strings
        e.g. '2017-01-01'
    '''
    dates = df.apply(transform_dates, axis=1)
    df['datetime'] = dates

    df = df[(df['datetime'] > start_date) & (df['datetime'] < end_date)]
    return df


def crime_statistics():
    data = get_both_years()
    ward43 = data[data['ward']=='43']

    jul2018 = crimes_by_type(get_dates(data, '2018-06-26', '2018-07-26'),
                             ['primary_type'])
    jul2017 = crimes_by_type(get_dates(data, '2017-06-26', '2017-07-26'),
                             ['primary_type'])

    yr_to_date_2018 = crimes_by_type(get_dates(data, '2018-01-01', '2018-07-26'),
                                     ['primary_type'])
    yr_to_date_2017 = crimes_by_type(get_dates(data, '2017-01-01', '2017-07-26'),
                                     ['primary_type'])

    new_data = jul2017.join(jul2018)

    pass

