'''
Homework 1 - Diagnostic
Rachel Ker
'''

import functions as fn
import augment_census


# Problem 1: Data Acquistion and Analysis

def problem1():
    '''
    Gets summary statistics of crime data
    '''
    data = fn.get_crime_data([2017,2018])

    print("Number of reported incidents of crime by year")
    num_reported = data.groupby('year').size().to_frame('count').reset_index()
    num_reported.to_csv("tables/reported_crime.csv")
    print()

    print("Top 10 Reported crimes by type by year")
    fn.get_table_and_graph(data,
                           ['primary_type', 'year'],
                           'top10reported_bytype_byyear',
                           'Theft is the most reported crime in Chicago in 2017 and 2018')
    
    print("Top 10 Arrested crimes by type by year")
    fn.get_table_and_graph(fn.arrest_data(data),
                           ['primary_type', 'year'],
                           'top10arrested_bytype_byyear',
                           'Narcotics results in the most arrests in Chicago in 2017 and 2018')

    print("Top 10 Community Areas with most reported crime by year")
    fn.get_table_and_graph(data,
                           ['community_area', 'year'],
                           'top10reported_bycommarea_byyear',
                           'Austin has the most reported crime in Chicago in 2017 and 2018')
    
    print("Top 10 Neighborhoods with most arrests by year")
    fn.get_table_and_graph(fn.arrest_data(data),
                           ['community_area', 'year'],
                           'top10arrested_bycommarea_byyear',
                           'Austin has the most arrests in Chicago in 2017 and 2018')

    print("Top 10 Wards with most reported crime by year")
    fn.get_table_and_graph(data,
                           ['ward', 'year'],
                           'top10reported_byward_byyear',
                           'Ward 42 has the most reported crime in Chicago in 2017 and 2018')

    print("Top 10 Wards with most arrests by year")
    fn.get_table_and_graph(fn.arrest_data(data),
                           ['ward', 'year'],
                           'top10arrested_byward_byyear',
                           'Ward 42 has the most arrests in Chicago in 2017 and 2018')


# Problem 2: Data Augmentation and APIS

def problem2():
    '''
    Analyzes crime data augmented with census data on zipcode to answer policy questions
    Returns a Geodataframe
    '''
    data = augment_census.augment()
 
    census_var = ['white', 'nointernetaccess', 'totalpop', 'avghhsize', 
                  'medianinc', 'incpovratiobelow0.5']
    for var in census_var:
        data[var] = data[var].apply(float)

    data['perc_white'] = (data['white']/data['totalpop'])
    data['perc_nointernet'] = (data['nointernetaccess']/data['totalpop'])
    data['perc_poverty'] = (data['incpovratiobelow0.5']/data['totalpop'])

    var_of_interest = ['perc_white', 'perc_nointernet', 'perc_poverty',
                       'avghhsize','medianinc']

    overall_stats = fn.get_descriptive_stats(data, var_of_interest)

    print("\nOverall\n")
    print(overall_stats)

    battery_2017 = data[(data['year']=='2017') & (data['primary_type']=='BATTERY')]
    print("\nBattery 2017\n")
    print(fn.get_descriptive_stats(battery_2017, var_of_interest))

    battery_2018 = data[(data['year']=='2018') & (data['primary_type']=='BATTERY')]
    print("\nBattery 2018\n")
    print(fn.get_descriptive_stats(battery_2018, var_of_interest))

    homicide_2017 = data[(data['year']=='2017') & (data['primary_type']=='HOMICIDE')]
    print("\nHomicide 2017\n")
    print(fn.get_descriptive_stats(homicide_2017, var_of_interest))

    homicide_2018 = data[(data['year']=='2018') & (data['primary_type']=='HOMICIDE')]
    print("\nHomicide 2018\n")
    print(fn.get_descriptive_stats(homicide_2018, var_of_interest))

    dp = fn.get_descriptive_stats(data[data['primary_type']=='DECEPTIVE PRACTICE'],
                                  var_of_interest)
    print("\nDeceptive Practice\n")
    print(dp)

    sex_offense = fn.get_descriptive_stats(data[data['primary_type']=='SEX OFFENSE'],
                                           var_of_interest)
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
    data = get_crime_data([2017,2018])
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

