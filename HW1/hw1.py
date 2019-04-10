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

    overall_stats = fn.get_descriptive_stats(data, var_of_interest, "overall")

    print("\nOverall\n")
    print(overall_stats)

    battery_2017 = data[(data['year']=='2017') & (data['primary_type']=='BATTERY')]
    print("\nBattery 2017\n")
    print(fn.get_descriptive_stats(battery_2017, var_of_interest, "battery2017"))

    battery_2018 = data[(data['year']=='2018') & (data['primary_type']=='BATTERY')]
    print("\nBattery 2018\n")
    print(fn.get_descriptive_stats(battery_2018, var_of_interest, "battery2018"))

    homicide_2017 = data[(data['year']=='2017') & (data['primary_type']=='HOMICIDE')]
    print("\nHomicide 2017\n")
    print(fn.get_descriptive_stats(homicide_2017, var_of_interest, "homicide2017"))

    homicide_2018 = data[(data['year']=='2018') & (data['primary_type']=='HOMICIDE')]
    print("\nHomicide 2018\n")
    print(fn.get_descriptive_stats(homicide_2018, var_of_interest, "homicide2018"))

    dp = fn.get_descriptive_stats(data[data['primary_type']=='DECEPTIVE PRACTICE'],
                                  var_of_interest, "deceptivepractice1718")
    print("\nDeceptive Practice\n")
    print(dp)

    sex_offense = fn.get_descriptive_stats(data[data['primary_type']=='SEX OFFENSE'],
                                           var_of_interest, "sexoffense1718")
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

def problem3():
    '''
    Calculating crime change from 2017 to 2018
    '''
    data = fn.get_crime_data([2017,2018])

    data2017 = fn.crimes_by_type(data[data['year'] == '2017'], 'primary_type')
    data2018 = fn.crimes_by_type(data[data['year'] == '2018'], 'primary_type')
    all_data = fn.calculate_perc_change(data2017, data2018, "alldata")
    print(all_data)

    jul2018 = fn.crimes_by_type(fn.get_dates(data, '2018-06-26', '2018-07-26'),
                                ['primary_type'])
    jul2017 = fn.crimes_by_type(fn.get_dates(data, '2017-06-26', '2017-07-26'),
                                ['primary_type'])
    jul_data = fn.calculate_perc_change(jul2017, jul2018, "monthofjul")
    print(jul_data)

    ward43 = data[data['ward']=='43']
    w43jul2018 = fn.crimes_by_type(fn.get_dates(ward43, '2018-06-26', '2018-07-26'),
                                   ['primary_type'])
    w43jul2017 = fn.crimes_by_type(fn.get_dates(ward43, '2017-06-26', '2017-07-26'),
                                   ['primary_type'])
    w43jul_data = fn.calculate_perc_change(w43jul2017, w43jul2018, "monthofjul_w43")
    print(w43jul_data)

    yr_to_date_2018 = fn.crimes_by_type(fn.get_dates(data, '2018-01-01', '2018-07-26'),
                                        ['primary_type'])
    yr_to_date_2017 = fn.crimes_by_type(fn.get_dates(data, '2017-01-01', '2017-07-26'),
                                        ['primary_type'])

    y2d_data = fn.calculate_perc_change(yr_to_date_2017, yr_to_date_2018, "yr_to_date")
    print(y2d_data)



# Problem 4

def problem4():
    '''
    Calculating probabilities by address or crime type
    '''
    data = fn.get_crime_data([2017,2018])
    michigan = fn.prob_crime_type_by_address(data, '021XX S MICHIGAN AVE')
    print(michigan)

    gar_theft = fn.prob_of_crimetype(data, "THEFT", [26,27])
    uptown_theft = fn.prob_of_crimetype(data, "THEFT", [3])
    # East Garfield Park and West Garfield Park as defined in CPD Community Areas
    # http://home.chicagopolice.org/wp-content/uploads/2014/11/communitymap_nov2016.pdf

    # Uptown as defined in CPD Community Area
    # http://home.chicagopolice.org/wp-content/uploads/2014/11/communitymap_nov2016.pdf
    
    print("\nProbability of call from Garfield Park given it is a battery report\n")
    print(gar_theft)
    print("\nProbability of call from Uptown given it is a battery report\n")
    print(uptown_theft)
    print("\nDifference\n")
    print(gar_theft - uptown_theft)












