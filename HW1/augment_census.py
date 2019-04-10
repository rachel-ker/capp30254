'''
Homework 1 - Diagnostic
Rachel Ker

This code contains functions to augment with census data on zipcode
'''
import geopandas as gpd
from shapely.geometry import Point

import functions as fn


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

    df.loc[:,'geometry'] = geometry
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
    data = fn.get_data(url)
    
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
    crime_df = fn.get_crime_data([2017,2018])
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