'''
Configuration File
'''

## DATA FILE / LABEL
DATAFILE = "data/projects_2012_2013.csv"

## CHANGE DATA TYPES
TO_DATE = ["date_posted", "datefullyfunded"]

## DATA CLEANING
MISSING = ['school_metro', 'school_district', 'primary_focus_subject', 'primary_focus_area',
           'secondary_focus_subject', 'secondary_focus_area', 'resource_type',
           'grade_level', 'students_reached']

CATEGORICAL = ['school_state', 'school_metro', 'school_charter', 'school_magnet', 
            'primary_focus_subject', 'primary_focus_area', 'secondary_focus_subject', 
            'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level',
            'total_price_including_optional_support', 'students_reached',
            'eligible_double_your_impact_match']
CONTINUOUS = ['school_latitude', 'school_longitude', 'total_price_including_optional_support', 
              'students_reached']

## FEATURES
FEATURES = CATEGORICAL + CONTINUOUS

