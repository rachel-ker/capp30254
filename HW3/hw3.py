'''
Homework 3
Applying the Pipeline to DonorsChoose Data

Rachel Ker
'''
import datetime

import etl
import pipeline
import classifiers


# Parameters for this code
filename = "data/projects_2012_2013.csv"

y_col = "notfullyfundedin60days"

features = ['school_latitude', 'school_longitude', 'total_price_including_optional_support', 'students_reached',
 'school_state_AL', 'school_state_AR', 'school_state_AZ', 'school_state_CA', 'school_state_CO', 'school_state_CT',
 'school_state_DC', 'school_state_DE', 'school_state_FL', 'school_state_GA', 'school_state_HI', 'school_state_IA',
 'school_state_ID', 'school_state_IL', 'school_state_IN', 'school_state_KS', 'school_state_KY', 'school_state_LA',
 'school_state_MA', 'school_state_MD', 'school_state_ME', 'school_state_MI', 'school_state_MN', 'school_state_MO',
 'school_state_MS', 'school_state_MT', 'school_state_NC', 'school_state_ND', 'school_state_NE', 'school_state_NH',
 'school_state_NJ', 'school_state_NM', 'school_state_NV', 'school_state_NY', 'school_state_OH', 'school_state_OK',
 'school_state_OR', 'school_state_PA', 'school_state_RI', 'school_state_SC', 'school_state_SD', 'school_state_TN',
 'school_state_TX', 'school_state_UT', 'school_state_VA', 'school_state_VT', 'school_state_WA', 'school_state_WI',
 'school_state_WV', 'school_state_WY', 'school_metro_suburban', 'school_metro_urban', 'school_charter_t',
 'school_magnet_t', 'teacher_prefix_Mr.', 'teacher_prefix_Mrs.', 'teacher_prefix_Ms.',
 'primary_focus_subject_Character Education', 'primary_focus_subject_Civics & Government', 'primary_focus_subject_College & Career Prep',
 'primary_focus_subject_Community Service', 'primary_focus_subject_ESL', 'primary_focus_subject_Early Development',
 'primary_focus_subject_Economics', 'primary_focus_subject_Environmental Science', 'primary_focus_subject_Extracurricular',
 'primary_focus_subject_Foreign Languages', 'primary_focus_subject_Gym & Fitness', 'primary_focus_subject_Health & Life Science',
 'primary_focus_subject_Health & Wellness', 'primary_focus_subject_History & Geography', 'primary_focus_subject_Literacy',
 'primary_focus_subject_Literature & Writing', 'primary_focus_subject_Mathematics', 'primary_focus_subject_Music',
 'primary_focus_subject_Nutrition', 'primary_focus_subject_Other', 'primary_focus_subject_Parent Involvement',
 'primary_focus_subject_Performing Arts', 'primary_focus_subject_Social Sciences', 'primary_focus_subject_Special Needs',
 'primary_focus_subject_Sports', 'primary_focus_subject_Visual Arts', 'secondary_focus_area_Health & Sports',
 'secondary_focus_area_History & Civics', 'secondary_focus_area_Literacy & Language', 'secondary_focus_area_Math & Science',
 'secondary_focus_area_Music & The Arts', 'secondary_focus_area_Special Needs', 'secondary_focus_subject_Character Education',
 'secondary_focus_subject_Civics & Government', 'secondary_focus_subject_College & Career Prep', 'secondary_focus_subject_Community Service',
 'secondary_focus_subject_ESL', 'secondary_focus_subject_Early Development', 'secondary_focus_subject_Economics',
 'secondary_focus_subject_Environmental Science', 'secondary_focus_subject_Extracurricular', 'secondary_focus_subject_Foreign Languages',
 'secondary_focus_subject_Gym & Fitness', 'secondary_focus_subject_Health & Life Science', 'secondary_focus_subject_Health & Wellness',
 'secondary_focus_subject_History & Geography', 'secondary_focus_subject_Literacy', 'secondary_focus_subject_Literature & Writing',
 'secondary_focus_subject_Mathematics', 'secondary_focus_subject_Music', 'secondary_focus_subject_Nutrition', 'secondary_focus_subject_Other',
 'secondary_focus_subject_Parent Involvement', 'secondary_focus_subject_Performing Arts', 'secondary_focus_subject_Social Sciences',
 'secondary_focus_subject_Special Needs', 'secondary_focus_subject_Sports', 'secondary_focus_subject_Visual Arts',
 'primary_focus_area_Health & Sports', 'primary_focus_area_History & Civics', 'primary_focus_area_Literacy & Language',
 'primary_focus_area_Math & Science', 'primary_focus_area_Music & The Arts', 'primary_focus_area_Special Needs',
 'resource_type_Other', 'resource_type_Supplies', 'resource_type_Technology', 'resource_type_Trips', 'resource_type_Visitors',
 'poverty_level_highest poverty', 'poverty_level_low poverty', 'poverty_level_moderate poverty',
 'grade_level_Grades 6-8', 'grade_level_Grades 9-12', 'grade_level_Grades PreK-2', 'eligible_double_your_impact_match_t']

date_col = "date_posted"
train1_start_date = (2012, 1, 1)
train1_end_date = (2012, 6, 30)
test1_start_date = (2012, 7, 1)
test1_end_date= (2012, 12, 31)

train2_start_date = (2012, 1, 1)
train2_end_date = (2012, 12, 31)
test2_start_date = (2013, 1, 1)
test2_end_date= (2013, 6, 30)

train3_start_date = (2012, 1, 1)
train3_end_date = (2013, 6, 30)
test3_start_date = (2013, 7, 1)
test3_end_date= (2013, 12, 31)




def hw3():
    df = etl.read_csvfile(filename)
    df = etl.replace_dates_with_datetime(df, ["date_posted", "datefullyfunded"])
    df = create_label(df, y_col)

    # Replace Missing Values
    df = etl.replace_missing_with_mode(df, ['school_metro', 'school_district', 'primary_focus_subject', 'primary_focus_area', 
                                            'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'grade_level'])
    df = etl.replace_missing_with_mean(df, ['students_reached'])

    # Create dummy variables
    df = etl.create_dummies(df, 'school_state')
    df = etl.create_dummies(df, 'school_metro')
    df = etl.create_dummies(df, 'school_charter')
    df = etl.create_dummies(df, 'school_magnet')
    df = etl.create_dummies(df, 'teacher_prefix')
    df = etl.create_dummies(df, 'primary_focus_subject')
    df = etl.create_dummies(df, 'secondary_focus_area')
    df = etl.create_dummies(df, 'secondary_focus_subject')
    df = etl.create_dummies(df, 'primary_focus_area')
    df = etl.create_dummies(df, 'resource_type')
    df = etl.create_dummies(df, 'poverty_level')
    df = etl.create_dummies(df, 'grade_level')
    df = etl.create_dummies(df, 'eligible_double_your_impact_match')

    # Temporal splits     
    x_train1, x_test1, y_train1, y_test1 = pipeline.temporal_split(df, y_col, features, date_col,
                                                                   train1_start_date, train1_end_date,
                                                                   test1_start_date, test1_end_date)

    x_train2, x_test2, y_train2, y_test2 = pipeline.temporal_split(df, y_col, features, date_col,
                                                                   train2_start_date, train2_end_date,
                                                                   test2_start_date, test2_end_date)

    x_train3, x_test3, y_train3, y_test3 = pipeline.temporal_split(df, y_col, features, date_col,
                                                                   train3_start_date, train3_end_date,
                                                                   test3_start_date, test3_end_date)
    
    
    # build models
    # evaluate models
    # create a table to tally the different results
    dt = classifiers.build_decision_tree(x_train1, y_train1,
                                         max_depth=10, min_leaf=100, criterion='gini')
    pipeline.vary_threshold(dt, x_test1, y_test1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    pipeline.build_decision_trees(x_train1, y_train1, x_test1, y_test1, y_col, 0.45, [20,10,8,5,3], [1000,100,10])
                







def create_label(df, label_name):
    df.loc[:,'60daysafterpost'] = df['date_posted'] + datetime.timedelta(days=60)
    df.loc[:, label_name] = df['datefullyfunded'] > df['60daysafterpost']
    df[label_name] = df[label_name].astype(int)
    return df


    


    
    
