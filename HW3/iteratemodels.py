'''
Code to iterate the models and get evaluation metrics
'''
import pandas as pd
import datetime
import csv
import itertools

from sklearn.model_selection import ParameterGrid
from sklearn.svm import LinearSVC

import etl
import classifiers
import pipeline



def iterate_models(data, grid_size, outcomes, outfile,
                   models_to_run=['RF', 'ET', 'GB', 'AB', 'BAG'
                                  'DT', 'KNN', 'LR', 'SVM']):

    # Train start-end/test start-end dates we want to loop over
    # [train_start_date, train_end_date, test_start_date, test_end_date]
    train_test_dates = [[(2012, 1, 1), (2012, 6, 30), (2012, 7, 1), (2012, 12, 31)],
                        [(2012, 1, 1), (2012, 12, 31), (2013, 1, 1), (2013, 6, 30)],
                        [(2012, 1, 1), (2013, 6, 30), (2013, 7, 1), (2013, 12, 31)]]
    labels = ["jan12-jun12/jul12-dec12", "jan12-dec12/jan13-jun13", "jan12-jun13/jul13-dec13"]

    clfs, grid = classifiers.define_clfs_params(grid_size)

    # Select features
    features = ['school_state','school_metro', 'school_charter',
       'school_magnet', 'teacher_prefix', 'primary_focus_subject',
       'primary_focus_area', 'secondary_focus_subject', 'secondary_focus_area',
       'resource_type', 'poverty_level', 'grade_level',
       'total_price_including_optional_support_discrete', 'students_reached_discrete',
       'eligible_double_your_impact_match']

    # Write header for the csv
    with open(outfile, "w") as myfile:
        myfile.write("model_type,clf,parameters,outcome,trainsetlabel,train_set_size,validation_set_size," +
                     "features,baseline,precision_at_1,precision_at_2,precision_at_5,precision_at_10,precision_at_20," +
                     "precision_at_30,precision_at_50, recall_at_1,recall_at_2,recall_at_5,recall_at_10,recall_at_20," +
                     "recall_at_30,recall_at_50,auc-roc")

    # Define dataframe to write results to
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'outcome', 'trainsetlabel',
                                        'train_set_size', 'validation_set_size','features',
                                        'baseline', 'precision_at_1', 'precision_at_2', 'precision_at_5', 
                                        'precision_at_10', 'precision_at_20', 'precision_at_30', 'precision_at_50',
                                        'recall_at_1', 'recall_at_2', 'recall_at_5', 'recall_at_10', 'recall_at_20',
                                        'recall_at_30', 'recall_at_50','auc-roc'))


    # Loop over models, parameters, outcomes, validation_Dates
    # and store several evaluation metrics
    
    for i,dates in enumerate(train_test_dates):
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    print(models_to_run[index], labels[i])
                    clf.set_params(**p)                 

                    date_col = "date_posted"
                    x_train, x_test, y_train, y_test = pipeline.temporal_split(data, outcomes, features, date_col,
                                                                               dates[0], dates[1], dates[2], dates[3])

                    missing = ['school_metro', 'primary_focus_subject', 'primary_focus_area',
                               'secondary_focus_subject', 'secondary_focus_area', 'resource_type',
                               'grade_level', 'students_reached_discrete']
                    x_train = etl.replace_missing_with_mode(x_train, x_train, missing)
                    x_test = etl.replace_missing_with_mode(x_test, data, missing)

                    for d in features:
                        x_train = etl.create_dummies(x_train, d)
                        x_test = etl.create_dummies(x_test, d)

                    col = list(x_train.columns)
                    model = clf.fit(x_train, y_train)
                    print("done with fitting ", models_to_run[index])

                    if isinstance(clf, LinearSVC):
                        score = pipeline.get_predicted_scores(clf, x_test[col], svm=True)
                    else:
                        score = pipeline.get_predicted_scores(clf, x_test[col], svm=False)

                    results_df.loc[len(results_df)] = [ models_to_run[index], clf, p, outcomes,
                                                        labels[i], len(x_train), len(x_test), features, 
                                                        pipeline.get_precision(y_test, score, 100),
                                                        pipeline.get_precision(y_test, score, 1),
                                                        pipeline.get_precision(y_test, score, 2),
                                                        pipeline.get_precision(y_test, score, 5),
                                                        pipeline.get_precision(y_test, score, 10),
                                                        pipeline.get_precision(y_test, score, 20),
                                                        pipeline.get_precision(y_test, score, 30),
                                                        pipeline.get_precision(y_test, score, 50),
                                                        pipeline.get_recall(y_test, score, 1),
                                                        pipeline.get_recall(y_test, score, 2),
                                                        pipeline.get_recall(y_test, score, 5),
                                                        pipeline.get_recall(y_test, score, 10),
                                                        pipeline.get_recall(y_test, score, 20),
                                                        pipeline.get_recall(y_test, score, 30),
                                                        pipeline.get_recall(y_test, score, 50),
                                                        pipeline.get_auc(y_test, score)]

                    # plot precision recall graph
#                    pipeline.plot_precision_recall_n(y_test, score, clf, 'show')

                    with open(outfile, "a") as myfile:
                        csvwriter = csv.writer(myfile, dialect='excel', quoting=csv.QUOTE_ALL)
                        strp = str(p)
                        strp.replace('\n', '')
                        strclf = str(clf)
                        strclf.replace('\n', '')
                        csvwriter.writerow([models_to_run[index],strclf, strp, outcomes,
                                            labels[i], len(x_train),len(x_test), features,
                                            pipeline.get_precision(y_test, score, 100),
                                            pipeline.get_precision(y_test, score, 1),
                                            pipeline.get_precision(y_test, score, 2),
                                            pipeline.get_precision(y_test, score, 5),
                                            pipeline.get_precision(y_test, score, 10),
                                            pipeline.get_precision(y_test, score, 20),
                                            pipeline.get_precision(y_test, score, 30),
                                            pipeline.get_precision(y_test, score, 50),
                                            pipeline.get_recall(y_test, score, 1),
                                            pipeline.get_recall(y_test, score, 2),
                                            pipeline.get_recall(y_test, score, 5),
                                            pipeline.get_recall(y_test, score, 10),
                                            pipeline.get_recall(y_test, score, 20),
                                            pipeline.get_recall(y_test, score, 30),
                                            pipeline.get_recall(y_test, score, 50),
                                            pipeline.get_auc(y_test, score)])
                except IndexError:
                    print("IndexError")
                    continue

    
    # write final dataframe to csv
    dfoutfile = 'df_' + outfile
    results_df.to_csv(dfoutfile, index=False)
    return results_df


def get_subsets(l):
    subsets = []
    for i in range(1, len(l) + 1):
        for combo in itertools.combinations(l, i):
            subsets.append(list(combo))
    return subsets
