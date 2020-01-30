'''
Prediction Challenge Processing steps
'''
import pandas as pd
import numpy as np
import math


def read(file):
    '''
    Read in training data.
    '''
    df = pd.read_csv(file)

    return df

def clean_data(dataframe):
    '''
    Rename unique id column, drop duplicate id column,
    and specify the training label.
    '''
    dataframe.rename(columns = {'Unnamed: 0':'id',
                                'U1031900':'label'}, inplace = True)
    dataframe.drop(columns=['diag.id'], inplace=True)

    return dataframe

def drop_nonresponse_y(dataframe):
    '''
    Drops invalid values in the training label.
    '''
    index_nums = dataframe[dataframe.label < 0].index
    small_data = dataframe.drop(index_nums)

    return small_data

def create_dummies(x_train, feature):
    '''
    Takes a categorical variable and creates dummy features from it,
    which are concatenated to the end of the dataframe. Drops the original
    variable from the dataset.

    Output:
        df (dataframe): dataframe with new dummy variable columns
    '''
    categories = x_train[feature].unique().tolist()
    dummies = pd.get_dummies(x_train[feature], prefix=feature)
    x_train = pd.concat([x_train, dummies], axis=1)
    x_train.drop([feature], axis=1, inplace=True)
    # Set up temporary 'other' column to account for test data
    other_col = feature + '_' + 'Other'
    if other_col not in x_train.columns:
        x_train[other_col] = 0

    return x_train, categories


def create_dummies_test(x_test, feature, categories):
    '''
    Creates dummy columns for x_test by including the same dummy columns
    as appear in x_train and categorizing records that do not fit into these
    categories as "Other". Drops the original variable from the dataset.
    '''
    for val in categories:
        col_name = feature + '_' + str(val)
        x_test[col_name] = 0
        x_test.loc[x_test[x_test[feature] == val].index, col_name] = 1
    other_col = feature + '_' + 'Other'
    x_test[other_col] = 1
    x_test.loc[x_test[x_test[feature].isin(categories)].index, other_col] = 0
    x_test.drop([feature], axis=1, inplace=True)

    return x_test

def engineering(df):
    '''
    Add features together with similarities
    '''
    pass

def prepare_train_test():
    '''
    Clean and prepare train and test sets.
    '''

    # Import and perform basic cleaning
    x_train = read('nlsy_training_set.csv')
    x_test = read('nlsy_test_set.csv')
    x_train_data = clean_data(x_train)
    x_test_data = clean_data(x_test)
    train = p.drop_nonresponse_y(x_train_data)
    test = x_test_data
    test_ids = test.id.to_list()

    # Step 1: School enrollment variables
    school_enrollment = train.loc[:, 'E5011701':'E5012905']
    school_enrollment_cols = list(school_enrollment.columns)
    for col in school_enrollment_cols:
        train, categories = create_dummies(train, col)
        test = create_dummies_test(test, col, categories)

    # Step 2: School type variables (extract first two digits)
    school_type = train.loc[:, 'E5021701':'E5022903']
    school_type_cols = list(school_type.columns)
    for col in school_type_cols:
        train[col] = train[col].apply(lambda x: (x // 10 **
                                     (int(math.log(x, 10)) - 1)
                                     if x > 0 else x))
        test[col] = test[col].apply(lambda x: (x // 10 **
                                    (int(math.log(x, 10)) - 1)
                                    if x > 0 else x))
        train, categories = create_dummies(train, col)
        test = create_dummies_test(test, col, categories)

    # Step 3: School ID (extract first two digits)
    school_id = train.loc[:, 'E5031701':'E5032903']
    school_id_cols = list(school_id)
    for col in school_id_cols:
        train[col] = train[col].apply(lambda x: (x // 10 **
                                     (int(math.log(x, 10)) - 1)
                                     if x > 0 else x))
        test[col] = test[col].apply(lambda x: (x // 10 **
                                    (int(math.log(x, 10)) - 1)
                                    if x > 0 else x))
        train, categories = create_dummies(train, col)
        test = create_dummies_test(test, col, categories)

    return train, test, test_ids

def go():
    '''
    Run entire model
    '''
    ## Need to go back and continue descretizing nominal variables
    ## Need to go back and engineer new features
    train, test, test_ids = prepare_train_test()
    ## For now, assume these datasets are "ready to go"
