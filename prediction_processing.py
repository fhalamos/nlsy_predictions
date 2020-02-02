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
    df = pd.read_csv(file).head(20)

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
    x_train[feature + '_' + 'Other'] = 0

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


def return_column_names(train, all_variables):
    '''
    Train: training dataframe
    col_list: list of all variables and intervals of interest
    '''
    lst = []
    for var in all_variables:
        if len(var) == 1:
            cols = list(train.loc[:, var].columns)
        else:
            var0 = var[0]
            var1 = var[1]
            cols = list(train.loc[:, var0:var1].columns)
        lst = lst + cols

    return lst


def find_mode_per_year(data, all_variables_list):
    '''
    For each column and year, creates a single column containing the mode
    value for that year (across all months). Drops the original columns,
    and concatenates this information to the new dataframe.
    '''

    new_cols = []
    df = pd.DataFrame()
    # Groups variables by the first 4 characters indicating same question
    d = {}
    for word in all_variables_list:
        d.setdefault(word[:4], []).append(word)
    grouped_list = list(d.values())
    for topic in grouped_list:
        # Group variables by year now that we are looking at a single question
        e = {}
        for word in topic:
            e.setdefault(word[4:6], []).append(word)
        year_grouped_list = list(e.values())
        for year in year_grouped_list:
            temp_data = data[year]
            mode = temp_data.mode(axis=1)[[0]]
            mode = mode.rename(columns = {0: year[0]})
            data = data.drop(columns=year)
            data = pd.concat([data, mode], axis=1)
            new_cols.append(year[0])

    return data, new_cols


def prepare_train_test():
    '''
    Clean and prepare train and test sets.
    '''
    # Import and perform basic cleaning
    print("Reading data...")
    x_train = read('nlsy_training_set.csv')
    x_test = read('nlsy_test_set.csv')

    print("Cleaning...")
    x_train_data = clean_data(x_train)
    x_test_data = clean_data(x_test)
    train = drop_nonresponse_y(x_train_data)
    test = x_test_data
    test_ids = test.id.to_list()

    # Step 1: categorical, no mode, dummies 
    print("Categorical, no mode, dummies")
    all_variables = [a, b, c, d, e] ## this is where we put our list of vars
    all_variables_list = return_column_names(train, all_variables)
    
    for col in all_variables_list:
        train, categories = create_dummies(train, col)
        test = create_dummies_test(test, col, categories)

    # Step 2: categorical, mode, dummies
    all_variables = [['E5011701', 'E5012905'], ['E5031701', 'E5032903']]
    all_variables_list = return_column_names(train, all_variables)
    train, new_cols = find_mode_per_year(train, all_variables_list)

    for col in new_cols:
        train, categories = create_dummies(train, col)
        test = create_dummies_test(test, col, categories)

    '''
    # Step 2: categorical, first two digits, find mode, dummies:
    # This isn't ready because doesn't find the mode

    print("Categorical, no mode, dummies")
    all_variables = [a, b, c, d, e] ## this is where we put our list of vars
    all_variables_list = return_column_names(train, all_variables) 

    for col in school_type_cols:
        train[col] = train[col].apply(lambda x: (x // 10 **
                                     (int(math.log(x, 10)) - 1)
                                     if x > 0 else x))
        test[col] = test[col].apply(lambda x: (x // 10 **
                                    (int(math.log(x, 10)) - 1)
                                    if x > 0 else x))
        train, categories = create_dummies(train, col)
        test = create_dummies_test(test, col, categories)
    '''

    train.drop(columns=['id'], inplace=True)
    test.drop(columns=['id'], inplace=True)

    return train, test, test_ids

def go():
    '''
    Run entire model
    '''
    ## Need to go back and continue descretizing nominal variables
    ## Need to go back and engineer new features
    
    return prepare_train_test()
    ## For now, assume these datasets are "ready to go"
