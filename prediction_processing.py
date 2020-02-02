'''
Prediction Challenge Processing steps
'''
import pandas as pd
import numpy as np
import math

import yaml


def read(file):
    '''
    Read in training data.
    '''
    df = pd.read_csv(file).head(200)

    return df

def format_id_and_label_columns(dataframe):
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

def drop_useless_columns(df,columns_to_drop):

    return df.drop(columns_to_drop, axis=1)

# def find_mode_per_year(df, columns_to_aggregate):
#     for col in columns_to_aggregate:

def get_range_columns_for_features(columns_category):
    config = yaml.load(open('config.yaml', 'r'))#, Loader=yaml.SafeLoader)

    list_to_return = []

    for c in config[columns_category]:
        list_to_return.append(config[columns_category][c])

    return list_to_return
    # return config[columns_category]


def expand_column_names(train, all_variables):
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

def discretize_columns(train_data, test_data, columns_to_discretize):

    for c in columns_to_discretize:
        
        print(train_data[c])

        #We use qcut instead of cut because of outliers (if not, in case of high outlieres, almost all datapoints end up in lowest bin and none in the middle ones)
        train_data[c+'_discrete'], bins = pd.qcut(train_data[c], 5, labels=['low', 'medium_low', 'medium', 'medium_high', 'high'], retbins=True, duplicates='drop')

        #Use same bins of train to discretize on test
        test_data[c+'_discrete'] = pd.cut(test_data[c], bins=bins, labels=['low', 'medium_low', 'medium', 'medium_high', 'high'], include_lowest=True)

    return train_data, test_data

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
    train_data = read('nlsy_training_set.csv')
    test_data = read('nlsy_test_set.csv')

    print("Cleaning...")
    train_data = format_id_and_label_columns(train_data)
    test_data = format_id_and_label_columns(test_data)

    train_data = drop_nonresponse_y(train_data)
    test_ids = test_data.id.to_list()


    #Drop useless columns
    range_columns_to_drop = get_range_columns_for_features('to_drop')
    columns_to_drop = expand_column_names(train_data, range_columns_to_drop)

    train_data = drop_useless_columns(train_data,columns_to_drop)
    test_data = drop_useless_columns(test_data,columns_to_drop)

    #Convert all negatives values to -1, np.nan
    train_data.replace([-1,-2,-3,-4,-5], -1, inplace=True)

    #Discretize continuous variables
    range_columns_to_discretize = get_range_columns_for_features('pure_continuous')
    columns_to_discretize = expand_column_names(train_data, range_columns_to_discretize)

    # train_data, test_data = discretize_columns(train_data, test_data, columns_to_discretize)
    


    # Step 1: categorical, no mode, dummies 
    print("Categorical, no mode, dummies")
    range_columns_for_features = get_range_columns_for_features('pure_categorical')#[a, b, c, d, e] ## this is where we put our list of vars

    columns_for_features = expand_column_names(train_data, range_columns_for_features)

    for col in columns_for_features:
        train_data, categories = create_dummies(train_data, col)
        test_data = create_dummies_test(test_data, col, categories)

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

    train_data.drop(columns=['id'], inplace=True)
    test_data.drop(columns=['id'], inplace=True)

    return train_data, test_data, test_ids

def go():
    '''
    Run entire model
    '''
    ## Need to go back and continue descretizing nominal variables
    ## Need to go back and engineer new features

    return prepare_train_test()
    ## For now, assume these datasets are "ready to go"
