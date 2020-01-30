import pandas as pd
import sklearn
import datetime
import timeit

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2


def find_best_model(models, parameters_grid, data, predictors, outcome_label):

    results_df =  pd.DataFrame(columns=(
      'model_name',
      'parameters',
      'MSE',
      'time_to_run'))

    min_mse = float('inf')
    best_model=""
    best_model_key=""
    best_p=""

    start_time = timeit.default_timer()

    for model_key in models:
        print("Starting "+ model_key+" at "+str(datetime.datetime.now()))

        model = models[model_key]

        parameter_values = parameters_grid[model_key]

        for p in ParameterGrid(parameter_values):

            s = timeit.default_timer()

            model.set_params(**p)

            #Calculate mean square error using cross validation
            #Changing signs because scoring used is neg_mean_squared_error
            scores = -cross_val_score(model, data[predictors], data[outcome_label], cv=5, scoring='neg_mean_squared_error')

            mse = scores.mean()

            time = timeit.default_timer() - s

            results_df.loc[len(results_df)] = [model_key, p, mse, time]

            if(mse < min_mse):
                min_mse = mse
                best_model = model
                best_p = p
                best_model_key = model_key

        elapsed = timeit.default_timer() - start_time


    print("Best Score "+str(min_mse))
    print("Best Model "+str(best_model))
    print('run_models_serial_parameters_parallel time:', elapsed)

    print(results_df)

    #Train best model with all data and return it
    best_model.set_params(**best_p) #Remember to set best parameters again, cause it could be set with newer but not optimal ones
    best_model.fit(data[predictors], data[outcome_label])

    return best_model
 

def main():

    models = {
        'DTR': DecisionTreeRegressor(max_depth=10),
        'LR': Lasso(alpha=0.1),
        'RR': Ridge(alpha=.5),
        'RFR': RandomForestRegressor(max_depth=2)
    }

    parameters_grid = {
        'DTR': { 'max_depth': [10,20]},
        'LR':{ 'alpha':[0.1,0.01]},
        'RR':{ 'alpha':[0.1,0.01]},
        'RFR':{'max_depth':[5,10]}
    }

    training_data = pd.read_csv("nlsy training set")

#>>>>>>> In the meantime, must be removed, using it for iterating fast
    training_data = training_data.head(200) 

    outcome_label = 'U1031900'

    predictors = training_data.columns.to_list()
    predictors.remove(outcome_label)
    predictors.remove('diag.id')
    predictors.remove('Unnamed: 0')

    best_model = find_best_model(models, parameters_grid, training_data, predictors, outcome_label)


    #Run predictions on test data and save file
    test_data = pd.read_csv("nlsy test set")
    diag_ids = test_data['diag.id'].to_list()
    test_data.drop(columns=['diag.id', 'Unnamed: 0'], inplace=True)

    y_hats = best_model.predict(test_data)

    results  = pd.DataFrame(list(zip(diag_ids, y_hats)), columns =['diag_id', 'y_hat']) 
    results.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()