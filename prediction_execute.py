'''
Prediction Challenge Main
'''
import prediction_processing as process
import prediction_loop as loop
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor



def main():

    models = {'Tree': DecisionTreeRegressor(max_depth=10),
              'Lasso': Lasso(alpha=0.1),
              'Ridge': Ridge(alpha=.5),
              'Forest': RandomForestRegressor(max_depth=2)
              }

    parameters_grid = {'Tree': {'max_depth': [10, 20, 50, 100]},#, 20, 30]},
                       'Lasso': {'alpha': [0.01, 0.1, 0.5, 1, 10]},#, 0.075, 0.1]},
                       'Ridge': {'alpha': [0.01, 0.1, 0.5, 1, 10]},#, 0.075, 0.1]},
                       'Forest': {'max_depth': [10, 20, 50, 100]}
                       }

    outcome = 'label'

    train, test, test_ids = process.go()

    best_model = loop.find_best_model(models, parameters_grid, train, outcome)

    #Run predictions on test data and save file
    y_hats = best_model.predict(test)

    results  = pd.DataFrame(list(zip(test_ids, y_hats)),
                            columns =['id', 'y_hat'])

    results.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()