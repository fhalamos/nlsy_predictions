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
    best_model.set_params(**best_p)
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


    data = pd.read_csv("nlsy training set")

    #In the meantime, must be removed, using it for iterating fast
    #>>>>>>>
    data = data.head(200) 


    outcome_label = 'U1031900'

    predictors = data.columns.to_list()
    predictors.remove(outcome_label)
    predictors.remove('diag.id')
    predictors.remove('Unnamed: 0')

    best_model = find_best_model(models, parameters_grid, data, predictors, outcome_label)

    #To Be Done: run_predictions_on_test()

main()


# NOT PAY ATTENTION OF CODE THAT FOLLOWS

# class MachineLearningModel():
#     def __init__(self):
#         self.ml_modelpath = None
#         self.features = None
# #         self.outcome_label = None

#     def features_scores(self, data, k_features):
#         y = np.array(data[self.outcome_label])
#         X = data[self.features]

#         if k_features> len(X.columns):
#             print("K>Number of Features. Passed max columns")
#             k_features = len(X.columns)

#         #apply SelectKBest class to extract top k best features
#         bestfeatures = SelectKBest(score_func=f_classif, k=k_features)
#         fit = bestfeatures.fit(X,y)
#         dfscores = pd.DataFrame(fit.scores_)
#         dfcolumns = pd.DataFrame(X.columns)
#         #concat two dataframes for better visualization
#         featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#         featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#         print('RANKING OF THE MOST IMPORTANT FEATURES. SCORE FUNCTION: f_classif')
#         print(featureScores.nlargest(k_features,'Score'))

# def save_models_files(self, model, auc):
#     # Check ml model number of jobs
#     if (model.n_jobs == None):
#         model.n_jobs = 1

#     if (model and auc>0.6):
#         pickle.dump(model, open(os.path.join(self.ml_modelpath,'ml_model.sav'), 'wb'))
#         print("Parameters saved in: ", os.path.join(self.ml_modelpath,'ml_model.sav'))

# def plot_precision_recall_n(y_true, y_score, model, parameter_values, output_type='save'):
#     '''
#     Plot precision recall curves
#     -y_true contains true values
#     -y_score contains predictions
#     -model is the model being run
#     -parameter_values contains parameters used in this model: we will use this for the plot name
#     -output_type: either saving plot or displaying
#     '''

#     #Compute precision-recall pairs for different probability thresholds
#     precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)


#     #The last precision and recall values are 1. and 0 in precision_recall_curve method, now removing them
#     precision_curve = precision_curve[:-1]
#     recall_curve = recall_curve[:-1]


#     #We transform the pr_thresholds (which is an array with scores thresholds, to an array of percentage thresholds)
#     pct_above_per_thresh = []
#     number_scored = len(y_score)
#     for value in pr_thresholds:
#         num_above_thresh = len(y_score[y_score>=value])
#         pct_above_thresh = num_above_thresh / float(number_scored)
#         pct_above_per_thresh.append(pct_above_thresh)

#     pct_above_per_thresh = np.array(pct_above_per_thresh)

#     #Clear any existing figure
#     plt.clf()

#     #Create a figure and access to its axis
#     fig, ax1 = plt.subplots()

#     #Create blue line for precision curve
#     ax1.plot(pct_above_per_thresh, precision_curve, 'b')
#     ax1.set_xlabel('percent of population')
#     ax1.set_ylabel('precision', color='b')

#     #Create a duplicate axis, and use it to plot recall curve
#     ax2 = ax1.twinx()
#     ax2.plot(pct_above_per_thresh, recall_curve, 'r')
#     ax2.set_ylabel('recall', color='r')

#     #Limit axis borders
#     ax1.set_ylim([0,1])
#     ax1.set_xlim([0,1])
#     ax2.set_ylim([0,1])

#     #Set name of plot
#     model_name = str(model).split('(')[0]
#     chosen_params = str(parameter_values)
#     plot_name = model_name+'-'+chosen_params


#     #Set title and position in plot
#     # title = ax1.set_title(textwrap.fill(plot_name))
#     title = ax1.set_title(plot_name)
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.75)

#     #Save or show plot
#     if (output_type == 'save'):
#         if not os.path.exists('precision-recall-curves'):
#             os.makedirs('precision-recall-curves')
#         plt.savefig('precision-recall-curves/'+str(plot_name)+'.png')
#     elif (output_type == 'show'):
#         plt.show()
#     plt.close()

# def plot_precision_recall_2(y_true, y_hat, y_score, model, parameter_values, output_type='save'):

#     # keep probabilities for the positive outcome only
#     lr_probs = y_score
#     lr_precision, lr_recall, _ = precision_recall_curve(y_true, lr_probs)
#     lr_f1, lr_auc = f1_score(y_true, y_hat), auc(lr_recall, lr_precision)

#     #Set name of plot
#     model_name = str(model).split('(')[0]
#     chosen_params = str(parameter_values)
#     plot_name = model_name+'-'+chosen_params

#     # summarize scores
#     print(plot_name + ' : f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
#     # plot the precision-recall curves
#     no_skill = len(y_true[y_true==1]) / len(y_true)
#     pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
#     pyplot.plot(lr_recall, lr_precision, marker='.', label= plot_name)
#     # axis labels
#     pyplot.xlabel('Recall')
#     pyplot.ylabel('Precision')
#     # show the legend
#     pyplot.legend()

#     #Save or show plot
#     if (output_type == 'save'):
#         if not os.path.exists('precision_recall2-curves'):
#             os.makedirs('precision_recall2-curves')
#         pyplot.savefig('precision_recall2-curves/'+str(plot_name)+'.png')
#     elif (output_type == 'show'):
#         pyplot.show()
#     pyplot.close()


# def plot_roc_auc_curve(y_true, y_score, model, parameter_values, output_type='save'):
#     # generate a no skill prediction (majority class)
#     ns_probs = [0 for _ in range(len(y_true))]

#     # predicted probabilities
#     lr_probs = y_score

#     # calculate scores
#     ns_auc = roc_auc_score(y_true, ns_probs)
#     lr_auc = roc_auc_score(y_true, lr_probs)

#     model_name = str(model).split('(')[0]
#     chosen_params = str(parameter_values)
#     plot_name = model_name+'-'+chosen_params

#     # summarize scores
#     print('No Skill: ROC AUC=%.3f' % (ns_auc))
#     print(plot_name + ' : ROC AUC=%.3f' % (lr_auc))

#     # calculate roc curves
#     ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
#     lr_fpr, lr_tpr, _ = roc_curve(y_true, lr_probs)

#     # plot the roc curve for the model
#     pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
#     pyplot.plot(lr_fpr, lr_tpr, marker='.', label= plot_name)
#     # axis labels
#     pyplot.xlabel('False Positive Rate')
#     pyplot.ylabel('True Positive Rate')
#     # show the legend
#     pyplot.legend()

#     #Save or show plot
#     if (output_type == 'save'):
#         if not os.path.exists('roc-auc-curves'):
#             os.makedirs('roc-auc-curves')
#         pyplot.savefig('roc-auc-curves/'+str(plot_name)+'.png')
#     elif (output_type == 'show'):
#         pyplot.show()
#     pyplot.close()



# def plot_relevant_curves(self, model_features, best_model, best_model_key, best_p):
#     # Split in Train and Test data to fit the model
#     X_train, X_test, y_train, y_test = train_test_split(model_features[self.features], model_features[self.outcome_label], test_size=0.30)
#     best_model.fit(X_train, y_train)

#     y_pred_scores = best_model.predict_proba(X_test)[:,1]
#     # predict class values
#     y_pred_class = best_model.predict(X_test)

#     # Plot the ROC curve and AUC
#     plot_roc_auc_curve(y_test, y_pred_scores, best_model_key, best_p, output_type='show')

#     # Plot the precision recall curve
#     plot_precision_recall_2(y_test, y_pred_class, y_pred_scores, best_model_key, best_p, output_type='show')
#     plot_precision_recall_n(y_test, y_pred_scores, best_model_key, best_p, output_type='show')

# def features_importance(self, model):
#     important_features_dict = {}
#     for x,i in enumerate(model.feature_importances_):
#         important_features_dict[x]=i

#     important_features_list = sorted(important_features_dict,
#                                      key=important_features_dict.get,
#                                      reverse=True)
#     print(important_features_list)