import pandas as pd
from scipy.stats import loguniform
from fastml.sklearn import RandomizedSearchCV
from fastml.sklearn import KNeighborsRegressor
from fastml.sklearn import SVR

from xgboost import XGBRegressor

DEFAULT_REGRESSION_MODELS = [
          ('KNN', KNeighborsRegressor(),dict(n_neighbors=[2,4,8,16,32,64,128,256])),
          ('SVR', SVR(), dict(
              C=loguniform(1e-4, 1e4),
              gamma=loguniform(1e-4, 1e4))),
          ('XGB', XGBRegressor(),dict(n_estimators=[100, 1000]))
        ]

# cv_reg
# entry = get_best_model(cv_results)
# best_model = best_models[entry.model]

def cv_regression(X_train: pd.DataFrame , y_train: pd.DataFrame, models=DEFAULT_REGRESSION_MODELS):
    '''
    Lightweight script to test many models and find winners:param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: DataFrame of predictions
    '''

    dfs = []

    best_models = dict()
    for name, model, distributions in models:
        random_search = RandomizedSearchCV(model,
                                           distributions,
                                           n_iter=20,
                                           refit=True,
                                           n_jobs=None,
                                           random_state=666,
                                           scoring='neg_mean_squared_error')
        random_search.fit(X_train, y_train)

        this_df = pd.DataFrame(random_search.cv_results_)
        this_df['model'] = name
        best_models[name] = random_search.best_estimator_
        dfs.append(this_df)
    return best_models, pd.concat(dfs, ignore_index=True)

def get_best_model(best_models, cv_results):
    idx = cv_results['mean_test_score'].idxmax()
    best_entry = cv_results.loc[idx].dropna()
    best_model = best_models[best_entry.model]
    return best_model
