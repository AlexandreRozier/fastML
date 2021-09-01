import pandas as pd
from scipy.stats import loguniform

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from tqdm.auto import tqdm
from xgboost import XGBRegressor, XGBClassifier

DEFAULT_REGRESSION_MODELS = [
    ('KNN', KNeighborsRegressor(), dict(n_neighbors=[2, 4, 8, 16, 32, 64, 128, 256])),
    ('SVR', SVR(), dict(
        C=loguniform(1e-4, 1e4),
        gamma=loguniform(1e-4, 1e4))),
    ('XGB', XGBRegressor(), dict(n_estimators=[100, 1000]))
]

DEFAULT_CLASSIFICATION_MODELS = [
    ('LogReg', LogisticRegression(), dict(C=loguniform(1e-4, 1e4))),
    ('RF', RandomForestClassifier(), dict(n_estimators=[100, 1000])),
    ('KNN', KNeighborsClassifier(), dict(leaf_size=[15, 30])),
    ('SVM', SVC(), dict(
        C=loguniform(1e-4, 1e4),
        gamma=loguniform(1e-4, 1e4))),
    ('GNB', GaussianNB(), dict(var_smoothing=[1e-9])),
    ('XGB', XGBClassifier(), dict(n_estimators=[100, 1000]))
]


def cv_regression(X_train: pd.DataFrame, y_train: pd.DataFrame, models=DEFAULT_REGRESSION_MODELS, **kwargs):
    dfs = []

    best_models = dict()
    for name, model, distributions in tqdm(models):
        random_search = RandomizedSearchCV(model,
                                           distributions,
                                           n_iter=20,
                                           refit=True,
                                           n_jobs=None,
                                           scoring=None,
                                           random_state=666,
                                           **kwargs)
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
