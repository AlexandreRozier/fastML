from fastml.sklearn import load_boston
from fastml.sklearn import train_test_split

from fastml.sklearn import cv_regression, get_best_model


def test_cv():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)
    best_models, cv_results = cv_regression(X_train, y_train)
    entry = get_best_model(best_models, cv_results)
    assert entry is not None
