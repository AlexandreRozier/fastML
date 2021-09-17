from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split

from warML.sklearn.utils import RandomizedCV, get_best_model, DEFAULT_CLASSIFICATION_MODELS


def test_sklearn_reg():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)
    best_models, cv_results = RandomizedCV(X_train, y_train, scoring='neg_mean_squared_error')
    entry = get_best_model(best_models, cv_results)
    assert entry is not None

def test_sklearn_classif():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)
    best_models, cv_results = RandomizedCV(X_train, y_train, models=DEFAULT_CLASSIFICATION_MODELS, scoring=['f1_weighted', 'f1_micro', 'f1_macro'])
    entry = get_best_model(best_models, cv_results)
    assert entry is not None
