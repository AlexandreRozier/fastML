from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split

from src.warML.sklearn.utils import RandomizedCV, get_best_model, DEFAULT_CLASSIFICATION_MODELS, DEFAULT_REGRESSION_MODELS


def test_sklearn_reg():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)
    out =  RandomizedCV(X_train, y_train,
                        models=DEFAULT_REGRESSION_MODELS,
                        refit=True,
                        scoring='neg_mean_squared_error',
                        return_preds=True)

    entry = get_best_model(out, metric='score')
    assert entry is not None
    preds = out.preds
    assert len(preds) == len(DEFAULT_REGRESSION_MODELS)


def test_sklearn_classif():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)
    out = RandomizedCV(X_train, y_train, models=DEFAULT_CLASSIFICATION_MODELS,
                       scoring=['f1_weighted', 'f1_micro', 'f1_macro'],
                       refit='f1_micro',
                        return_preds=True)
    entry = get_best_model(out, metric='f1_micro')
    assert entry is not None
    preds = out.preds
    assert len(preds) == len(DEFAULT_CLASSIFICATION_MODELS)