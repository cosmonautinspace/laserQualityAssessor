from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.base import clone
import numpy as np


def nested_cv(
    estimator,
    X,
    y,
    param_grid,
    inner_cv=5,
    outer_cv=5,
    scoring="accuracy",
    search_type="grid",
    n_iter=50,
    random_state=None,
    return_models=False,
):
    """
    Perform nested cross-validation for sklearn-style models.

    Parameters
    ----------
    estimator : sklearn estimator
        Any estimator following the scikit-learn API.

    X : array-like
        Feature matrix.

    y : array-like
        Target vector.

    param_grid : dict
        Dictionary of hyperparameters to tune.

    inner_cv : int
        Number of folds for inner cross-validation (hyperparameter tuning).

    outer_cv : int
        Number of folds for outer cross-validation (performance estimation).

    scoring : str or callable
        Scoring metric to use.

    search_type : {'grid', 'random'}
        Whether to use GridSearchCV or RandomizedSearchCV.

    n_iter : int
        Number of iterations for RandomizedSearchCV (ignored if search_type='grid').

    random_state : int, optional
        Random seed for reproducibility.

    return_models : bool
        If True, return the trained best estimators from each outer fold.

    Returns
    -------
    results : dict
        Dictionary containing:
            'outer_scores': list of scores for each outer fold
            'mean_score': mean of outer scores
            'std_score': standard deviation of outer scores
            'best_params': list of best hyperparameters from each fold
            'models': list of trained models (if return_models=True)
    """

    outer_cv_splitter = StratifiedKFold(
        n_splits=outer_cv, shuffle=True, random_state=random_state
    )
    outer_scores = []
    best_params = []
    models = []

    for train_idx, test_idx in outer_cv_splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for hyperparameter tuning
        if search_type == "grid":
            search = GridSearchCV(
                estimator=clone(estimator),
                param_grid=param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=-1,
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                estimator=clone(estimator),
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=inner_cv,
                scoring=scoring,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError("search_type must be 'grid' or 'random'")

        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        best_params.append(search.best_params_)

        # Evaluate on outer fold test set
        score = search.score(X_test, y_test)
        outer_scores.append(score)

        if return_models:
            models.append(best_estimator)

    results = {
        "outer_scores": outer_scores,
        "mean_score": np.mean(outer_scores),
        "std_score": np.std(outer_scores),
        "best_params": best_params,
    }
    if return_models:
        results["models"] = models

    return results
