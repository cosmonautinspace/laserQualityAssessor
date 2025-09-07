from sklearn.model_selection import (
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.base import clone
import numpy as np


def nested_cv_multi(
    estimator,
    X,
    y,
    param_grid,
    inner_cv=5,
    outer_cv=5,
    scoring="accuracy",
    random_state=42,
    return_models=False,
):
    """
    Perform nested cross-validation for sklearn-style models.

    :param estimator: Any estimator following the scikit-learn API.
    :param X: Feature matrix (n_samples, n_features).
    :param y: Target vector (n_samples,).
    :param param_grid: Dictionary of hyperparameters to tune.
    :param inner_cv: Number of folds for inner CV (hyperparameter tuning).
    :param outer_cv: Number of folds for outer CV (performance estimation).
    :param scoring: Scoring strategy (str, callable, or dict).
    :param random_state: Random seed.
    :param return_models: If True, return the trained best estimators.
    :returns: dict with outer fold scores, mean/std per metric, best params per fold, etc.
    """

    outer_cv_splitter = StratifiedKFold(
        n_splits=outer_cv, shuffle=True, random_state=random_state
    )

    outer_scores = []  # list of dicts (per fold)
    best_params = []
    models = []

    # Decide what to use for refit
    if isinstance(scoring, dict):
        # pick the first key as primary refit metric
        primary_scorer = next(iter(scoring.keys()))
    else:
        primary_scorer = True  # default behavior if single scorer

    for train_idx, test_idx in outer_cv_splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
            refit=primary_scorer,
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        best_params.append(search.best_params_)

        # Evaluate all scorers on outer test
        fold_scores = {}
        if isinstance(scoring, dict):
            for name, scorer in scoring.items():
                scorer_func = get_scorer(scorer) if isinstance(scorer, str) else scorer
                fold_scores[name] = scorer_func(best_estimator, X_test, y_test)
        else:
            # single scorer string or callable
            scorer_func = get_scorer(scoring) if isinstance(scoring, str) else scoring
            fold_scores[scoring if isinstance(scoring, str) else "score"] = scorer_func(
                best_estimator, X_test, y_test
            )

        outer_scores.append(fold_scores)

        if return_models:
            models.append(best_estimator)

    # Run a final search on all data (optional "star search")
    star_search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=-1,
        refit=primary_scorer,
    )
    star_search.fit(X, y)
    star_params = star_search.best_params_

    # Aggregate results
    metrics = outer_scores[0].keys()
    mean_scores = {m: np.mean([fs[m] for fs in outer_scores]) for m in metrics}
    std_scores = {m: np.std([fs[m] for fs in outer_scores]) for m in metrics}

    results = {
        "outer_scores": outer_scores,  # list of dicts, one per fold
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "best_params": best_params,
        "star_params": star_params,
    }
    if return_models:
        results["models"] = models

    return results


# NOTE no longer in use, but keeping it, since the rewrite hasn't been tested
def nested_cv(
    estimator,
    X,
    y,
    param_grid,
    inner_cv=5,
    outer_cv=5,
    scoring="accuracy",
    random_state=42,
    return_models=False,
):
    """
    Perform nested cross-validation for sklearn-style models.

    :param estimator: Any estimator following the scikit-learn API.
    :type estimator: sklearn.base.BaseEstimator

    :param X: Feature matrix.
    :type X: array-like of shape (n_samples, n_features)

    :param y: Target vector.
    :type y: array-like of shape (n_samples,)

    :param param_grid: Dictionary of hyperparameters to tune.
    :type param_grid: dict

    :param inner_cv: Number of folds for inner cross-validation (hyperparameter tuning).
    :type inner_cv: int

    :param outer_cv: Number of folds for outer cross-validation (performance estimation).
    :type outer_cv: int

    :param scoring: Scoring strategy. If multiple are provided, the first scorer in the
                    dictionary will be used for refit.
    :type scoring: str or callable or dict

    :param search_type: Whether to use ``GridSearchCV`` or ``RandomizedSearchCV``.
    :type search_type: {'grid', 'random'}

    :param n_iter: Number of iterations for ``RandomizedSearchCV`` (ignored if
                ``search_type='grid'``).
    :type n_iter: int

    :param random_state: Random seed for reproducibility.
    :type random_state: int, optional

    :param return_models: If ``True``, return the trained best estimators from each outer fold.
    :type return_models: bool

    :returns: Dictionary containing:
            - **outer_scores** (*list*): scores for each outer fold
            - **mean_score** (*float*): mean of outer scores
            - **std_score** (*float*): standard deviation of outer scores
            - **best_params** (*list*): best hyperparameters from each fold
            - **models** (*list*): trained models (if ``return_models=True``)
    :rtype: dict
    """

    outer_cv_splitter = StratifiedKFold(
        n_splits=outer_cv, shuffle=True, random_state=random_state
    )
    outer_scores = []
    best_params = []
    models = []

    if type(scoring) is dict:
        for element in scoring:
            primary_scorer = str(element)
            break
    else:
        primary_scorer = True

    for train_idx, test_idx in outer_cv_splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
            refit=primary_scorer,
        )

        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        best_params.append(search.best_params_)

        # Evaluate on outer fold test set
        score = search.score(X_test, y_test)
        outer_scores.append(score)

        if return_models:
            models.append(best_estimator)

    star_search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=-1,
        refit=primary_scorer,
    )

    star_search.fit(X, y)
    star_params = star_search.best_params_
    results = {
        "mean_score": np.mean(outer_scores),
        "star_params": star_params,
        "outer_scores": outer_scores,
        "std_score": np.std(outer_scores),
        "best_params": best_params,
    }
    if return_models:
        results["models"] = models

    return results


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import get_scorer
import numpy as np
