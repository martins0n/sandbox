import random
import numpy as np
import pandas as pd
from itertools import tee
from typing import List, TypeVar, Any
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.datasets import make_moons, make_classification


np.random.seed(1)


def one_hot_encoding(
    categorical_feature: List[str],
    continuous_feature: List[str],
):
    transformer = ColumnTransformer(
        [
            ("cat_vars", OneHotEncoder(), categorical_feature),
            ("cont_vars", StandardScaler(), continuous_feature)
        ]
    )

    return transformer

def ordinal_encoding(
    categorical_feature: List[str],
    continuous_feature: List[str],
):
    transformer = ColumnTransformer(
        [
            ("cat_vars", StandardScaler(), categorical_feature),
            ("cont_vars", StandardScaler(), continuous_feature)
        ]
    )

    return transformer


def cv(
    dataset: pd.DataFrame,
    model: Any,
    transformer: ColumnTransformer,
    target: str
):

    pipe = Pipeline(steps=
        [
            ("transformer", transformer),
            ("model", model())
        ]
    )
#    weight = 1 / dataset.groupby(target).transform('count').id.values
    search = cross_validate(
        pipe, dataset, dataset[target].values,
        n_jobs=6, 
#       fit_params={"model__sample_weight": weight},
        cv=5, scoring=["roc_auc", "accuracy", "f1"],
        return_train_score=True
    )

    return search

def report(
    cv_search: dict,
    name: str
): 
    to_report = f"{name} \n"
    to_report += f"{pd.DataFrame(cv_search).describe().to_markdown()}"
    print(to_report)


def investigation(
    dataset: pd.DataFrame,
    categorical_feature: List[str],
    continuous_feature: List[str],
    model: Any,
    target: str
):
    one_hot = one_hot_encoding(categorical_feature, continuous_feature)
    
    one_hot_cv = cv(dataset, model, one_hot, target)

    report(one_hot_cv, "one_hot_encoding")

    ordinal = ordinal_encoding(categorical_feature, continuous_feature)

    ordinal_cv = cv(dataset, model, ordinal, target)

    report(ordinal_cv, "ordinal_encoding")


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def random_categorizer(x: np.ndarray, number_of_categories: int, to_shuffle=True):
    _max = np.max(x) + 1
    _min = np.min(x) - 1
    intervals = pairwise(
        np.hstack(
            [[_min], np.sort(np.random.choice(x, size=number_of_categories-1, replace=False)), [_max]]
        )
    )
    intervals = list(intervals)
    if to_shuffle:
        random.shuffle(intervals)
    category = np.empty_like(x)
    for i, _x in enumerate(x):
        for j, interval in enumerate(intervals):
            if _x >= interval[0] and _x < interval[1]:
                category[i] = j
    return category


def syntetic_dataset(n_samples=10000, to_categorize=[0,1], to_shuffle=True, number_of_categories=20, *args, **kwargs):
    X, y = make_classification(n_samples=n_samples, **kwargs)
    
    for i in to_categorize:
        X[:, i] = random_categorizer(X[:, i], number_of_categories, to_shuffle=to_shuffle)
    

    return X, y
