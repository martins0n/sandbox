import numpy as np
import pandas as pd
from typing import List, TypeVar, Any
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate


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

    search = cross_validate(
        pipe, dataset, dataset[target].values,
        n_jobs=6, 
        fit_params={"sample_weight": [1/dataset[target].value_counts()[0], 1/dataset[target].value_counts()[1]]},
        cv=5, scoring=["f1_macro", "accuracy", "f1", "precision", "recall"]
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
