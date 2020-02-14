import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.linear_model import LinearRegression
from loguru import logger
from typing import Callable, List, Dict, Tuple, Union


class MakingRowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features_fn, forecast=None, shift=30):
        self.features_fn = features_fn
        self.X: pd.Series = None
        self.shift = shift
        self.forecast = forecast

    def fit(self, X: pd.Series, y=None):
        return self

    def transform(self, X: pd.Series):
        if self.X is None:
            x_array = np.empty(shape=(len(X), len(self.features_fn)))
            self.X = X
            for i in range(len(X)):
                x_array[i] = window(X.iloc[: i + 1], self.features_fn)
            return x_array[self.shift :]
        elif self.forecast is None:
            current_lenght = len(X)
            x_array = np.empty(shape=(len(X), len(self.features_fn)))
            X = self.X.append(X)
            for i in range(current_lenght):
                x_array[i] = window(X.iloc[: len(self.X) + i + 1], self.features_fn)
            return x_array
        else:
            current_lenght = len(X)
            x_array = np.empty(shape=(len(X), len(self.features_fn)))
            X = self.X.append(self.forecast).append(X)
            for i in range(current_lenght):
                x_array[i] = window(
                    X.iloc[: len(self.X) + len(self.forecast) + i + 1], self.features_fn
                )
            return x_array


def making_dataset(df: pd.Series, fns: List[Union[Callable, Tuple]], shift=0):
    x_array = np.empty(shape=(len(df), len(fns)))
    for i in range(len(df)):
        x_array[i] = window(df.iloc[: i + 1], fns)
    return x_array[shift:], df.iloc[shift:].values.flatten()


def window(df: pd.Series, fns: List[Union[Callable, Tuple]]):
    array_result = np.empty(shape=(len(fns),))
    for i, fn in enumerate(fns):
        array_result[i] = fn[0](df, *fn[1])
    return array_result


class ForecastPipeline(Pipeline):
    def forecast(self, X, horizon=20):
        step_size = X.index[-1] - X.index[-2]
        time_range = pd.date_range(
            X.index[-1] + step_size, periods=horizon, name="time", freq=step_size
        )
        forecast_ts = pd.Series(np.zeros(horizon), index=time_range, name="y")
        forecast_ts.iloc[0] = self.predict(forecast_ts.iloc[:1])[0]
        for i in range(1, horizon):
            self.set_params(MakingRowTransformer__forecast=forecast_ts.iloc[:i])
            forecast_ts.iloc[i] = self.predict(forecast_ts.iloc[i : i + 1])
            self.set_params(MakingRowTransformer__forecast=None)
        return forecast_ts
