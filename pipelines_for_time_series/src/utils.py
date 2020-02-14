import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from loguru import logger
from typing import Callable, List, Dict, Tuple, Union


def get_lags(df, lag=1):
    try:
        return df.iloc[-1 - lag]
    except:
        return np.nan


def get_lags_weekly(df, lag=1):
    try:
        return df.iloc[-1 - lag * 7]
    except:
        return np.nan


def get_weekly_aggregated_lags(df, lag=0, aggregation=np.median):
    weekday = df.index[-1].weekday()
    ar_len = df.shape[0]
    try:
        # fmt: off
        if lag == 0:
            return np.apply_along_axis(aggregation, 0, df.iloc[ar_len - weekday - 1: ar_len].values) 
        else:
            return np.apply_along_axis(aggregation, 0, df.iloc[ar_len - 1 - weekday - (lag+1)*7 : ar_len - weekday - lag*7 - 1])
        # fmt: on
    except:
        return np.nan


def get_weekly_aggregated_lags_relation(
    df, numerator=0, denominator=1, aggregation=np.median
):
    weekday = df.index[-1].weekday()
    ar_len = df.shape[0]
    try:
        # fmt: off
        if numerator == 0:
            return (
                np.apply_along_axis(aggregation, 0, df.iloc[ar_len - weekday - 1:ar_len].values) /  
                np.apply_along_axis(aggregation, 0, df.iloc[ar_len - 1 - weekday - (denominator+1)*7 : ar_len - weekday - denominator*7 - 1])
            )
        # fmt: on
        else:
        # fmt: off
            return (
                np.apply_along_axis(aggregation, 0, df.iloc[ar_len - 1 - weekday - (numerator+1)*7 : ar_len - weekday - numerator*7 - 1]) / 
                np.apply_along_axis(aggregation, 0, df.iloc[ar_len - 1 - weekday - (denominator+1)*7 : ar_len - weekday - denominator*7 - 1])
            )
        # fmt: on
    except:
        return np.nan


def get_cat_holiday(df, lag=0):
    try:
        return holiday_check(df.index[-1 - lag])
    except:
        return np.nan


def get_cat_weekday(df, lag=0):
    try:
        return df.index[-1 - lag].weekday()
    except:
        return np.nan


def get_sin_month(df, lag=0):
    try:
        return np.sin((df.index[-1 - lag].month) / 12 * np.pi)
    except:
        return np.nan


def get_sin_day(df, lag=0):
    try:
        return np.day((df.index[-1 - lag].day) / 31 * np.pi)
    except:
        return np.nan


def get_dummy_day_of_week(df, weekday=0, lag=0):
    return int(df.index[-1 - 0].weekday() == weekday)


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
