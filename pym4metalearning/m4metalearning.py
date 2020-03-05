import os
from pkg_resources import resource_filename
from typing import List, Optional
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects.vectors import IntVector, FloatVector, ListVector, BoolVector


m4metalearning_r_script_path = resource_filename("pym4metalearning", "m4metalearning.R")


def pd_series_to_ts(ts: pd.Series):
    """
    
    Series start is provided in year format for Yearly, Monthly and Quarterly series.
    For Daily and Hourly series, it is given in the number of days after 1970-01-01 and for Weekly data
    it is given in number of weeks since 1970-01-01.

    https://github.com/carlanetto/M4comp2018/blob/master/man/M4.Rd
    
    """
    freq = ts.index.freq
    rstats = importr("stats")

    ts = ts.astype("float")
    start_year = ts.index[0].year
    start_month = ts.index[0].month
    end_year = ts.index[-1].year
    end_month = ts.index[-1].month

    if ts.index.freq == "M":
        return rstats.ts(
            FloatVector(ts.values),
            start=IntVector((start_year, start_month)),
            end=IntVector((end_year, end_month)),
            frequency=12,
        )

    elif ts.index.freq == "Y":
        return rstats.ts(
            FloatVector(ts.values),
            start=IntVector((start_year,)),
            end=IntVector((end_year,)),
            frequency=1,
        )

    elif ts.index.freq == "Q":
        return rstats.ts(
            FloatVector(ts.values),
            start=IntVector((start_year, start_month)),
            end=IntVector((end_year, end_month)),
            frequency=4,
        )

    elif ts.index.freq == "W":
        start_index = (
            (ts.index[0] - pd.to_datetime("1970-01-01")).total_seconds()
            / 3600
            / 24
            // 7
        )
        end_index = (
            (ts.index[-1] - pd.to_datetime("1970-01-01")).total_seconds()
            / 3600
            / 24
            // 7
        )
        return rstats.ts(
            FloatVector(ts.values),
            start=IntVector((start_index)),
            end=IntVector((end_index)),
            frequency=1,
        )

    elif ts.index.freq == "H":
        start_index = (
            ts.index[0] - pd.to_datetime("1970-01-01")
        ).total_seconds() // 3600
        end_index = (
            ts.index[-1] - pd.to_datetime("1970-01-01")
        ).total_seconds() // 3600
        return rstats.ts(
            FloatVector(ts.values),
            start=IntVector((start_index,)),
            end=IntVector((end_index,)),
            frequency=1,
        )

    elif ts.index.freq == "D":
        start_index = (ts.index[0] - pd.to_datetime("1970-01-01")).total_seconds() // (
            3600 * 24
        )
        end_index = (ts.index[-1] - pd.to_datetime("1970-01-01")).total_seconds() // (
            3600 * 24
        )
        return rstats.ts(
            FloatVector(ts.values),
            start=IntVector((start_index,)),
            end=IntVector((end_index,)),
            frequency=1,
        )
    else:
        raise NotImplementedError


def m4meta_train(model_path: str, full_train: bool = False, ts_to_add_to_train: Optional[List[pd.Series]] = None):
    r(f"""source('{m4metalearning_r_script_path}')""")
    train_model_func = r["train_model"]
    if ts_to_add_to_train is None:
        train_model_func(model_path, BoolVector([full_train]))
    else:
        ts_to_add_to_train = map(pd_series_to_ts, ts_to_add_to_train)
        ts_to_add_to_train = ListVector(ts_to_add_to_train)
        train_model_func(model_path, BoolVector([full_train]), ts_to_add_to_train)


def load_model(model_path: str):
    r["load"](model_path)
    model = r["meta_model"]
    return model


def make_prediction(ts: pd.Series, model_path: str, h: int = 6, full: bool = False):

    meta_model = load_model(model_path)

    ts_r = pd_series_to_ts(ts)

    ts_freq = ts.index[-1] - ts.index[-2]
    ts_end = ts.index[-1]

    r(f"""source('{m4metalearning_r_script_path}')""")

    r_make_prediction = r["make_prediction"]
    r_make_prediction = r_make_prediction(ts_r, meta_model, h)

    if full:
        return r_make_prediction
    else:
        return pd.Series(
            np.array(r_make_prediction.rx2("y_hat")),
            index=pd.date_range(
                start=ts_end + pd.to_timedelta(ts_freq), freq=ts_freq, periods=h
            ),
        )
    return r_make_prediction
