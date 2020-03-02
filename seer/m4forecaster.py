import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects.vectors import IntVector, FloatVector, ListVector


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


"""
def forecast(ts: pd.Series):
    ts = pd_series_to_ts(ts)
    _forecast = importr('forecast')
    return _forecast.Arima(ts, order=IntVector((2,0,0)))
"""


def m4meta_train(model_path):
    r_string = """
    train_model_func <- function(name){
        library(M4metalearning)
        library(M4comp2018)
        library(tsfeatures)
        set.seed(31-05-2018)
        M4_train <- M4
        M4_train <- temp_holdout(M4_train)
        M4_train <- calc_forecasts(M4_train, forec_methods(), n.cores=4)
        M4_train <- calc_errors(M4_train)
        M4_train <- THA_features(M4_train)
        train_data <- create_feat_classif_problem(M4_train)
        meta_model <- train_selection_ensemble(train_data$data, train_data$errors)
        save(meta_model, file = name)
    }
    """
    train_model_func = r(r_string)
    train_model_func(model_path)


def load_model(str_name="temp.Rdata"):
    r["load"](str_name)
    model = r["meta_model"]
    return model


def make_prediction(ts: pd.Series, model_path, h=6, full=False):

    meta_model = load_model(model_path)

    ts_r = pd_series_to_ts(ts)

    ts_freq = ts.index[-1] - ts.index[-2]
    ts_end = ts.index[-1]

    r_string = """
    function(ts, meta_model, h){
        library(M4metalearning)
        library(M4comp2018)
        library(tsfeatures)
        ts <- list(list(x=ts, h=h))
        ts <- calc_forecasts(ts, forec_methods(), n.cores=1)
        ts <- THA_features(ts, n.cores=1)
        weights <- create_feat_classif_problem(ts)
        weights <- predict_selection_ensemble(meta_model, weights$data)
        ts <- ensemble_forecast(weights, ts)
        return(ts[[1]])
    }
    """
    r_forecast = r(r_string)
    r_forecast = r_forecast(ts_r, meta_model, h)
    if full:
        return r_forecast
    else:
        return pd.Series(
            np.array(r_forecast.rx2("y_hat")),
            index=pd.date_range(
                start=ts_end + pd.to_timedelta(ts_freq), freq=ts_freq, periods=h
            ),
        )
    return r_forecast
