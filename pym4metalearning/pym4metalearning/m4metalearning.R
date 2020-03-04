library(M4metalearning)
library(M4comp2018)
library(tsfeatures)
set.seed(31-05-2018)


train_model <- function(model_path, full, ts_to_add){
    M4_train <- M4
    if (isFALSE(full)){
        indices <- sample(length(M4))
        M4_train <- M4[indices[1:15]]
    }
    if (!missing(ts_to_add)){
        M4_train <- c(M4_train, ts_to_add)
    }

    print(length(M4_train))

    M4_train <- temp_holdout(M4_train)
    M4_train <- calc_forecasts(M4_train, forec_methods(), n.cores=4)
    M4_train <- calc_errors(M4_train)
    M4_train <- THA_features(M4_train)
    train_data <- create_feat_classif_problem(M4_train)
    meta_model <- train_selection_ensemble(train_data$data, train_data$errors)
    save(meta_model, file = model_path)
}


make_prediction <-  function(ts, meta_model, h){
    ts <- list(list(x=ts, h=h))
    ts <- calc_forecasts(ts, forec_methods(), n.cores=1)
    ts <- THA_features(ts, n.cores=1)
    weights <- create_feat_classif_problem(ts)
    weights <- predict_selection_ensemble(meta_model, weights$data)
    ts <- ensemble_forecast(weights, ts)
    ts[[1]]
}