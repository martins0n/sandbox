from etna.datasets.datasets_generation import generate_ar_df, generate_from_patterns_df
from etna.pipeline import Pipeline
from tide_etna import TiDeModel
from etna.transforms import StandardScalerTransform
from etna.datasets import TSDataset
import torch
import numpy as np
import random
from etna.metrics import MSE, MAE, SMAPE


if __name__ == "__main__":
    # set seed
    seed = 11
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    df = generate_from_patterns_df(
        start_time="2021-01-01",
        patterns=[[1, 100, 100, 233, 1], [222, 333, 333, 333, 333, 222]],
        periods=1000
    )
    
    horizon = 7
    lookback = horizon * 5
    ne_blocks = 2
    nd_blocks = 2
    hidden_size = 32
    dropout_level = 0
    covariates_size = 0
    temporal_decoder_hidden_size = 32
    decoder_output_size = horizon * 32
    static_covariates_size = 1
    lr = 0.001
    max_epochs = 1000
    
    trainer_params = {
        'max_epochs': max_epochs
    }
    
    tsdataset = TSDataset.to_dataset(df)
    tsdataset = TSDataset(tsdataset, freq="D")
    pipeline = Pipeline(
        model=TiDeModel(
            encoder_length=lookback,
            decoder_length=horizon,
            lr=lr,
            ne_blocks=ne_blocks,
            nd_blocks=nd_blocks,
            hidden_size=hidden_size,
            covariates_size=covariates_size,
            p=dropout_level,
            lookback=lookback,
            temporal_decoder_hidden_size=temporal_decoder_hidden_size,
            decoder_output_size=decoder_output_size,
            feature_projection_output_size=32,
            feature_projection_hidden_size=32,
            horizon=horizon,
            trainer_params=trainer_params,
            static_covariates_size=static_covariates_size
        ),
        transforms=[StandardScalerTransform()],
        horizon=horizon,
    )
    
    metrics_df, _, _ = pipeline.backtest(
        tsdataset, metrics=[MSE(), MAE(), SMAPE()],
        n_folds=1
    )
    
    print(metrics_df)