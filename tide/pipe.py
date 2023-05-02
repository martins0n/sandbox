from etna.datasets.datasets_generation import generate_ar_df, generate_from_patterns_df
from etna.pipeline import Pipeline
from tide_etna import TiDeModel
from etna.transforms import StandardScalerTransform, TimeFlagsTransform, DateFlagsTransform
from etna.datasets import TSDataset
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import numpy as np
import random
from etna.metrics import MSE, MAE, SMAPE
from dataclasses import dataclass
import hydra
from omegaconf import MISSING
from hydra.utils import instantiate
import pathlib
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from etna.loggers import WandbLogger

from etna.loggers import tslogger

OmegaConf.register_new_resolver('mul', lambda x, y: x * y)


FILE_FOLDER = pathlib.Path(__file__).parent.absolute()


@dataclass
class ModelConfig:
    horizon: int
    lookback: int
    ne_blocks: int
    nd_blocks: int
    hidden_size: int
    dropout_level: float
    covariates_size: int
    temporal_decoder_hidden_size: int
    decoder_output_size: int
    static_covariates_size: int
    lr: float
    max_epochs: int
    feature_projection_output_size: int
    feature_projection_hidden_size: int
    train_batch_size: int
    test_batch_size: int
    train_size: float

@dataclass
class DatasetConfig:
    name: str
    freq: str
    

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    seed: int = 11
    accelerator: str = 'cpu'


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="configs", config_name="config")
def run_pipeline(cfg):
    
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # set seed
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
        
    df = pd.read_parquet(FILE_FOLDER / 'data' / cfg.dataset.name)
    
    tsdataset = TSDataset.to_dataset(df)
    tsdataset = TSDataset(tsdataset, freq=cfg.dataset.freq)
    
    train_dataset, test_dataset = tsdataset.train_test_split(test_size=cfg.model.horizon)
    
    transform = [StandardScalerTransform()]
    train_dataset.fit_transform(transform)
    test_dataset.transform(transform)
    train_dataset = train_dataset.to_pandas()
    test_dataset = test_dataset.to_pandas()
    
    tsdataset = pd.concat([train_dataset, test_dataset])
    
    
    tsdataset = TSDataset(tsdataset, freq=cfg.dataset.freq, known_future='all')
    
    
    
    tslogger.add(WandbLogger(project="tide", config=OmegaConf.to_container(cfg, resolve=True)))
    
    horizon = cfg.model.horizon
    lookback = cfg.model.lookback
    ne_blocks = cfg.model.ne_blocks
    nd_blocks = cfg.model.nd_blocks
    hidden_size = cfg.model.hidden_size
    dropout_level = cfg.model.dropout_level
    covariates_size = cfg.model.covariates_size
    temporal_decoder_hidden_size = cfg.model.temporal_decoder_hidden_size
    decoder_output_size = cfg.model.decoder_output_size
    static_covariates_size = cfg.model.static_covariates_size
    lr =  cfg.model.lr
    max_epochs = cfg.model.max_epochs
    feature_projection_output_size = cfg.model.feature_projection_output_size
    feature_projection_hidden_size = cfg.model.feature_projection_hidden_size
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer_params = {
        'max_epochs': max_epochs,
        'accelerator': cfg.accelerator,
        'callbacks': [lr_monitor],
    }

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
            feature_projection_output_size=feature_projection_output_size,
            feature_projection_hidden_size=feature_projection_hidden_size,
            horizon=horizon,
            trainer_params=trainer_params,
            static_covariates_size=static_covariates_size,
            train_batch_size=cfg.model.train_batch_size,
            test_batch_size=cfg.model.test_batch_size,
            split_params={
                'train_size': cfg.model.train_size,
            }
    
        ),
        transforms=[
            TimeFlagsTransform(minute_in_hour_number=True, hour_number=True, out_column="time"),
            DateFlagsTransform(day_number_in_week=True, day_number_in_month=True, is_weekend=False, out_column="date"),
            StandardScalerTransform(in_column=["time_minute_in_hour_number", "time_hour_number", "date_day_number_in_week", "date_day_number_in_month"])
        ],
        horizon=horizon,
    )
    
    pipeline.transforms
    metrics_df, forecast, _ = pipeline.backtest(
        tsdataset, metrics=[MSE(), MAE(), SMAPE()],
        n_folds=1
    )
        
    print(metrics_df.head())
    print(metrics_df.mean())
    
    
    
    


if __name__ == "__main__":
    
    run_pipeline()