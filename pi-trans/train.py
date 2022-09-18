from etna.analysis import plot_backtest
from etna.models.base import DeepBaseNet, DeepBaseModel
from etna.models.nn.rnn import RNNNet
from pi_trans import PITransformer, mask_gen
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pathlib
import random

import numpy as np
import pandas as pd
import torch
import typer

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


from etna.datasets.tsdataset import TSDataset
from etna.loggers import LocalFileLogger, WandbLogger, tslogger
from etna.metrics import MAE, MSE, SMAPE, Sign
from etna.models.nn import RNNModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform, StandardScalerTransform
from etna.datasets import generate_periodic_df, generate_from_patterns_df

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

from typing import Iterator, Optional, Dict, Any

class MASELoss(nn.Module):
    def __init__(self, seasonality: int = 7):
        super().__init__()
        self.seasonality = seasonality
    
    def forward(self, input, target, in_sample):
        sample_mean = (target - input).abs().mean(dim=1)
        seasonal_sample_mean = (in_sample[:, self.seasonality:] - in_sample[:, :-self.seasonality]).abs().mean(dim=1)
        return (sample_mean / seasonal_sample_mean).mean()
        



class PITransformerNet(RNNNet, DeepBaseNet):
    def __init__(self, lr, loss, d_model: int = 512, d_ff: int = 2048, num_heads: int =  4, n_layers: int = 4, optimizer_params = None):
        super(DeepBaseNet, self).__init__()
        self.pi_trans = PITransformer(d_model=d_model, d_ff=d_ff, num_heads = num_heads, n_layers=n_layers)
        self.num_heads = num_heads
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params

    def forward(self, x):
        encoder_real = x["encoder_real"].float()  # (batch_size, encoder_length-1, input_size)
        decoder_target = x["decoder_target"].float()  # (batch_size, decoder_length, 1)
        decoder_real = x["decoder_real"].float()
        decoder_length = decoder_target.shape[1]
        forecast = torch.zeros_like(decoder_target)  # (batch_size, decoder_length, 1)
        mask = mask_gen(encoder_real, self.num_heads)
        input_ = torch.cat([encoder_real, decoder_real[:, :1]], dim=1)  
        forecast[:, 0, :] = self.pi_trans(input_, shift=0, window_size=decoder_length, mask=mask)[:, -1 , :]
        for t in range(1, decoder_length):
            input_ = torch.cat([encoder_real, decoder_real[:, :1], forecast[:, :t, :]], dim=1)
            mask = mask_gen(input_, self.num_heads)
            forecast[:, t, :] = self.pi_trans(input_, shift=t, window_size=decoder_length, mask=mask)[:, -1 , :]
        return forecast

    def step(self, batch, *args, **kwargs):
        decoder_real = batch["decoder_real"].float()
        encoder_real = batch["encoder_real"].float()
        decoder_target = batch["decoder_target"].float()
        encoder_target = batch["encoder_target"].float()
        input_ = torch.cat([encoder_real, decoder_real], dim=1)
        mask = mask_gen(input_, self.num_heads) 
        output = self.pi_trans(input_, shift=decoder_real.shape[1], window_size=decoder_real.shape[1], mask=mask)[:, -decoder_target.shape[1]:, :]
        # loss = self.loss(output, decoder_target, encoder_target)
        loss = self.loss(output, decoder_target)
        return loss, decoder_target, output
    def training_step(self, batch: dict, *args, **kwargs):  # type: ignore
        """Training step.

        Parameters
        ----------
        batch:
            batch of data

        Returns
        -------
        :
            loss
        """
        loss, a, b = self.step(batch, *args, **kwargs)  # type: ignore
        if loss.item() < 2:
            print(2)
        self.log("train_loss", loss, on_epoch=True)
        return loss
class PITransformerModel(DeepBaseModel):

    def __init__(
        self,
        decoder_length: int,
        encoder_length: int,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int =  4,
        n_layers: int = 4,
        lr: float = 1e-3,
        loss: Optional["torch.nn.Module"] = None,
        train_batch_size: int = 16,
        test_batch_size: int = 16,
        optimizer_params: Optional[dict] = None,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
    ):
        super().__init__(
            net=PITransformerNet(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                n_layers=n_layers,
                lr=lr,
                loss=nn.MSELoss() if loss is None else loss,
                optimizer_params= dict() if optimizer_params is None else optimizer_params,
            ),
            decoder_length=decoder_length,
            encoder_length=encoder_length,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_params=train_dataloader_params,
            test_dataloader_params=test_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            trainer_params=trainer_params,
            split_params=split_params,
        )

def train_backtest(
    horizon: int = 7,
    n_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 64,
    seed: int = 11,
    dataset_path: str = "/Users/marti/Projects/etna/examples/data/example_dataset.csv",
    experiments_folder: pathlib.Path = pathlib.Path("experiments"),
    dataset_freq: str = "D",
):
    parameters = dict(locals())
    parameters["dataset_path"] = str(dataset_path)
    parameters["experiments_folder"] = str(experiments_folder)

    set_seed(seed)

    original_df = pd.read_csv(dataset_path)
    #original_df = generate_from_patterns_df(
    #    periods=600, start_time="2020-01-01", patterns=[[30, 50, 50, 100, 100]], freq='D', add_noise=True,  sigma=1,
    #)
    df = TSDataset.to_dataset(original_df)
    ts = TSDataset(df, freq=dataset_freq)


    model = PITransformerModel(
        n_layers=4,
        # loss = MASELoss(7),
        d_model=32,
        num_heads=4,
        d_ff=128,
        loss=nn.L1Loss(),
        decoder_length=horizon,
        encoder_length=2 * horizon + 1,
        trainer_params={"max_epochs": n_epochs},
        lr=lr,
        train_batch_size=batch_size,
    )
    num_lags = 14
    # model = RNNModel(
    #     decoder_length=horizon,
    #     encoder_length=2 * horizon,
    #     input_size=1,
    #     trainer_params={"max_epochs": n_epochs},
    #     loss=nn.L1Loss(),
    #     lr=lr,
    #     train_batch_size=batch_size,
    # )
    # transform_lag = LagTransform(
    #     in_column="target",
    #     lags=[horizon + i for i in range(num_lags)],
    #     out_column="target_lag",
    # )
    pipeline = Pipeline(
        model=model,
        horizon=horizon,
        # transforms=[StandardScalerTransform(in_column="target")]
    )

        # tslogger.add(LocalFileLogger(config=parameters, experiments_folder=experiments_folder))
    # tslogger.add(
    #     WandbLogger(
    #         project="test-pi-transformer",
    #         config=parameters,
    #     )
    # )

    metrics = [SMAPE(), MSE(), MAE(), Sign()]
    metrics, forecast, fold_info = pipeline.backtest(ts, metrics=metrics, n_folds=3, n_jobs=1)    
    print(metrics)
    print(forecast.tail(10))
    print(ts.df.tail(10))
    plot_backtest(forecast, ts, history_len=20)
    plt.show(block=True)
    

if __name__ == "__main__":
    typer.run(train_backtest)