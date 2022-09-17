from turtle import forward
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
        decoder_length = decoder_target.shape[1]
        forecast = torch.zeros_like(decoder_target)  # (batch_size, decoder_length, 1)
        mask = mask_gen(encoder_real, self.num_heads) 
        forecast[:, 0, :] = self.pi_trans(encoder_real, shift=0, window_size=decoder_length, mask=mask)[:, -1 , :]
        for t in range(1, decoder_length):
            input_ = torch.cat([encoder_real, forecast[:, :t, :]], dim=1)
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
        #loss = self.loss(output, decoder_target, encoder_target)
        loss = self.loss(output, decoder_target)
        return loss, decoder_target, output

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
    lr: float = 0.0001,
    batch_size: int = 32,
    seed: int = 11,
    dataset_path: pathlib.Path = pathlib.Path("/Users/marti/Projects/etna/examples/data/example_dataset.csv"),
    experiments_folder: pathlib.Path = pathlib.Path("experiments"),
    dataset_freq: str = "D",
):
    parameters = dict(locals())
    parameters["dataset_path"] = str(dataset_path)
    parameters["experiments_folder"] = str(experiments_folder)

    set_seed(seed)

    #original_df = pd.read_csv(dataset_path)
    original_df = generate_from_patterns_df(
        periods=200, start_time="2020-01-01", patterns=[[50, 50, 50, 50, 50, 90, 100]], freq='D', add_noise=True,  sigma=4,
    )
    df = TSDataset.to_dataset(original_df)
    ts = TSDataset(df, freq=dataset_freq)

    model = PITransformerModel(
        n_layers=1,
        #loss = MASELoss(7),
        decoder_length=horizon,
        encoder_length=3 * horizon,
        trainer_params={"max_epochs": n_epochs},
        lr=lr,
        train_batch_size=batch_size,
    )

    pipeline = Pipeline(
        model=model,
        horizon=horizon,
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

if __name__ == "__main__":
    typer.run(train_backtest)