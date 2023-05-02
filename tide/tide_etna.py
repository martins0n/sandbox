from typing import Iterable
from etna.models.base import DeepBaseModel
from etna.models.base import DeepBaseNet
import pandas as pd
from tide import TiDe
import torch.nn as nn
import torch
import numpy as np
from pandas import DataFrame
from typing import Optional
from tide import TiDeBatch


class TiDeNet(DeepBaseNet):
    def __init__(
        self,
        lr: float = 1e-2,
        loss: nn.Module = nn.MSELoss(),
        optimizer_params: dict = None,
        **kwargs,
    ):
        super().__init__()
        
        self.tide = TiDe(**kwargs)
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params or {}
        
    def forward(self, x):
        return self.tide(x)

    def step(self, batch: TiDeBatch, *args, **kwargs):
        
        y_hat = self(batch)
        
        decoder_target = batch['decoder_target']
        
        loss = self.loss(y_hat, decoder_target)
        return loss, decoder_target, y_hat
    
    def make_samples(self, df: DataFrame, encoder_length: int, decoder_length: int) -> Iterable[dict]:
        max_sequence_length = encoder_length + decoder_length
        number_of_sequences = int(len(df) //  max_sequence_length * 10)
        if len(df) == max_sequence_length:
            sequence_start_idx = np.array([0])
        else:
            sequence_start_idx = np.random.randint(0, len(df) - max_sequence_length, size=number_of_sequences)
        
        samples = []
        
        for idx in sequence_start_idx:
            sample = dict()
            view = df.iloc[idx:idx + max_sequence_length].select_dtypes(include=[np.number])
            sample['encoder_target'] = view[['target']][:encoder_length].values.astype(np.float32)
            sample['decoder_target'] = view[['target']][encoder_length:].values.astype(np.float32)
            sample['encoder_covariates'] = view.drop(columns=['target'])[:encoder_length].values.astype(np.float32)
            sample['decoder_covariates'] = view.drop(columns=['target'])[encoder_length:].values.astype(np.float32)
            sample['attributes'] = np.array([0.0]).astype(np.float32)
            sample['segment'] = df['segment'].values[0]
            
            samples.append(sample)
        
        return samples
    
    def configure_optimizers(self):
        """Optimizer configuration."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return [optimizer], [scheduler]



class TiDeModel(DeepBaseModel):
    
    def __init__(
        self,
        decoder_length: int,
        encoder_length: int,
        lr: float = 1e-3,
        loss: Optional["torch.nn.Module"] = nn.MSELoss(),
        train_batch_size: int = 16,
        test_batch_size: int = 16,
        optimizer_params: Optional[dict] = None,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params
        super().__init__(
            net=TiDeNet(
                lr=lr,
                loss=loss,
                optimizer_params=optimizer_params,
                **kwargs,
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