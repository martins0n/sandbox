import torch

from torch.nn import Module, Parameter
import torch.nn as nn
from typing import TypedDict
import numpy as np
import random


class TiDeBatch(TypedDict):
    decoder_covariates: torch.Tensor # (batch_size, horizon, covariates_size)
    encoder_covariates: torch.Tensor # (batch_size, lookback, covariates_size)
    attributes: torch.Tensor    # (batch_size, attributes_size)
    decoder_target: torch.Tensor # (batch_size, horizon, output_size)
    encoder_target: torch.Tensor # (batch_size, lookback, output_size)


class ResidualBlock(Module):
    
    
    def __init__(self, hidden_size, input_size, output_size, p, layer_norm = True) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.hidden = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_size)
        else:
            self.layer_norm = nn.Identity()
        self.proj_residual = nn.Linear(in_features=input_size, out_features=output_size)
        
    def forward(self, x):
        residual = self.proj_residual(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.proj(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class ResidualSequenceBlock(Module):
    def __init__(self, n_blocks: int, hidden_size: int, input_size: int, output_size: int, p:float) -> None:
        super().__init__()
        if n_blocks == 1:
            self.sequence = ResidualBlock(hidden_size=hidden_size, input_size=input_size, output_size=output_size, p=p)
        else:
            self.sequence = nn.Sequential(
                *(  [ResidualBlock(hidden_size=hidden_size, input_size=input_size, output_size=hidden_size, p=p)]
                    + [ResidualBlock(hidden_size=hidden_size, input_size=hidden_size, output_size=hidden_size, p=p) for _ in range(n_blocks - 2)]
                    + [ResidualBlock(hidden_size=hidden_size, input_size=hidden_size, output_size=output_size, p=p)]
                )
            )  
    def forward(self, x):
        return self.sequence(x)
    

class TiDe(Module):
    def __init__(
        self, 
        ne_blocks: int,
        nd_blocks,
        hidden_size: int,
        covariates_size: int,
        p: float,
        lookback: int,
        decoder_output_size: int,
        temporal_decoder_hidden_size: int,
        feature_projection_output_size: int,
        feature_projection_hidden_size: int,
        horizon: int,
        static_covariates_size: int,
    ) -> None:
        super().__init__()
        
        self.feature_projection = ResidualBlock(
            input_size=covariates_size,
            output_size=feature_projection_output_size,
            hidden_size=feature_projection_hidden_size,
            p=p
        )
        
        self.temporal_decoder = ResidualBlock(
            input_size=decoder_output_size // horizon + feature_projection_output_size,
            output_size=1,
            hidden_size=temporal_decoder_hidden_size,
            p=p   
        )
        self.residual_lookback_projection = nn.Linear(in_features=lookback, out_features=horizon)
        
        self.encoder = ResidualSequenceBlock(
            n_blocks=ne_blocks,
            p=p,
            input_size=lookback + static_covariates_size + feature_projection_output_size * lookback,
            output_size=hidden_size,
            hidden_size=hidden_size
        )
        
        self.decoder = ResidualSequenceBlock(
            n_blocks=nd_blocks,
            p=p,
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=decoder_output_size
        )
        
        
    def forward(self, x: TiDeBatch) -> torch.Tensor:
        
        batch_size = x['encoder_covariates'].shape[0]
        horizon = x['decoder_covariates'].shape[1]
        
        encoder_covariates = x['encoder_covariates']
        decoder_covariates = x['decoder_covariates']
        attributes = x['attributes']
        encoder_target = x['encoder_target']

        encoder_feature_projection = self.feature_projection(encoder_covariates)
        decoder_feature_projection = self.feature_projection(decoder_covariates)
        
        history = torch.cat(
            [
                encoder_target.reshape(batch_size, -1),
                attributes,
                encoder_feature_projection.reshape(batch_size, -1)
        ], dim=-1)
        
        encoded_history = self.encoder(history)
        decoded_history = self.decoder(encoded_history)
        
        temporal_decoder_input = torch.cat(
            [
                decoded_history.reshape(batch_size, horizon, -1),
                decoder_feature_projection
            ], dim=-1
        )
        
        temporal_decoder_projection = self.temporal_decoder(temporal_decoder_input)
        
        residual_lookback_projection = self.residual_lookback_projection(
            encoder_target.reshape(batch_size, -1)
        ).reshape(batch_size, horizon, -1)
        
        y_hat = residual_lookback_projection + temporal_decoder_projection
        
        return y_hat
        
        


if __name__  == "__main__":
    
    # set seed
    seed = 11
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    lookback = 14
    horizon = 7
    batch_size = 4
    covariates_size = 3
    static_covariates_size = 2
    
    # paper hyperparameters
    hidden_size = 32
    ne_blocks = 2
    nd_blocks = 2
    decoder_output_size = horizon * 10
    temporal_decoder_hidden_size = 32
    dropout_level = 0.1
    layer_norm = True
    lr = 0.001
    revin = False
    
    # fixed hyperparameters
    # lookback = 720
    # metric = 'MSE'
    # standard normalization for all variables
    # dynamic covariates:  like minute ofthe hour, hour of the day, day of the week etc
    # Adam optimizer and tensorflow
    
    

    batch = {
        'decoder_covariates': torch.randn(batch_size, horizon, covariates_size),
        'encoder_covariates': torch.randn(batch_size, lookback, covariates_size),
        'attributes': torch.randn(batch_size, static_covariates_size),
        'encoder_target': torch.randn(batch_size, lookback, 1),
        'decoder_target': torch.randn(batch_size, horizon, 1)
    }
    
    model = TiDe(
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
        static_covariates_size=static_covariates_size
    )
    
    with torch.no_grad():
        model.eval()
        y_hat = model(batch)
        assert y_hat.shape == (batch_size, horizon, 1)
