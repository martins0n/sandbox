from typing import Dict, List
from importlib_metadata import metadata
import torch
import torch.nn as nn
import pytorch_lightning as pl

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
)
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify


class RNNet(nn.Module):
    def __init__(
        self,
        freq: str,
        hidden_dim: List[int],
        prediction_length: int,
        context_length: int,
        input_size: int = 1,
    ):
        super().__init__()

        assert prediction_length > 0
        assert len(hidden_dim) > 0
        assert context_length > 0

        self.input_size = input_size
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.shapes = [input_size] + hidden_dim
        self.cell = nn.RNN
        modules = [
            self.cell(input_size=in_dim, hidden_size=out_dim, batch_first=True)
            for (in_dim, out_dim) in zip(self.shapes[:-1], self.shapes[1:])
        ]
        modules.append(nn.Linear(in_features=self.shapes[-1], out_features=1))
        self.layers = nn.ModuleList(modules=modules)
        self.loss = nn.MSELoss()

    def forward(self, context: torch.Tensor):
        modules = list(iter(self.layers))
        if self.training:
            output = context.unsqueeze(dim=-1)
            for layer in modules[:-1]:
                output, hs = layer(output)
            decoder_output = output[:, -self.prediction_length :]
            decoder_output = modules[-1](decoder_output)
            return decoder_output.reshape(decoder_output.shape[0], 1, -1)
        else:
            forecasts: List[torch.Tensor] = list()
            hidden_states = list()
            output = context.unsqueeze(dim=-1)
            for layer in modules[:-1]:
                output, hs = layer(output)
                hidden_states.append(hs)
            decoder_output = output[:, -1]
            decoder_output = modules[-1](decoder_output)
            forecasts.append(decoder_output)
            for _ in range(1, self.prediction_length):
                output = forecasts[-1].unsqueeze(dim=1)
                for idx, layer in enumerate(modules[:-1]):
                    output, hs = layer(output, hidden_states[idx])
                    hidden_states[idx] = hs
                decoder_output = output[:, -1]
                decoder_output = modules[-1](decoder_output)
                forecasts.append(decoder_output)
            forecast = torch.cat(forecasts, dim=1).unsqueeze(1)
            return forecast

    def get_predictor(self, input_transform, batch_size: int = 32, device="cpu"):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            freq=self.freq,
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )


class RNNetGluon(RNNet, pl.LightningModule):
    def __init__(
        self,
        freq: str,
        hidden_dim: List[int],
        prediction_length: int,
        context_length: int,
        input_size: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__(
            freq, hidden_dim, prediction_length, context_length, input_size
        )
        self.lr = lr

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        context = batch["past_target"]
        target = batch["future_target"]
        context = torch.cat((context, target), dim=1)
        target_prediction = self(context)
        return self.loss(target_prediction, target.unsqueeze(dim=1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":

    batch_size = 2
    prediction_length = 10
    context_length = 11
    hidden_dims = [10, 10]
    input_tensor = torch.rand((batch_size, context_length + prediction_length))

    net = RNNet(
        freq="1D",
        hidden_dim=hidden_dims,
        prediction_length=prediction_length,
        context_length=context_length,
    )

    output = net(input_tensor)
    assert output.shape == (batch_size, 1, prediction_length)

    net.eval()

    output = net(input_tensor)
    assert output.shape == (batch_size, 1, prediction_length)

    dataset = get_dataset("electricity")

    freq = "1H"
    context_length = 2 * 7 * 24
    prediction_length = dataset.metadata.prediction_length
    print(dataset.metadata)
    hidden_dimensions = [96, 48]

    net = RNNetGluon(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        hidden_dim=hidden_dimensions,
    )

    for p in net.parameters():
        # print(p.shape)
        pass
    mask_unobserved = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )
    training_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=prediction_length,
        ),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )

    batch_size = 32
    num_batches_per_epoch = 100

    data_loader = TrainDataLoader(
        Cached(dataset.train),
        batch_size=batch_size,
        stack_fn=batchify,
        transform=mask_unobserved + training_splitter,
        num_batches_per_epoch=num_batches_per_epoch,
    )
    from gluonts.evaluation import make_evaluation_predictions, Evaluator

    trainer = pl.Trainer(max_epochs=1, gpus=None)
    trainer.fit(net, train_dataloader=data_loader)

    prediction_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=TestSplitSampler(),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )
    predictor_pytorch = net.get_predictor(
        mask_unobserved + prediction_splitter, batch_size=1
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test, predictor=predictor_pytorch, num_samples=1
    )

    forecasts_pytorch = list(f for f in forecast_it)
    tss_pytorch = list(ts_it)
