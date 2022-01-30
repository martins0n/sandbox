from itertools import islice
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
)
from importlib_metadata import pathlib


class FFNet(nn.Module):
    def __init__(
        self,
        freq: str,
        hidden_dim: List[int],
        prediction_length: int,
        context_length: int,
    ):
        super().__init__()

        assert prediction_length > 0
        assert len(hidden_dim) > 0
        assert context_length > 0

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.shapes = [context_length] + hidden_dim + [prediction_length]
        modules = [
            nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim), nn.ReLU()
            )
            for (in_dim, out_dim) in zip(self.shapes[:-1], self.shapes[1:])
        ]
        modules[-1] = nn.Linear(
            in_features=self.shapes[-2], out_features=self.shapes[-1]
        )
        self.net = nn.Sequential(*modules)

        self.loss = nn.MSELoss()

    def forward(
        self, context: torch.Tensor  # (batch_size, context_length + num_add_features)
    ) -> torch.Tensor:
        # (batch_size, 1, prediction_length)
        return self.net(context).unsqueeze(dim=1)


class FFNetGluon(FFNet, pl.LightningModule):
    def __init__(
        self,
        freq: str,
        hidden_dim: List[int],
        prediction_length: int,
        context_length: int,
        lr: float = 1e-3,
    ):
        super().__init__(
            freq=freq,
            hidden_dim=hidden_dim,
            prediction_length=prediction_length,
            context_length=context_length,
        )
        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):

        context = batch["past_target"]
        target = batch["future_target"]
        target_prediction = self(context)

        loss = self.loss(target_prediction, target.unsqueeze(dim=1))
        self.log(
            "train_loss",
            loss.item() ** (0.5),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class FFNetEstimator(PyTorchLightningEstimator):
    def __init__(
        self,
        freq: str,
        hidden_dim: List[int],
        prediction_length: int,
        context_length: int,
        batch_size: int,
        num_batches_per_epoch: int,
        trainer_kwargs: Dict[str, Any] = dict(),
        lr: float = 1e-4,
    ) -> None:
        super().__init__(trainer_kwargs=trainer_kwargs)

        self.freq = freq
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.lr = lr
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

    def create_transformation(self) -> Transformation:
        mask_unobserved = AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )
        return mask_unobserved

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1,
                min_future=self.prediction_length,
            ),
            "validation": ExpectedNumInstanceSampler(
                num_instances=1,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )

    def create_lightning_module(self) -> pl.LightningModule:
        return FFNetGluon(
            freq=self.freq,
            hidden_dim=self.hidden_dim,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            lr=self.lr,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: pl.LightningModule,
        device: str = "cpu",
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter("test")
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            freq=self.freq,
            input_names=["past_target"],
            prediction_net=module,
            batch_size=self.batch_size,
            input_transform=transformation + prediction_splitter,
            device=device,
        )

    def create_training_data_loader(self, data: Dataset, network: nn.Module, **kwargs):
        data_loader = TrainDataLoader(
            Cached(data),
            batch_size=self.batch_size,
            stack_fn=batchify,
            transform=self.create_transformation()
            + self._create_instance_splitter("training"),
            num_batches_per_epoch=self.num_batches_per_epoch,
        )
        return data_loader


if __name__ == "__main__":

    batch_size = 32
    prediction_length = 10
    context_length = 11
    hidden_dims = [10, 10]
    input_tensor = torch.rand((batch_size, context_length))

    net = FFNet(
        freq="1D",
        hidden_dim=hidden_dims,
        prediction_length=prediction_length,
        context_length=context_length,
    )

    assert net(input_tensor).shape == (batch_size, 1, prediction_length)

    dataset = get_dataset("electricity")

    freq = "1H"
    context_length = 2 * 7 * 24
    prediction_length = dataset.metadata.prediction_length
    hidden_dimensions = [100]

    batch_size = 16
    num_batches_per_epoch = 100

    ASSETS_PATH = pathlib.Path(__file__).parent / "assets"

    model = FFNetEstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        hidden_dim=hidden_dimensions,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        trainer_kwargs=dict(max_epochs=50),
        lr=1e-4,
    )
    predictor = model.train(
        training_data=dataset.train,
        shuffle_buffer_length=5,
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test, predictor=predictor
    )

    forecasts_pytorch = list(f for f in forecast_it)
    tss_pytorch = list(ts_it)

    # figure plot

    plt.figure(figsize=(20, 15))

    for idx, (forecast, ts) in islice(
        enumerate(zip(forecasts_pytorch, tss_pytorch)), 9
    ):
        ax = plt.subplot(3, 3, idx + 1)
        plt.plot(ts[-5 * prediction_length :], label="target")
        forecast.plot()

    plt.gcf().tight_layout()
    plt.legend()
    plt.savefig(ASSETS_PATH / "ffnet-electricity.png", dpi=200)

    # metrics

    evaluator = Evaluator(quantiles=[0.5])
    metrics_pytorch, _ = evaluator(
        iter(tss_pytorch), iter(forecasts_pytorch), num_series=len(dataset.test)
    )
    with open(ASSETS_PATH / "ffnet-electricity.metrics", "w") as f:
        pd.DataFrame.from_records(metrics_pytorch, index=["FFNet"]).to_markdown(f)
