import pathlib
from functools import partial
from itertools import islice
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AdhocTransform,
    ConcatFeatures,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    Transformation,
)

from ffn import FFNetEstimator


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
        # self.cell = partial(nn.RNN, nonlinearity="relu")
        self.cell = nn.GRU
        modules = [
            self.cell(input_size=in_dim, hidden_size=out_dim, batch_first=True)
            for (in_dim, out_dim) in zip(self.shapes[:-1], self.shapes[1:])
        ]
        modules.append(nn.Linear(in_features=self.shapes[-1], out_features=1))
        self.layers = nn.ModuleList(modules=modules)
        self.loss = nn.MSELoss()

    def forward(
        self,
        past_target: torch.Tensor,  # (batch_size, context_length + prediction_length | 0 ),
        past_exog: Optional[
            torch.Tensor
        ] = None,  # (batch_size, context_length + prediction_length)
        future_exog: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        modules = list(iter(self.layers))
        if False:
            # (batch_size, context_length + prediction_length, 1)
            output = context.unsqueeze(dim=-1)
            # (batch_size, 1)
            scale = (
                context[:, : self.context_length]
                .mean(dim=1, keepdim=True)
                .abs()
                .clip(1e-5)
            )
            output = output / scale.unsqueeze(dim=-1)
            for layer in modules[:-1]:
                output, hs = layer(output)
            # (batch_size, prediction_length, self.shapes[-1])
            decoder_output = output[:, -self.prediction_length - 1 : -1]
            # (batch_size, prediction_length, 1)
            decoder_output = modules[-1](decoder_output) * scale.unsqueeze(dim=-1)
            # (batch_size, 1, prediction_length)
            return decoder_output.reshape(decoder_output.shape[0], 1, -1)
        else:
            batch_size = past_target.shape[0]
            forecasts: List[torch.Tensor] = list()
            hidden_states = list()

            if future_exog is not None:
                exog_features = torch.cat((past_exog, future_exog), dim=1)
            else:
                exog_features = None

            if exog_features is not None:
                assert exog_features.shape[2] == self.input_size - 1
            # (batch_size, context_length, 1)
            output = past_target.unsqueeze(dim=-1)
            # (batch_size, 1)
            scale = (
                past_target[:, : self.context_length]
                .mean(dim=1, keepdim=True)
                .abs()
                .clip(1e-5, 1e5)
            )
            # (batch_size, context_length, 1)
            output = output / scale.unsqueeze(dim=-1)
            assert output.shape == (batch_size, self.context_length, 1)

            if exog_features is not None:
                exog_features = exog_features / exog_features.mean(dim=1, keepdim=True)
                output = torch.cat(
                    (output, exog_features[:, : self.context_length]), dim=2
                )
            for layer in modules[:-1]:
                output, hs = layer(output)
                hidden_states.append(hs)
            # (batch_size, self.hidden_dim[-1])
            decoder_output = output[:, -1]
            # (batch_size, 1)
            decoder_output: torch.Tensor = modules[-1](decoder_output) * scale
            assert decoder_output.shape == (decoder_output.shape[0], 1)
            forecasts.append(decoder_output)
            for idx in range(1, self.prediction_length):
                # (batch_size, 1, 1)
                output = forecasts[-1].unsqueeze(dim=1)
                output = output / scale.unsqueeze(dim=-1)
                assert output.shape == (batch_size, 1, 1)
                if exog_features is not None:
                    output = torch.cat(
                        (
                            output,
                            exog_features[
                                :,
                                self.context_length
                                + idx
                                - 1 : self.context_length
                                + idx,
                            ],
                        ),
                        dim=2,
                    )
                for idx, layer in enumerate(modules[:-1]):
                    output, hs = layer(output, hidden_states[idx])
                    hidden_states[idx] = hs
                decoder_output = output[:, -1]
                # (batch_size, 1)
                decoder_output = modules[-1](decoder_output) * scale
                forecasts.append(decoder_output)
            forecast = torch.cat(forecasts, dim=1).unsqueeze(1)
            assert forecast.shape == (past_target.shape[0], 1, self.prediction_length)
            return forecast


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
        self.save_hyperparameters()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        past_target = batch["past_target"]
        target = batch["future_target"]
        future_exog_features = batch.get("future_exog")
        past_exog_features = batch.get("past_exog")
        # context = torch.cat((context, target), dim=1)
        target_prediction = self(past_target, past_exog_features, future_exog_features)
        scale = (
            past_target.abs().mean(dim=1, keepdim=True).clip(1e-5, 1e10)
        ).unsqueeze(dim=1)
        scaled_prediction = target_prediction / scale
        scaled_target = target.unsqueeze(dim=1) / scale
        loss = self.loss(scaled_prediction, scaled_target)

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


class RNNNetEstimator(FFNetEstimator):
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
        input_size: int = 1,
    ) -> None:
        PyTorchLightningEstimator.__init__(self, trainer_kwargs=trainer_kwargs)

        self.freq = freq
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.lr = lr
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.input_size = input_size

    def create_lightning_module(self) -> pl.LightningModule:
        return RNNetGluon(
            freq=self.freq,
            hidden_dim=self.hidden_dim,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            lr=self.lr,
            input_size=self.input_size,
        )

    def create_transformation(self) -> Transformation:
        mask_unobserved = AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )
        add_age_feature = AddAgeFeature(
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_AGE,
            pred_length=self.prediction_length,
            log_scale=True,
        )
        concat_feature = ConcatFeatures(
            output_field="exog", input_fields=[FieldName.FEAT_AGE], drop_inputs=True
        )
        return mask_unobserved + add_age_feature + concat_feature

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
            input_names=["past_target", "past_exog", "future_exog"],
            prediction_net=module,
            batch_size=self.batch_size,
            input_transform=transformation + prediction_splitter,
            device=device,
        )

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
            time_series_fields=[FieldName.OBSERVED_VALUES, "exog"],
        )


if __name__ == "__main__":

    batch_size = 2
    prediction_length = 10
    context_length = 11
    hidden_dims = [10, 10]
    exog_dim = 2
    input_tensor = torch.rand((batch_size, context_length))
    past_exog_features = torch.rand((batch_size, context_length, exog_dim))
    future_exog_features = torch.rand((batch_size, prediction_length, exog_dim))

    net = RNNet(
        freq="1D",
        hidden_dim=hidden_dims,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=1 + exog_dim,
    )

    output = net(input_tensor, past_exog_features, future_exog_features)
    assert output.shape == (batch_size, 1, prediction_length)

    net.eval()

    output = net(input_tensor, past_exog_features, future_exog_features)
    assert output.shape == (batch_size, 1, prediction_length)

    dataset = get_dataset("electricity")

    freq = "1H"
    context_length = 2 * 7 * 24
    prediction_length = dataset.metadata.prediction_length
    hidden_dimensions = [64]

    batch_size = 16
    num_batches_per_epoch = 100

    ASSETS_PATH = pathlib.Path(__file__).parent / "assets"

    model = RNNNetEstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        hidden_dim=hidden_dimensions,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        trainer_kwargs=dict(max_epochs=10),
        lr=1e-2,
        input_size=2,
    )
    predictor = model.train(
        training_data=dataset.train,
        shuffle_buffer_length=5,
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test, predictor=predictor, num_samples=1
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
    plt.savefig(ASSETS_PATH / "rnnnet-electricity.png", dpi=200)

    # metrics

    evaluator = Evaluator(quantiles=[0.5])
    metrics_pytorch, _ = evaluator(
        iter(tss_pytorch), iter(forecasts_pytorch), num_series=len(dataset.test)
    )
    with open(ASSETS_PATH / "rnnnet-electricity.metrics", "w") as f:
        pd.DataFrame.from_records(metrics_pytorch, index=["RNNNet"]).to_markdown(f)
