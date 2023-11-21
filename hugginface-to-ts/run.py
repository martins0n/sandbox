import os
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from accelerate import Accelerator, init_empty_weights, infer_auto_device_map, dispatch_model


context_length = int(os.environ.get("CONTEXT_LENGTH", 30))
target_length = int(os.environ.get("TARGET_LENGTH", 10))
test_size = float(os.environ.get("TEST_SIZE", 0.2))
batch_size = int(os.environ.get("BATCH_SIZE", 32))
lr = float(os.environ.get("LR", 5e-5))
num_epochs = int(os.environ.get("NUM_EPOCHS", 10))
model_name = str(os.environ.get("MODEL", "EleutherAI/pythia-70m"))

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name)

def init():
    global model, model_name
    if model_name == "EleutherAI/pythia-70m":

        model.gpt_neox.embed_in = nn.Linear(1, 512)

        model.gpt_neox.embed_out = nn.Linear(512, 1)

        model.embed_out = nn.Linear(512, 1)

    if model_name == "EleutherAI/pythia-1B":
        model.gpt_neox.embed_in = nn.Linear(1, 2048)

        model.embed_out = nn.Linear(2048, 1)

# todo fix hardcoded max_memory
device_map = infer_auto_device_map(model, max_memory={0:"1GB", 1:"14GB", "cpu": "60GiB"})

print(f"device map {device_map}")
model = AutoModelForCausalLM.from_pretrained(model_name)

init()

model = dispatch_model(model, device_map=device_map)

df = pd.read_csv(
    "https://raw.githubusercontent.com/tinkoff-ai/etna/master/examples/data/example_dataset.csv"
)

accelerator = Accelerator()


class Sample(TypedDict):
    context: torch.FloatTensor
    target: torch.FloatTensor
    timestamp: pd.Timestamp
    segment: str


samples = []

timestamps = sorted(df.timestamp.values)
test_timestmap = timestamps[-target_length - int(len(timestamps) * test_size)]

train_mean = (
    df[df.timestamp < test_timestmap].groupby("segment").target.mean().to_dict()
)

df = df.sort_values(by="timestamp")

for segment in df.segment.unique():
    segment_df = df[df.segment == segment]
    for i in range(len(segment_df) - context_length - target_length):
        context = (
            segment_df.target.iloc[i : i + context_length]
            .values.reshape(-1, 1)
            .astype(np.float32)
        ) / train_mean[segment]
        target = (
            segment_df.target.iloc[
                i + context_length : i + context_length + target_length
            ]
            .values.reshape(-1, 1)
            .astype(np.float32)
        ) / train_mean[segment]
        timestamp = segment_df.timestamp.iloc[i + context_length]
        samples.append(
            Sample(context=context, target=target, timestamp=timestamp, segment=segment)
        )

segment_0 = df.segment.unique()[0]

timestamps = df[df.segment == segment_0].timestamp.values


train_samples = [sample for sample in samples if sample["timestamp"] < test_timestmap]
test_samples = [sample for sample in samples if sample["timestamp"] >= test_timestmap]



optimizer = AdamW(model.parameters(), lr=lr)

train_dataset = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
test_dataset = DataLoader(test_samples, batch_size=batch_size, shuffle=False)

optimizer, train_dataset, test_dataset = accelerator.prepare(optimizer, train_dataset, test_dataset)

model.train()
for epoch in range(num_epochs):
    test_losses = []

    for batch in train_dataset:
        full_seq = torch.cat([batch["context"], batch["target"]], dim=1)
        outputs = model(full_seq)
        shifted_outputs = outputs.logits[:, :-1]
        target = full_seq[:, 1:]
        loss = (shifted_outputs - target).pow(2).mean()
        accelerator.backward(loss)
        loss = loss.item()
        test_losses.append(loss)
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch}", f"Loss {np.mean(test_losses)}")


def decode(model, context, target_length):
    model.eval()
    context = torch.tensor(context)
    with torch.no_grad():
        for _ in range(target_length):
            outputs = model(context)
            logits = outputs.logits[:, -1].unsqueeze(1)
            context = torch.cat([context, logits], dim=1)

    return context[:, -target_length:]


model.eval()

mse_losses = []
smape_losses = []

for batch in test_dataset:
    context = batch["context"]
    target = batch["target"]

    pred = decode(model, context, target_length)

    loss = (pred - target).pow(2).mean().item()
    smape = (
        torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target))
    ).mean().item() * 100
    mse_losses.append(loss)
    smape_losses.append(smape)

print(f"MSE {np.mean(mse_losses)}")
print(f"SMAPE {np.mean(smape_losses)}")
