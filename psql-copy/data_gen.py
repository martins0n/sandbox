import pandas as pd
import os
import numpy as np


segments = int(os.environ.get('SEGMENTS', 10000))
n_timestamps = int(os.environ.get('TIMESTAMPS', 1000))
timestamp = pd.date_range('2019-01-01', periods=n_timestamps, freq='D')
dataset = {
    "timestamp":[],
    "target":[],
    "segment":[]
}
for i in range(segments):
    dataset["segment"] += [f"segment_{i}"] * n_timestamps
    dataset["timestamp"] += [f"{i}" for i in timestamp.tolist()]
    dataset["target"] += list(100 * np.random.normal(0, 1, n_timestamps))
    
df = pd.DataFrame(dataset)
df.to_csv(f"data_{segments}.csv", index=False, header=False)