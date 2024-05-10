import pandas as pd
from tqdm import tqdm
import dataclasses
from pathlib import Path


FILE = Path(__file__).absolute()

df_map = pd.read_csv(
    FILE.parent.parent / "raw" / "M4-info.csv",
)

freq_dict = {
    "1D": 365,
    "1Q": 4,
    "1H": 365*24,
    "1Y": 1,
    "1W": 53,
    "1M": 12
}

def timestamp_noramlizer(x, series_size=None):
    try:
        x = pd.to_datetime(x, format="%d-%m-%y %H:%M")
    except:
        try:
            x = pd.to_datetime(x, format="%d-%m-%y %H:%M:%S")
        except:
            x = pd.to_datetime(x)
#    if x.year > 2017:
#        x = x.replace(year=x.year-100)
#    if x.year + series_size > 2200:
#        x = x.replace(year=x.year-((x.year+series_size-2200) - (x.year+series_size-2200) % 100))
    return x


@dataclasses.dataclass
class M4datasets:
    train_path: str
    test_path: str
    SP: str
    freq: str
    etna_prefix_train: str
    etna_prefix_test: str


PREFIX = FILE.parent.parent / "raw"

list_of_datasets = [
    M4datasets(
        PREFIX /  "Daily-train.csv",
        PREFIX / "Daily-test.csv",
        "Daily",
        "1D",
        "Daily-train",
        "Daily-test",
    ),
    M4datasets(
        PREFIX /  "Quarterly-train.csv",
        PREFIX / "Quarterly-test.csv",
        "Quarterly",
        "1D",
        "Quarterly-train",
        "Quarterly-test",
    ),
    M4datasets(
        PREFIX / "Weekly-train.csv",
        PREFIX / "Weekly-test.csv",
        "Weekly",
        "1W",
        "Weekly-train",
        "Weekly-test",
    ),
    M4datasets(
        PREFIX / "Hourly-train.csv",
        PREFIX / "Hourly-test.csv",
        "Hourly",
        "1H",
        "Hourly-train",
        "Hourly-test",
    ),
    M4datasets(
        PREFIX / "Yearly-train.csv",
        PREFIX / "Yearly-test.csv",
        "Yearly",
        "1D",
        "Yearly-train",
        "Yearly-test",
    ),
    M4datasets(
        PREFIX / "Monthly-train.csv",
        PREFIX / "Monthly-test.csv",
        "Monthly",
        "1D",
        "Monthly-train",
        "Monthly-test",
    ),
]

def transform_dataset(m4dataset: M4datasets):
    global df_map
    df = pd.read_csv(m4dataset.train_path)
    df = df.set_index("V1").T

    dict_dataset = dict(timestamp=list(), segment=list(), target=list())
    data_len = dict()
    uniq_columns = df.columns
    
    for idx, x in tqdm(df_map[df_map.SP == m4dataset.SP].iterrows(), total=len(uniq_columns), desc=m4dataset.etna_prefix_train):
        column = x["M4id"]
        values = df[column].loc[:df[column].last_valid_index()]
        dict_dataset["target"] += list(values.values)
        dict_dataset["timestamp"] += list(
            pd.date_range(
                start=timestamp_noramlizer(x["StartingDate"], (len(values)+14)// freq_dict[m4dataset.freq] + 1),
                freq=m4dataset.freq, periods=len(values)
            )
        )
        dict_dataset["segment"] += [column]*len(values)
        data_len[column] = len(values)
    
    df_train = pd.DataFrame(dict_dataset)
    file_name = m4dataset.etna_prefix_train +".parquet"
    df_train.to_parquet(FILE.parent.parent / "etna" / file_name, index=False, compression="gzip")
    print(file_name)

    df = pd.read_csv(
        m4dataset.test_path,
    )
    df = df.set_index("V1").T

    dict_dataset = dict(timestamp=list(), segment=list(), target=list())
    uniq_columns = df.columns
    
    for idx, x in tqdm(df_map[df_map.SP == m4dataset.SP].iterrows(), total=len(uniq_columns), desc=m4dataset.etna_prefix_test):
        column = x["M4id"]
        values = df[column].loc[:df[column].last_valid_index()]
        dict_dataset["target"] += list(values.values)
        dict_dataset["timestamp"] += list(
            pd.date_range(
                start=timestamp_noramlizer(x["StartingDate"], (len(values)+14)// freq_dict[m4dataset.freq] + 1),
                freq=m4dataset.freq, periods=len(values)+data_len[column]
            )[-len(values):]
        )
        dict_dataset["segment"] += [column]*len(values)
        data_len[column] = len(values)

    df_test = pd.DataFrame(dict_dataset)
    file_name = m4dataset.etna_prefix_test +".parquet"

    df_test.to_parquet(FILE.parent.parent / "etna" / file_name, index=False, compression="gzip")
    print(file_name)

if __name__ ==  "__main__":
    _ = [transform_dataset(i) for i in list_of_datasets]