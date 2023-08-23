import pathlib
import pandas as pd
import json
import gc


FILE_PATH = pathlib.Path(__file__)

END_TIMESTAMP = "2020-01-01"

DATASET = [
    ("Monthly", "MS", END_TIMESTAMP),
    ("Weekly", "W-MON", END_TIMESTAMP),
    ("Daily", "D", END_TIMESTAMP),
    ("Quarterly", "QS-JAN", END_TIMESTAMP),
    ("Yearly", "D", END_TIMESTAMP),
    ("Hourly", "H", END_TIMESTAMP),
]
if __name__ == "__main__":
    files = list((FILE_PATH.parents[1] / "etna").glob("*.parquet"))

    for (prefix, freq, end_timestamp) in DATASET:
        test_file = [
            _file for _file in files if _file.name.startswith(f"{prefix}-test")
        ][0]
        train_file = [
            _file for _file in files if _file.name.startswith(f"{prefix}-train")
        ][0]

        if ((FILE_PATH.parents[1] / "etna") / f"M4{prefix}.json").exists():
            continue

        df_train = pd.read_parquet(train_file)
        df_test = pd.read_parquet(test_file)

        test_points = df_test.groupby("segment").count().iloc[0, 0]

        df = pd.concat([df_train, df_test], axis=0)

        del df_train, df_test

        df = df.reset_index(drop=True)
        df = df.sort_values(["segment", "timestamp"])
        df_segment_counter = df.groupby(["segment"]).count()

        gc.collect()

        df_timestamps = list()
        for segment_nm, row in df_segment_counter.iterrows():
            df_timestamps += list(
                pd.date_range(end=end_timestamp, freq=freq, periods=row.target)
            )

        df["timestamp"] = df_timestamps

        train_end_timestamp = df_timestamps[-test_points - 1]

        assert df.groupby("segment").timestamp.max().nunique() == 1

        df.to_parquet(
            (FILE_PATH.parents[1] / "etna") / f"M4{prefix}.parquet",
            compression="gzip",
            index=False,
        )
        json_describe = {
            "tags": ["forecasting"],
            "freq": freq,
            "name": f"M4{prefix}",
            "last_train_datetime": str(train_end_timestamp),
        }
        with open((FILE_PATH.parents[1] / "etna") / f"M4{prefix}.json", "w") as f:
            json.dump(json_describe, f, indent=4)

        gc.collect()
