import pandas as pd
import re
from datasets import Dataset

from functools import partial
import numpy as np
import random

import json
import pathlib 

from typing import TypedDict, List, Callable


SEED = 11

random.seed(SEED)
np.random.seed(SEED)


ScrapedData = TypedDict("ScrapedData", {"title": str, "content": str, "href": str})


FILE_DIR = pathlib.Path(__file__).parent.absolute()


def kashin_guru_to_list() -> List[ScrapedData]:
    with open(FILE_DIR.parent / "scraper" / "kashin_guru.jsonlines") as f:
        data = [json.loads(line) for line in f]
    return data

def kashin_lj_to_list() -> List[ScrapedData]:
    with open(FILE_DIR.parent / "scraper" / "kashin_lj.jsonlines") as f:
        data = [json.loads(line) for line in f]
    return data


def split_text(text: str, prob: float = 0.5) -> List[str]:
    end_of_seq = set([".", "!", "?"])
    end_indexes = [i for i, c in enumerate(text) if c in end_of_seq]
    end_indexes = end_indexes[:-1]
    mask = np.random.random(len(end_indexes)) > prob
    end_indexes = list(np.array(end_indexes)[mask])
    
    end_indexes.insert(0, -1)
    end_indexes.append(len(text)-1)
        
    text = [text[i+1:j+1].strip() for i, j in zip(end_indexes[:-1], end_indexes[1:])]

    return text

def create_dataset(data_generators: List[Callable[[], List[ScrapedData]]], prob: float = 0.5, split: bool = False) -> Dataset:
    data = []
    for generator in data_generators:
        data.extend(generator())
    df = pd.DataFrame(data)
    df = df[df["content"] != ""]
    df = df.drop_duplicates(subset=["content"])
    df["content"] = df["content"].apply(lambda x: re.sub(r'http\S+', '', x))
    if split:
        df['text'] = df['content'].apply(partial(split_text, prob=prob))
        df = df[['text']]
        df = df.explode('text')
    else:
        df['text'] = df['content']
        df = df[['text']]
    return Dataset.from_pandas(df)

if __name__ == "__main__":
        
    dataset = create_dataset([kashin_guru_to_list, kashin_lj_to_list], prob=0.5, split=False)
    
    dataset = dataset.train_test_split(train_size=0.9, seed=SEED)
    
    for split, dataset in dataset.items():
        dataset.to_json(FILE_DIR / "datasets" / f"kashin-v1-{split}.json")