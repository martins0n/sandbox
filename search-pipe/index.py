import heapq
from pathlib import Path
from time import time

import joblib
import pandas as pd
import typer
import yaml
from hydra_slayer import get_from_params
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from pymorphy3 import MorphAnalyzer
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

tokenizer = RegexpTokenizer(r"[a-zA-Zа-яА-Я]+")
stem = SnowballStemmer("english")
rus_stopwords = stopwords.words("russian")
eng_stopwords = stopwords.words("english")

ru_chars = set(list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"))


def normalize_word(x):
    x = x.lower()
    if x[0] in ru_chars:
        morph = MorphAnalyzer()
        x = morph.parse(x)[0].normal_form
        if x in rus_stopwords:
            return ""
    else:
        x = stem.stem(x)
        if x in eng_stopwords:
            return ""
    return x


def normalize(text):
    words = tokenizer.tokenize(text)
    words = [normalize_word(word) for word in words]
    words = [word for word in words if word]
    return " ".join(words)


class IIndex:
    def fit(self, target_list):
        pass

    def find_most_similar_texts(self, sample_text, top_k=None):
        pass


class TfidfIndex(IIndex):
    def __init__(self, top_k=3, metric="cosine", tfidf_params={}, nn_params={}):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        self.nn = NearestNeighbors(n_neighbors=top_k, metric=metric, **nn_params)

    def fit(self, target_list):
        self.target_list = target_list
        target_list = self.vectorizer.fit_transform(target_list)
        self.nn.fit(target_list)

    def find_most_similar_texts(self, sample_text, top_k=None):
        if top_k is None:
            top_k = self.top_k
        sample_vector = self.vectorizer.transform([sample_text])
        distances, indices = self.nn.kneighbors(sample_vector)
        return [
            (self.target_list[idx], 1 - dist)
            for idx, dist in zip(indices[0], distances[0])
        ]


class FuzzyIndex(IIndex):
    def __init__(self, top_k=3):
        self.top_k = top_k

    def fit(self, target_list):
        self.target_list = target_list

    def find_most_similar_texts_fuzzy(self, sample_text, top_k=None):
        if top_k is None:
            top_k = self.top_k
        return heapq.nlargest(
            top_k,
            [(text, fuzz.ratio(sample_text, text) / 100) for text in self.target_list],
            key=lambda x: x[1],
        )

    def find_most_similar_texts(self, sample_text, top_k=None):
        return self.find_most_similar_texts_fuzzy(sample_text, top_k)


app = typer.Typer()


@app.command()
def create(save_path, file_path, index_config_path):
    with open(index_config_path, "r") as f:
        index_config = yaml.safe_load(f)

    target_list = pd.read_json(file_path, lines=True)["text"].values.tolist()

    index = get_from_params(**index_config)

    index.fit([normalize(text) for text in target_list])

    with open(save_path, "wb") as f:
        joblib.dump(index, f)


@app.command()
def infer(items_path: Path, index_path: Path, result_path: Path, top_k: int = 3):
    with open(index_path, "rb") as f:
        index = joblib.load(f)

    start_time = time()
    df_items = pd.read_json(items_path, lines=True)

    items = df_items["text"].values.tolist()

    results = []
    for i, item in tqdm(enumerate(items)):
        results.append(index.find_most_similar_texts(normalize(item), top_k))

    df_items["result"] = results

    df_items.to_json(result_path, orient="records", force_ascii=False, lines=True)


if __name__ == "__main__":
    app()
