from typing import List
from typing import Union

import nltk
import requests
from nltk import ngrams, stem, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r"\w{3,}")
stop_words = set(stopwords.words("english"))
stemmer = stem.SnowballStemmer(language="english")


def get_citation_index(x: str):
    """Get citation counter from crossref api."""
    if x is not None and len(x) > 4:
        try:
            data = requests.get(f"https://api.crossref.org/works/{x}").json()
            return data["message"]["is-referenced-by-count"]
        except:
            print(x)


def arxiv_published_flag(comment_message: str, doi: str) -> Union[List[str], str, None]:
    """Find out papers published or conference accepted."""
    key_words = {
        "workshop",
        "journal",
        "accepted",
        "code",
        "conference",
        "publication",
        "neurips",
        "submitted",
        "icml",
        "ieee",
    }
    if doi:
        return doi
    if comment_message is None:
        return None
    paper_tags = key_words.intersection(set(tokenizer.tokenize(comment_message.lower())))
    if (
        len(paper_tags)
        > 0
    ):
        return list(paper_tags)
    else:
        return None


def pipeline_normalize(x: str) -> List[str]:
    global stemmer
    global tokenizer
    global stop_words
    if x is None or len(x) < 1:
        return []
    _list_of_words = list()
    for i in tokenizer.tokenize(x):
        word = stemmer.stem(i).lower()
        if word in stop_words:
            continue
        else:
            _list_of_words.append(word)
    return _list_of_words
