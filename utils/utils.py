import sys
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
import pandas as pd


def get_logger(name: str = None, level=logging.INFO):
    """
    Sets up the logger handlers for jupyter notebook, ipython or python.

    Separate initialization each time is needed to ensure that logger is set when calling from subprocess
    (e.g. joblib.Parallel)

    :param name: name of the logger. If None, will return root logger.
    :param level: Log level (default - INFO)
    :return: logger with correct handlers
    """
    logger = logging.getLogger(name)
    logger.handlers = []
    stdout = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    stdout.setFormatter(fmt)
    stdout.setLevel(level)
    logger.addHandler(stdout)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def load_artefact(artefact_path: Path, type_: str):
    if type_ == "pickle":
        with open(artefact_path, 'rb') as f:
            artefact = pickle.load(f)
    elif type_ == 'numpy':
        artefact_path = str(artefact_path).split(".")[0]
        artefact_path = str(artefact_path).split(".")[0]
        print(artefact_path, str(artefact_path), str(artefact_path).split("."))
        artefact_path = artefact_path + ".npy"
        with open(artefact_path, 'wb') as f:
            artefact = np.load(f)
    return artefact


def save_artefact(artefact, artefact_path: Path, type_: str):
    if type_ == "pickle":
        with open(artefact_path, 'wb') as f:
            pickle.dump(artefact, f)
    elif type_ == 'numpy':
        artefact_path = str(artefact_path).split(".")[0]
        print(artefact_path, str(artefact_path), str(artefact_path).split("."))
        artefact_path = artefact_path + ".npy"
        with open(artefact_path, 'wb') as f:
            np.save(f, artefact)


def get_user_product_frequencies(x: pd.DataFrame) -> pd.DataFrame:
    user_reordered_products = x.query("reordered == 1").groupby(["user_id"])["product_id"].apply(list)
    frequencies = user_reordered_products.apply(lambda x: Counter(x).most_common()).to_frame()
    frequencies["total_num"] = frequencies["product_id"].apply(lambda x: sum([xx[1] for xx in x]))
    frequencies["relative_freqs"] = frequencies.apply(get_relative_frequencies, axis=1)
    return frequencies


def get_relative_frequencies(row: pd.Series):
    freqs = row["product_id"]
    total_num = row["total_num"]
    rf = [(f[0], f[1] / total_num) for f in freqs]
    return rf


def filter_products_by_frequency(user_prod_frequencies: pd.DataFrame, frequency: float) -> pd.Series:
    user_prod_frequencies = user_prod_frequencies.copy()
    prediction = user_prod_frequencies["relative_freqs"].apply(
        lambda x: np.array([xx[0] for xx in x if xx[1] > frequency])
    )
    return prediction


def get_top_k(scores: np.ndarray, k: int):
    scores *= -1
    k_ = k + 1

    ind = np.argpartition(scores, k_, axis=-1)
    ind = np.take(ind, np.arange(k_), axis=-1)  # k non-sorted indices
    input = np.take_along_axis(scores, ind, axis=-1)  # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input)

    input *= -1
    ind = np.take_along_axis(ind, ind_part, axis=-1)
    return ind[:, 1:k_]


def predict_products_from_mapping(prod_ids: List[int], similar_products_map: Dict[int, List[int]]) -> List[int]:
    result = []
    for prod_id in prod_ids:
        result.extend(similar_products_map[prod_id])
    return result
