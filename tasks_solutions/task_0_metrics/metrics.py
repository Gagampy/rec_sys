import pandas as pd
import numpy as np


def hitrate_score_sample(y_true: np.ndarray, y_pred: np.ndarray, at_k: int = 3) -> int:
    if len(set(y_true).intersection(y_pred[:at_k])) > 0:
        return 1
    return 0


def hitrate_score(y_true: pd.Series, y_pred: pd.Series, at_k: int = 3) -> float:
    return round(np.mean([hitrate_score_sample(yt, yp, at_k) for yt, yp in zip(y_true, y_pred)]), 3)


def mean_average_precision_score(y_true: pd.Series, y_pred: pd.Series, at_k: int = 5) -> float:
    average_precisions = []
    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        average_precisions.append(average_precision_score(y_true_sample, y_pred_sample, at_k=at_k))

    mean_average_precision = round(np.mean(average_precisions), 3)
    return mean_average_precision


def average_precision_score(y_true_sample: np.ndarray, y_pred_sample: np.ndarray, at_k: int = 5) -> float:
    at_k = min(y_true_sample.shape[0], y_pred_sample.shape[0], at_k)

    precision_at_k = 0
    relevant_num_at_i = 0
    for i in range(at_k):
        is_relevant = y_pred_sample[i] in y_true_sample

        if is_relevant:
            relevant_num_at_i += 1
            precision_at_k += relevant_num_at_i / (i + 1)

    if relevant_num_at_i == 0:
        return 0

    average_precision = precision_at_k / relevant_num_at_i
    return average_precision


def ndcg_score(
        y_true: pd.Series,
        y_pred: pd.Series,
        at_k: int = 3,
) -> float:
    ndcg = []
    idcg = sum([1 / np.log2(i + 2) for i in range(at_k)])
    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        dcg = sum([
            1 / np.log2(i + 2)
            for i, elem in enumerate(y_pred_sample[:at_k])
            if elem in y_true_sample
        ])
        ndcg.append(dcg / idcg)
    return round(np.mean(ndcg), 3)


def mean_f1_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    f1_scores = []
    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        f1_scores.append(f1_score(y_true_sample, y_pred_sample))
    mean_f1 = round(np.mean(f1_scores), 3)
    return mean_f1


def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    y_pred_set = set(y_pred)
    y_true_set = set(y_true)
    n_intersected = len(y_true_set.intersection(y_pred_set))
    recall = n_intersected / (len(y_true_set) + 0.01)
    precision = n_intersected / (len(y_pred_set) + 0.01)
    return round((2 * recall * precision) / (recall + precision + 0.01), 3)





