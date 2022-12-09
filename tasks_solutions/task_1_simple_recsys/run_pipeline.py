import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from fire import Fire
from scipy.sparse import save_npz, load_npz
from scipy.stats.contingency import crosstab

from tasks_solutions.meta import TASK_1_ARTEFACTS_PATH, TASK_1_PATH_TO_CROSSTABLE, TASK_1_PATH_TO_USER_IDS, \
    TASK_1_PATH_TO_PRODUCT_IDS
from tasks_solutions.task_1_simple_recsys.data_preparation import get_train_and_test_orders
from utils.utils import get_logger


LOGGER = get_logger("task_1_run_script")


def run_pipeline(
    model_name: str,
    pipeline_type: str = 'predict'
):
    orders_detailed, orders_detailed_test = get_train_and_test_orders()

    if model_name == 'most_frequent':
        from tasks_solutions.task_1_simple_recsys.train.most_frequent import MostFrequentProductsModel

        model = MostFrequentProductsModel()
        folder_name = model_name

        if pipeline_type in ('train', 'end2end'):
            model.train(orders_detailed)

    elif model_name in ('svd_product_sim', 'svd_user_sim'):
        from tasks_solutions.task_1_simple_recsys.train.svd_recommender import SVDRecommender

        model = SVDRecommender()
        folder_name = "svd"

        if pipeline_type in ('train', 'end2end'):
            X, user_ids, product_ids = get_crosstab_artefacts(orders_detailed=orders_detailed)
            model.train(X=X)
    else:
        raise ValueError("Unrecognized model.")

    if pipeline_type in ('predict', 'end2end'):
        prediction = model.predict(orders_history=orders_detailed, query_input=orders_detailed_test)

        assert len(orders_detailed_test.order_id.unique()) == len(prediction.order_id.unique()), \
            print(len(orders_detailed_test.order_id.unique()), len(prediction.order_id.unique()))

        prediction.to_csv(
            TASK_1_ARTEFACTS_PATH / folder_name / f'prediction_{model_name}.csv', index=False, encoding='utf-8'
        )


def get_crosstab_artefacts(orders_detailed: pd.DataFrame):
    """Loads USER-PRODUCTS table if it is exists. Otherwise, calculates, saves and returns it."""
    X, user_ids, product_ids = load_crosstab_artefacts(
        path_to_crosstable=TASK_1_PATH_TO_CROSSTABLE,
        user_ids_filepath=TASK_1_PATH_TO_USER_IDS,
        product_ids_filepath=TASK_1_PATH_TO_PRODUCT_IDS
    )
    if X is None:
        LOGGER.info("Crosstab is not found, calculating it...")
        X, user_ids, product_ids = calculate_crosstab(orders_detailed)
        save_npz(file=TASK_1_PATH_TO_CROSSTABLE, matrix=X)
        pickle_id_vector(user_ids, filepath=TASK_1_PATH_TO_USER_IDS)
        pickle_id_vector(product_ids, filepath=TASK_1_PATH_TO_PRODUCT_IDS)
    return X, user_ids, product_ids


def load_crosstab_artefacts(path_to_crosstable: Path, user_ids_filepath: Path, product_ids_filepath: Path):
    if path_to_crosstable.exists() and user_ids_filepath.exists() and product_ids_filepath.exists():
        return load_npz(path_to_crosstable), load_id_vector(user_ids_filepath), load_id_vector(product_ids_filepath)
    return None, None, None


def calculate_crosstab(orders_detailed: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (user_ids, product_ids), X = crosstab(
        orders_detailed["user_id"].values, orders_detailed["product_id"].values, sparse=True
    )
    return X, user_ids, product_ids


def pickle_id_vector(vector: np.ndarray, filepath: Path):
    with open(filepath, 'wb') as f:
        pickle.dump(vector, f)


def load_id_vector(filepath: Path):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# svd_product_sim
if __name__ == "__main__":
    Fire(run_pipeline)
