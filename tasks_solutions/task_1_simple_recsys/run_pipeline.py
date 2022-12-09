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
from utils.utils import get_logger, get_crosstab_artefacts

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


# svd_product_sim
if __name__ == "__main__":
    Fire(run_pipeline)
