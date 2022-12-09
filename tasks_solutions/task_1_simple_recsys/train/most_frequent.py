from tqdm import tqdm

import numpy as np
import pandas as pd

from tasks_solutions.task_0_metrics.metrics import mean_f1_score
from tasks_solutions.task_1_simple_recsys.data_preparation import get_last_order_for_users
from utils.utils import get_logger, get_user_product_frequencies, filter_products_by_frequency

LOGGER = get_logger("MostFrequentProductsModel")

# todo: current split is bullshit, it should be TS-split.
# todo: estimate product frequencies on train orders, predict products with most_frequent > threshold


class MostFrequentProductsModel:
    """
    That model just estimates frequencies of the reordered products for each user and predicts them
    if their
    """

    def __init__(self):
        self.relative_frequencies_thresholds = np.linspace(.0, 1, 100)
        self.relative_freq_threshold = 0.030303030303030304

    def train(self, orders_history: pd.DataFrame):

        X_valid = get_last_order_for_users(order_detailed=orders_history)
        y_valid = X_valid[["user_id", "reordered", "product_id"]].copy()
        y_valid = y_valid.query("reordered == 1").groupby("user_id")["product_id"].apply(np.array)

        X = orders_history.drop(index=X_valid.index)

        user_prod_frequencies = get_user_product_frequencies(x=X)
        pred_users = set(user_prod_frequencies.index)
        gt_users = set(y_valid.index)
        empty_reordered = pred_users.difference(gt_users)

        y_valid = y_valid.append(
            pd.Series([np.array([]) for _ in range(len(empty_reordered))], index=empty_reordered)
        ).sort_index()

        rft_metrics_dict = dict()
        for rft in tqdm(self.relative_frequencies_thresholds):
            prediction = filter_products_by_frequency(
                user_prod_frequencies=user_prod_frequencies, frequency=rft
            )

            metric = self.evaluate(y_true=y_valid, y_pred=prediction)
            rft_metrics_dict[rft] = metric

        LOGGER.info(f"Estimated these MAP@K for thresholds: {rft_metrics_dict}")

        # get threshold with maximum metric value:
        selected_threshold = sorted(rft_metrics_dict.items(), key=lambda x: x[1])[-1]
        self.relative_freq_threshold = selected_threshold
        LOGGER.info(f"The best threshold chosen: {self.relative_freq_threshold}")

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: pd.Series) -> float:
        f1_score = mean_f1_score(y_true=y_true, y_pred=y_pred)
        return f1_score

    def predict(self, orders_history: pd.DataFrame, query_input: pd.DataFrame):
        orders_history = orders_history.query("user_id in @query_input.user_id")
        user_prod_frequencies = get_user_product_frequencies(x=orders_history)

        prediction = filter_products_by_frequency(
            user_prod_frequencies=user_prod_frequencies, frequency=self.relative_freq_threshold
        )

        prediction = query_input[["user_id", "order_id"]].merge(
            prediction.to_frame(),
            how='left',
            on=['user_id']
        )
        prediction = prediction.rename(columns={'relative_freqs': 'products'})[["order_id", "products"]]
        prediction["products"] = prediction["products"].fillna('').apply(
            lambda x: ' '.join([str(int(e)) for e in set(x)])
        ).replace({'': 'None'})
        return prediction
