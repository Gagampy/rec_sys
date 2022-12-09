from itertools import accumulate
from typing import List

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from tasks_solutions.meta import TASK_1_PATH_TO_USER_VECTORS, TASK_1_PATH_TO_PRODUCT_VECTORS, \
    TASK_1_PATH_TO_SIGMA_VECTORS, TASK_1_PATH_TO_USER_IDS, TASK_1_PATH_TO_PRODUCT_IDS
from utils.utils import get_logger, load_artefact, save_artefact, get_user_product_frequencies, \
    filter_products_by_frequency, get_top_k, predict_products_from_mapping

LOGGER = get_logger("SVDRecommender")


class SVDRecommender:

    explained_var_threshold = 0.90
    n_components = 175
    relative_freq_threshold = 0.030303030303030304
    similarity_thresholds = np.linspace(.5, 1, 50)
    top_k_product_based = 1

    def train(self, X: pd.DataFrame, **kwargs):
        # n_components = self.optimize_n_components(X)
        n_components = self.n_components
        LOGGER.info(f"Found {n_components} components as optimal out of {X.shape[1]} total num.")
        u, sigma, v = randomized_svd(X, n_components=n_components)
        self.save_artefacts(users=u, sigma=sigma, products=v)

    def optimize_n_components(self, X: pd.DataFrame):
        """Finds num of components equals to the 90% of the explained variance."""
        LOGGER.info(f"Started to searching components num for {self.explained_var_threshold} of explained Variance.")

        expl_variances = TruncatedSVD(n_components=self.n_components).fit(X=X).explained_variance_ratio_
        LOGGER.info(f"Finished.")
        LOGGER.info(f"Explained variances: {expl_variances}")
        cumulative_variances = list(accumulate(expl_variances, lambda x, y: x + y))
        LOGGER.info(f"Cumulative variances: {cumulative_variances}")
        return self.n_components

    @staticmethod
    def save_artefacts(users: np.ndarray, sigma: np.ndarray, products: np.ndarray):
        LOGGER.info(f"Saving SVD vectors.")
        paths = [TASK_1_PATH_TO_USER_VECTORS, TASK_1_PATH_TO_SIGMA_VECTORS, TASK_1_PATH_TO_PRODUCT_VECTORS]
        objs = [users, sigma, products]
        for path, obj in zip(paths, objs):
            save_artefact(artefact=obj, artefact_path=path, type_='pickle')

    def predict(
            self, orders_history: pd.DataFrame, query_input: pd.DataFrame, user_sim: bool=False
    ):
        user_ids = load_artefact(artefact_path=TASK_1_PATH_TO_USER_IDS, type_='pickle')
        product_ids = load_artefact(artefact_path=TASK_1_PATH_TO_PRODUCT_IDS, type_='pickle')

        vectors_path = TASK_1_PATH_TO_USER_VECTORS if user_sim else TASK_1_PATH_TO_PRODUCT_VECTORS
        vectors = load_artefact(artefact_path=vectors_path, type_='pickle')

        pred_func = self._get_prediction_by_user_similarity if user_sim else self._get_prediction_by_product_similarity
        prediction = pred_func(
            vectors=vectors,
            user_ids=user_ids,
            product_ids=product_ids,
            orders_history=orders_history,
            query_input=query_input
        )
        return prediction

    def _get_prediction_by_user_similarity(
            self, vectors: np.ndarray, user_ids: List[str], product_ids: List[str], user_ids_query: pd.Series
    ):
        """
        Find most similar users, then sample products from their orders as predictions for query users.
        :param vectors:
        :param user_ids:
        :param product_ids:
        :param user_ids_query:
        :return:
        """
        mask = user_ids_query.isin(user_ids)
        vectors_query = vectors[mask, :]
        self.pairwise_cosine_similarity

    def _get_prediction_by_product_similarity(
            self,
            vectors: np.ndarray,
            product_ids: List[str],
            orders_history: pd.DataFrame,
            query_input: pd.DataFrame,
            **kwargs
    ):
        """
        Find most similar products for user's most frequent products and sample them as predictions.
        :param vectors:
        :param user_ids:
        :param product_ids:
        :return:
        """
        user_prod_frequencies = get_user_product_frequencies(
            x=orders_history.query("user_id in @query_input.user_id and eval_set == 'prior'")
        )
        user_most_popular_prods = filter_products_by_frequency(
            user_prod_frequencies=user_prod_frequencies, frequency=self.relative_freq_threshold
        )

        # Take only most popular products IDs from prior to speed up pairwise calculation:
        query_product_ids = set([prod_id for prod_ids in user_most_popular_prods.to_list() for prod_id in prod_ids])
        query_products_mask = np.isin(product_ids,  list(query_product_ids))
        query_product_ids = np.array(product_ids)[query_products_mask]

        vectors = vectors.T
        query_vectors = vectors[query_products_mask, :]

        LOGGER.info(f"Calculating pairwise cosine similarities...")

        # Not sure about the space cost efficiency:
        similarities = cosine_similarity(X=query_vectors, Y=vectors)
        top_k_products_idx = get_top_k(similarities, self.top_k_product_based)

        product_ids = np.array(product_ids)
        top_k_products = np.apply_along_axis(lambda x: product_ids[x],  axis=0, arr=top_k_products_idx).tolist()
        similar_products_map = {prod: products for prod, products in zip(query_product_ids, top_k_products)}

        user_product_predicted = user_most_popular_prods.apply(
            predict_products_from_mapping, similar_products_map=similar_products_map
        )

        prediction = query_input[["user_id", "order_id"]].merge(
            user_product_predicted.to_frame(),
            how='left',
            on=['user_id']
        )
        prediction = prediction.rename(columns={'relative_freqs': 'products'})
        prediction = prediction.drop(columns=['user_id'])
        prediction['products'] = prediction['products'].apply(
            lambda x: ' '.join([str(xx) for xx in x]) if isinstance(x, list) else None
        )
        return prediction
