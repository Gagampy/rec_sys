from pathlib import Path


DATA_PATH = Path('../../data/')
ARTEFACTS_PATH = Path("../../artefacts")


# Task 0 utilities:
TASK_0_DATASET = DATA_PATH / Path('recsys_task0_dataset.parquet')
TASK_0_EXPECTED_SCORES = {
    'hitrate_at_3': 0.418,
    'hitrate_at_5': 0.492,
    'map_at_3': 0.325,
    'map_at_5': 0.333,
    'ndcg_at_3': 0.238,
    'ndcg_at_5': 0.223,
}

# Task 1 utilities:
TASK_1_DATASET = DATA_PATH / Path("instacart-market-basket-analysis/data/instacart-market-basket-analysis")
TASK_1_ARTEFACTS_PATH = ARTEFACTS_PATH / Path('task_1')
TASK_1_PATH_TO_CROSSTABLE = TASK_1_ARTEFACTS_PATH / Path("user_product_crosstable.npz")
TASK_1_PATH_TO_USER_IDS = TASK_1_ARTEFACTS_PATH / Path("user_ids.pkl")
TASK_1_PATH_TO_PRODUCT_IDS = TASK_1_ARTEFACTS_PATH / Path("product_ids.pkl")
TASK_1_PATH_TO_USER_VECTORS = TASK_1_ARTEFACTS_PATH / Path("svd/user_vectors.pkl")
TASK_1_PATH_TO_PRODUCT_VECTORS = TASK_1_ARTEFACTS_PATH / Path("svd/product_vectors.pkl")
TASK_1_PATH_TO_SIGMA_VECTORS = TASK_1_ARTEFACTS_PATH / Path("svd/sigma_vectors.pkl")


# Task 2 utilities:

# Task 3 utilities:
