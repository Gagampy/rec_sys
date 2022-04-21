from pathlib import Path

# https://kb.epam.com/display/EPMCBDCCDS/RecSys+course
DATA_PATH = Path('./data')


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

# Task 2 utilities:

# Task 3 utilities:
