import os
os.chdir('../..')

import pandas as pd
from tasks_solutions.meta import TASK_0_DATASET, TASK_0_EXPECTED_SCORES
import tasks_solutions.task_0_metrics.metrics as metrics


if __name__ == '__main__':
    dataset = pd.read_parquet(TASK_0_DATASET)
    metrics_dict = dict()
    metrics_dict['hitrate_at_3'] = metrics.hitrate_score(dataset['ground_truth'], dataset['prediction'])
    metrics_dict['hitrate_at_5'] = metrics.hitrate_score(dataset['ground_truth'], dataset['prediction'], at_k=5)
    metrics_dict['map_at_3'] = metrics.mean_average_precision_score(dataset['ground_truth'], dataset['prediction'])
    metrics_dict['map_at_5'] = metrics.mean_average_precision_score(
        dataset['ground_truth'], dataset['prediction'], at_k=5
    )
    metrics_dict['ndcg_at_3'] = metrics.ndcg_score(dataset['ground_truth'], dataset['prediction'])
    metrics_dict['ndcg_at_5'] = metrics.ndcg_score(dataset['ground_truth'], dataset['prediction'], at_k=5)

    print(f"{'Metrics comparison':-^41}\n{'Expected:':>28} {'Estimated:':>12}")
    for mtrc_name, expected_value in TASK_0_EXPECTED_SCORES.items():
        estimated_value = metrics_dict.get(mtrc_name)
        if estimated_value is not None:
            print(f"{mtrc_name.capitalize()}: {expected_value:>{25-len(mtrc_name)}} {estimated_value:>8}")
    print(f"{'':-^41}")
