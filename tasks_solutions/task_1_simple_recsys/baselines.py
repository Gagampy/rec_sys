import pandas as pd


def get_last_reordered_products(order_detailed: pd.DataFrame, test_orders: pd.DataFrame):
    """
    Simply adds a product to the prediction, if it was reordered by the user in the last order.
    The result is saved.

    :param order_detailed: Dataframe with information about all orders and their features
    :param test_orders: Dataframe with user IDs to generate predictions for
    """
    test_history = order_detailed[(order_detailed.user_id.isin(test_orders.user_id))]
    last_orders = test_history.groupby('user_id')['order_number'].max()

    predictions = pd.merge(
            left=pd.merge(
                    left=last_orders.reset_index(),
                    right=test_history[test_history.reordered == 1],
                    how='left',
                    on=['user_id', 'order_number']
                )[['user_id', 'product_id']],
            right=test_orders[['user_id', 'order_id']],
            how='left',
            on='user_id'
        ).fillna(-1).groupby('order_id')['product_id'].apply(
            lambda x: ' '.join([str(int(e)) for e in set(x)])
        ).reset_index().replace(to_replace='-1', value='None')

    predictions.columns = ['order_id', 'products']
    predictions.to_csv("../artefacts/task_1/baseline/predictions.csv", encoding='utf-8', index=False)




