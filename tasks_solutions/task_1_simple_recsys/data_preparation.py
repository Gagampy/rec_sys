from functools import partial

import pandas as pd
import numpy as np

from tasks_solutions.meta import TASK_1_DATASET


def get_train_and_test_orders():

    # 'Products' datasets:
    products = pd.read_csv(TASK_1_DATASET / 'products.csv/products.csv')
    aisles = pd.read_csv(TASK_1_DATASET / 'aisles.csv/aisles.csv')
    departments = pd.read_csv(TASK_1_DATASET / 'departments.csv/departments.csv')

    # Orders datasets:
    orders = pd.read_csv(TASK_1_DATASET / 'orders.csv/orders.csv')
    orderProductsTrain = pd.read_csv(TASK_1_DATASET / 'order_products__train.csv/order_products__train.csv')
    orderProductsPrior = pd.read_csv(TASK_1_DATASET / 'order_products__prior.csv/order_products__prior.csv')

    goods = add_departments_and_aisles_info(products=products, departments=departments, aisles=aisles)

    orders_detailed_train = combine_orders_info(orders=orders, orders_products=orderProductsTrain)
    orders_detailed_train = merge_orders_and_goods(order_detailed=orders_detailed_train, goods=goods)

    orders_detailed_prior = combine_orders_info(orders=orders, orders_products=orderProductsPrior)
    orders_detailed_prior = merge_orders_and_goods(order_detailed=orders_detailed_prior, goods=goods)

    orders_detailed = concat_prior_and_train_orders(
        order_detailed_train=orders_detailed_train, order_detailed_prior=orders_detailed_prior
    )
    return orders_detailed, orders.query("eval_set == 'test'")


def add_departments_and_aisles_info(products: pd.DataFrame, departments: pd.DataFrame, aisles: pd.DataFrame):
    # combine aisles, departments and products (left joined to products)
    goods = pd.merge(
        left=pd.merge(
            left=products, right=departments, how='left'
        ),
        right=aisles,
        how='left'
    )
    # to retain '-' and make product names more "standard"
    goods.product_name = goods.product_name.str.replace(' ', '_').str.lower()
    return goods


def combine_orders_info(orders: pd.DataFrame, orders_products: pd.DataFrame):
    # initialize it with train dataset
    order_details = pd.merge(
        left=orders_products,
        right=orders,
        how='left',
        on='order_id'
    ).apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))
    return order_details


def merge_orders_and_goods(order_detailed: pd.DataFrame, goods: pd.DataFrame):
    # add order hierarchy
    order_detailed = pd.merge(
        left=order_detailed.copy(),
        right=goods[['product_id',
                     'aisle_id',
                     'department_id']].apply(partial(pd.to_numeric,
                                                     errors='ignore',
                                                     downcast='integer')),
        how='left',
        on='product_id'
    )
    return order_detailed


def concat_prior_and_train_orders(order_detailed_train: pd.DataFrame, order_detailed_prior: pd.DataFrame):
    order_detailed_train = order_detailed_train.copy()
    indexes = np.linspace(0, len(order_detailed_prior), num=10, dtype=np.int32)

    for i in range(len(indexes) - 1):
        order_detailed_train = pd.concat([order_detailed_train, order_detailed_prior.iloc[indexes[i]:indexes[i+1], :]])
    return order_detailed_train


def get_last_order_for_users(order_detailed: pd.DataFrame) -> pd.DataFrame:
    mask = order_detailed.groupby("user_id")["order_number"].transform(max) == order_detailed['order_number']
    last_orders = order_detailed.loc[mask]
    return last_orders
