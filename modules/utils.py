import typing as t

from surprise import KNNBasic, SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from modules import models

import pandas as pd


def reformat_evaluation_results_to_single_dataframe(eval_results_list: t.List[t.Dict]) -> pd.DataFrame:
    df = pd.DataFrame() 
    for eval_result_meta in eval_results_list:
        eval_results = [
            (r.a1, r.a2, r.test_error.rmse, r.test_error.mae, eval_result_meta["model_name"])
            for r in eval_result_meta["results"] 
        ]
        eval_results_df = pd.DataFrame(eval_results, columns=["a1", "a2", "rmse", "mae", "model"])
        df = pd.concat([df, eval_results_df])
    return df 


def group_points_by_minimum_error(points_df) -> pd.DataFrame:
    groupped = points_df[["a1", "a2", "rmse", "sample_size"]].groupby(
        by=["a1", "a2", "sample_size"], as_index=False
    ).min().join(
        points_df.set_index(["a1", "a2", "rmse", "sample_size"]), 
        on=["a1", "a2", "rmse", "sample_size"]
    )
    return groupped


def to_category_codes(series):
    return series.astype("category").cat.codes


def map_idx_to_matrix_indices(df):
    df["user_id"] = to_category_codes(df["user_id"])
    df["item_id"] = to_category_codes(df["item_id"])
    return df


def train_svd(df):
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

    svd_algo = SVD()
    error = cross_validate(svd_algo, data, cv=2)
    return error


def train_knn(df):
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

    svd_algo = KNNBasic()
    error = cross_validate(svd_algo, data, cv=2)
    return error

def train_autorec(df):
    _, error_log = models.train_test_autorec(df)
    return error_log