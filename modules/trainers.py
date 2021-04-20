import typing as t

import pandas as pd
import surprise
from sklearn.model_selection import KFold
from surprise import Dataset, Reader, SVD, accuracy, KNNBasic
from surprise.model_selection import cross_validate, train_test_split

from . import evaluator


def _rating_matrix_to_long_table(rating_matrix) -> pd.DataFrame:
    df = pd.DataFrame(rating_matrix)
    df["user_id"] = df.index
    return df.melt(id_vars=["user_id"], var_name="item_id", value_name="rating")


class SvdTrainTestExecutor(evaluator.TrainTestExecutorABC):
    def __init__(self, config: t.Optional[dict] = None) -> None:
        self._config = config or {}

    @property
    def model_name(self) -> str:
        return "svd"

    @property
    def config(self) -> dict:
        return self._config

    def __call__(self, rating_matrix: pd.DataFrame, test_size: float) -> dict:
        long_table  = _rating_matrix_to_long_table(rating_matrix)
        dataset = surprise.Dataset.load_from_df(long_table[['user_id', 'item_id', 'rating']], surprise.Reader(rating_scale=(0, 2)))
        train_set, test_set = train_test_split(dataset, test_size=test_size)

        algo = surprise.SVD(**self._config)
        algo.fit(train_set)
        predictions = algo.test(test_set)
        return {"test_rmse": [surprise.accuracy.rmse(predictions)]}


class KnnTrainTestExecutor(evaluator.TrainTestExecutorABC):
    def __init__(self, config: t.Optional[dict] = None) -> None:
        self._config = config or {}

    @property
    def model_name(self) -> str:
        return "knn"

    @property
    def config(self) -> dict:
        return self._config

    def __call__(self, rating_matrix: pd.DataFrame, test_size: float) -> dict:
        long_table  = _rating_matrix_to_long_table(rating_matrix)
        dataset = surprise.Dataset.load_from_df(long_table[['user_id', 'item_id', 'rating']], surprise.Reader(rating_scale=(0, 2)))
        train_set, test_set = train_test_split(dataset, test_size=test_size)

        algo = surprise.KNNBasic(**self._config)
        algo.fit(train_set)
        predictions = algo.test(test_set)
        return {"test_rmse": [accuracy.rmse(predictions)]}
