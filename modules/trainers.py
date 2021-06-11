import typing as t
import tensorflow as tf

import pandas as pd
import surprise
from sklearn.model_selection import KFold
from surprise import Dataset, Reader, SVD, accuracy, KNNBasic
import surprise.model_selection as sm
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


from . import evaluator
from . import models


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

    def __call__(self, rating_matrix: pd.DataFrame, test_size: float, sample_size: float) -> evaluator.TestSetError:
        long_table  = _rating_matrix_to_long_table(rating_matrix)
        long_table = long_table.sample(frac=sample_size)
        dataset = surprise.Dataset.load_from_df(long_table[['user_id', 'item_id', 'rating']], surprise.Reader(rating_scale=(0, 1)))
        train_set, test_set = sm.train_test_split(dataset, test_size=test_size)

        algo = surprise.SVD(**self._config)
        algo.fit(train_set)
        predictions = algo.test(test_set)
        return evaluator.TestSetError(
            rmse=surprise.accuracy.rmse(predictions),
            mae=surprise.accuracy.mae(predictions)
        )


class KnnTrainTestExecutor(evaluator.TrainTestExecutorABC):
    def __init__(self, config: t.Optional[dict] = None) -> None:
        self._config = config or {}

    @property
    def model_name(self) -> str:
        return "knn"

    @property
    def config(self) -> dict:
        return self._config

    def __call__(self, rating_matrix: pd.DataFrame, test_size: float, sample_size: float) -> evaluator.TestSetError:
        long_table  = _rating_matrix_to_long_table(rating_matrix)
        long_table = long_table.sample(frac=sample_size)
        dataset = surprise.Dataset.load_from_df(long_table[['user_id', 'item_id', 'rating']], surprise.Reader(rating_scale=(0, 1)))
        train_set, test_set = sm.train_test_split(dataset, test_size=test_size)

        algo = surprise.KNNBasic(**self._config)
        algo.fit(train_set)
        predictions = algo.test(test_set)

        return evaluator.TestSetError(
            rmse=surprise.accuracy.rmse(predictions),
            mae=surprise.accuracy.mae(predictions)
        )


class AutoRecTrainTestExecutor(evaluator.TrainTestExecutorABC):
    def __init__(self, config: t.Optional[dict] = None) -> None:
        self._config = config or {}

    @property
    def model_name(self) -> str:
        return "autorec"

    @property
    def config(self) -> dict:
        return self._config

    def __call__(self, rating_matrix: pd.DataFrame, test_size: float, sample_size: float) -> evaluator.TestSetError:
        long_table  = _rating_matrix_to_long_table(rating_matrix)
        long_table = long_table.sample(frac=sample_size)
        train_matrix, test_matrix, n_users, n_items = self._transform_long_table_to_sparse_matrix(long_table, test_size)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as sess:
            model = models.IAutoRec(sess, n_users, n_items, **self._config)
            model.build_network()
            errors_log = model.execute(train_matrix, test_matrix)
            return evaluator.TestSetError(
                rmse=errors_log[-1]["rmse"],
                mae=errors_log[-1]["mae"],
            )

    def _transform_long_table_to_sparse_matrix(self, df, test_size):
        n_users = df.user_id.unique().shape[0]
        n_items = df.item_id.unique().shape[0]

        train_data, test_data = train_test_split(df, test_size=test_size)
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)

        train_row = []
        train_col = []
        train_rating = []

        for line in train_data.itertuples():
            u = line[1]
            i = line[2]
            train_row.append(u)
            train_col.append(i)
            train_rating.append(line[3])
        train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

        test_row = []
        test_col = []
        test_rating = []
        for line in test_data.itertuples():
            test_row.append(line[1])
            test_col.append(line[2])
            test_rating.append(line[3])
        test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
        print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
        return train_matrix.todok(), test_matrix.todok(), n_users, n_items
