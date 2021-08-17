import pandas as pd
import numpy as np

import dataclasses
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from . import models


@dataclasses.dataclass
class DeepFmInputDataset:
    data: object
    dnn_feats: object
    linear_feats: object
    feat_names: object


class DeepFMDataLoader:
    def __init__(self, *, sparse_features, dense_features):
        self._sparse_feats = sparse_features
        self._dense_feats = dense_features
        
    def load(self, dataset):
        nn_input = pd.DataFrame()
        nn_input[self._sparse_feats] = dataset[self._sparse_feats]
        nn_input[self._dense_feats] = dataset[self._dense_feats]
        
        for feat in self._sparse_feats:
            encoder = LabelEncoder()
            nn_input[feat] = encoder.fit_transform(nn_input[feat])
            
        mms = MinMaxScaler(feature_range=(0,1))
        nn_input[self._dense_feats] = mms.fit_transform(nn_input[self._dense_feats])
        
        # problems may be here
        sparse_feature_columns = [
            SparseFeat(feat, vocabulary_size=nn_input[feat].nunique(), embedding_dim=4) 
            for i, feat in enumerate(self._sparse_feats)
        ]

        dense_feature_columns = [DenseFeat(feat, 1,) for feat in self._dense_feats]
        
        dnn_feat_cols = sparse_feature_columns + dense_feature_columns
        linear_feat_cols = sparse_feature_columns + dense_feature_columns
        
        feat_names = get_feature_names(linear_feat_cols + dnn_feat_cols)
        input_dataset = DeepFmInputDataset(
            data=nn_input,
            dnn_feats=dnn_feat_cols,
            linear_feats=linear_feat_cols,
            feat_names=feat_names
        )
        return input_dataset


def to_rating_long_table(dataset, predicted_response):
    result = pd.DataFrame()
    result["rating"] = predicted_response.reshape((len(predicted_response),))
    result["user_id"] = dataset["user_id"]
    result["item_id"] = dataset["item_id"]
    return result


class DeepFmWrapper:
    def __init__(self, data_loader):
        self._data_loader = data_loader
        self._deepfm = None 

    @property
    def data_loader(self):
        return self._data_loader

    def train_model(self, train_set, test_set):
        nn_train_input = self._data_loader.load(train_set)
        nn_test_input = self._data_loader.load(test_set)
        y = train_set["rating"].values
        
        merged_feats = merge_feats(nn_train_input.dnn_feats, nn_test_input.dnn_feats)
        deepfm = self._train(merged_feats, nn_train_input.feat_names, x=nn_train_input.data, y=y)
        self._deepfm = deepfm
        return deepfm

    def predict(self, test_dataset) -> None:
        nn_test_dataset = self._data_loader.load(test_dataset)
        y = self._deepfm.predict(nn_test_dataset.data)
        return to_rating_long_table(test_dataset, y)

    def _train(self, feats, feat_names, x, y):
        deepfm = models.DeepFmModel(feats, feats, feat_names)
        deepfm.train(x, target_values=y[:len(x)])
        return deepfm


def merge_feats(feats_a, feats_b):
    assert len(feats_a) == len(feats_b)
    merged = []
    for feat_a, feat_b in zip(feats_a, feats_b):
        if isinstance(feat_a, DenseFeat):
            continue
        if feat_a.vocabulary_size >= feat_b.vocabulary_size:
            merged.append(feat_a)
        else:
            merged.append(feat_b)
    return merged
            

class NNModelWrapper:
    def __init__(self, trained_nn):
        self._nn = trained_nn

    def predict_rating_matrix(self, nn_input, merged_df):
        y = self._nn.predict(nn_input)
        result = pd.DataFrame()
        result["rating"] = y.reshape((len(y),))
        result["user_id"] = merged_df["user_id"]
        result["item_id"] = merged_df["item_id"]
        output_matrix = result.pivot(index="user_id", columns="item_id", values="rating")
        return output_matrix