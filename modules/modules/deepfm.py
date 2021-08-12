import pandas as pd
import numpy as np

import dataclasses
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

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
        return nn_input, dnn_feat_cols, linear_feat_cols, feat_names


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
      

def train_deepfm(feats, feat_names, x, y):
    deepfm = DeepFmModel(feats, feats, feat_names)
    train_set, test_set = train_test_split(x, test_size=0.2)
    deepfm.train(train_set, target_values=y[:len(train_set)])
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