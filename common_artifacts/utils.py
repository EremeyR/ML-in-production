from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np


class OHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_for_encoding: list):
        self.col_for_enc = columns_for_encoding

        self.one_hot_encoders = {}
        for col in columns_for_encoding:
            self.one_hot_encoders[col] = OneHotEncoder(sparse=None)

    def fit(self, x, y=None):
        for col in self.col_for_enc:
            self.one_hot_encoders[col].fit(x.loc[:, [col]])
        print(self.one_hot_encoders.keys())
        return self

    def transform(self, x, y=None):
        x_ = x.copy()

        for col in self.col_for_enc:
            ohe_cols = self.one_hot_encoders[col].transform(x_.loc[:, [col]])
            x_[[f"ohe_{col}_{i}" for i in range(ohe_cols.shape[1])]] = ohe_cols

            del x_[col]

        return x_


def create_features(size):
    df = pd.DataFrame()
    df["age"] = np.random.randint(30, 90, size)
    df["sex"] = np.random.randint(0, 2, size)
    df["cp"] = np.random.randint(0, 4, size)
    df["trestbps"] = np.random.randint(90, 200, size)
    df["chol"] = np.random.randint(170, 200, size)
    df["fbs"] = np.random.randint(0, 2, size)
    df["restecg"] = np.random.randint(0, 3, size)
    df["thalach"] = np.random.randint(90, 200, size)
    df["exang"] = np.random.randint(0, 2, size)
    df["oldpeak"] = np.random.uniform(0, 5, size)
    df["slope"] = np.random.randint(0, 2, size)
    df["ca"] = np.random.randint(0, 4, size)
    df["thal"] = np.random.randint(0, 3, size)
    return df
