import json
import pickle
import time
from contextlib import contextmanager
from typing import Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import wandb


@contextmanager
def timer(name: str):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f"[{name}] {elapsed:.3f}sec")


def reduce_dtype(df):
    for c in df.columns:
        if df[c].dtype.name == "float64":
            df[c] = df[c].astype(np.float32)
    return df


class LabelEncoders:
    def __init__(self):
        self.encoders = {}

    def fit(self, df):
        for c in df:
            if df[c].dtype.name == "object":
                enc = self.encoders.get(c, LabelEncoder())
                enc.fit(df[c])
                self.encoders[c] = enc

    def transform(self, df):
        for c in df:
            if c in self.encoders:
                df[c] = self.encoders[c].transform(df[c])
        return df

    def fit_one(self, s):
        enc = self.encoders.get(s.name, LabelEncoder())
        enc.fit(s)
        self.encoders[s.name] = enc

    def transform_one(self, s):
        if s.name in self.encoders:
            return self.encoders[s.name].transform(s)
        else:
            return s

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def fit_transform_one(self, s):
        self.fit_one(s)
        return self.transform_one(s)


class LGBMSerializer:
    def __init__(self,
                 booster: lgb.CVBooster,
                 encoders: LabelEncoders,
                 threshold_1: float,
                 threshold_2: float):
        self.booster = booster
        self.encoders = encoders
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2

    def to_file(self, filename: str, save_dir: Optional[str] = None):
        save_dir = save_dir or '.'
        model = {
            "boosters": [b.model_to_string() for b in self.booster.boosters],
            "best_iteration": self.booster.best_iteration,
            "threshold_1": self.threshold_1,
            "threshold_2": self.threshold_2
        }

        with open(f"{save_dir}/{filename}_model.json", "w") as f:
            json.dump(model, f)

        with open(f"{save_dir}/{filename}_encoder.bin", "wb") as f:
            pickle.dump(self.encoders, f)

    @classmethod
    def from_file(cls, filename: str, save_dir: Optional[str] = None):
        save_dir = save_dir or '.'

        with open(f"{save_dir}/{filename}_model.json", "r") as f:
            model = json.load(f)

        cvbooster = lgb.CVBooster()
        cvbooster.boosters = [lgb.Booster(model_str=b)
                              for b in model["boosters"]]
        cvbooster.best_iteration = model["best_iteration"]
        for b in cvbooster.boosters:
            b.best_iteration = cvbooster.best_iteration

        with open(f"{save_dir}/{filename}_encoder.bin", "rb") as f:
            encoders = pickle.load(f)

        return cls(
            cvbooster,
            encoders,
            model["threshold_1"],
            model["threshold_2"])


def make_oof(boosters, X_train: np.ndarray, y_train: pd.Series, split):
    oof = np.zeros(len(X_train))

    for booster, (idx_train, idx_valid) in zip(boosters, split):
        y_pred = booster.predict(X_train[idx_valid])
        oof[idx_valid] = y_pred
        print(f"{roc_auc_score(y_train.iloc[idx_valid].values, y_pred)}")

    return oof


def plot_importance(cvbooster, figsize=(12, 20)):
    raw_importances = cvbooster.feature_importance(importance_type='gain')
    feature_name = cvbooster.boosters[0].feature_name()
    importance_df = pd.DataFrame(data=raw_importances,
                                 columns=feature_name)
    # order by average importance across folds
    sorted_indices = importance_df.mean(
        axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    # plot top-n
    PLOT_TOP_N = 100
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.tight_layout()
    plt.savefig('importance.png')
    wandb.log({"importance": wandb.Image('importance.png')})
    plt.show()


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df