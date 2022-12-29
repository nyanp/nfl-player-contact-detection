import json
import pickle
import time
from contextlib import contextmanager

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

    def to_file(self, filename: str):
        model = {
            "boosters": [b.model_to_string() for b in self.booster.boosters],
            "best_iteration": self.booster.best_iteration,
            "threshold_1": self.threshold_1,
            "threshold_2": self.threshold_2
        }

        with open(f"{filename}_model.json", "w") as f:
            json.dump(model, f)

        with open(f"{filename}_encoder.bin", "wb") as f:
            pickle.dump(self.encoders, f)

    @classmethod
    def from_file(cls, filename: str):

        with open(f"{filename}_model.json", "r") as f:
            model = json.load(f)

        cvbooster = lgb.CVBooster()
        cvbooster.boosters = [lgb.Booster(model_str=b)
                              for b in model["boosters"]]
        cvbooster.best_iteration = model["best_iteration"]
        for b in cvbooster.boosters:
            b.best_iteration = cvbooster.best_iteration

        with open(f"{filename}_encoder.bin", "rb") as f:
            encoders = pickle.load(f)

        return cls(
            cvbooster,
            encoders,
            model["threshold_1"],
            model["threshold_2"])


def make_oof(cvbooster, X_train: np.ndarray, y_train: pd.Series, split):
    oof = np.zeros(len(X_train))

    for booster, (idx_train, idx_valid) in zip(cvbooster.boosters, split):
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
    plt.savefig('importance.png')
    wandb.log(wandb.Image('importance.png'))
    plt.show()
