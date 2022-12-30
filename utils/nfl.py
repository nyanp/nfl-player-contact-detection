
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .general import reduce_dtype


TRACK_COLS = [
    "game_play",
    "nfl_player_id",
    "datetime",
    "step",
    "team",
    "position",
    "x_position",
    "y_position",
    "speed",
    "distance",
    "direction",
    "orientation",
    "acceleration",
    "sa"
]
TRAIN_COLS = [
    "game_play",
    "step",
    "nfl_player_id_1",
    "nfl_player_id_2",
    "contact",
    "datetime"
]

NON_FEATURE_COLS = [
    "contacgt_id",
    "game_play",
    "datetime",
    "step",
    "nfl_player_id_1",
    "nfl_player_id_2",
    "contact",
    "team_1",
    "team_2",
    "contact_id",
    # "position_1",
    # "position_2"
    # "direction_1",
    # "direction_2",
    "x_position_1",
    "x_position_2",
    "y_position_1",
    "y_position_2",
    "x_position_start_1",
    "x_position_start_2",
    "y_position_start_1",
    "y_position_start_2",

    "x_position_future5_1",
    "x_position_future5_2",
    "y_position_future5_1",
    "y_position_future5_2",
    "x_position_past5_1",
    "x_position_past5_2",
    "y_position_past5_1",
    "y_position_past5_2",

    # "orientation_past5_1",
    # "direction_past5_1",
    # "orientation_past5_2",
    # "direction_past5_2",
]

class ModelSize(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


@dataclass
class Config:
    PROJECT: str = 'nfl2022-lgbm'
    EXP_NAME: str = 'exp000'
    INPUT: str = "../input/nfl-player-contact-detection"
    CACHE: str = "./"
    USE_PRETRAINED_MODEL: bool = False
    PRETRAINED_MODEL_PATH: str = "../input/nfl-baseline-oof"
    SPLIT_FILE_PATH: str = "../input/game_fold.csv"
    MODEL_SIZE: ModelSize = ModelSize.LARGE
    DEBUG: bool = False
    CAMARO_DF_PATH: Optional[str] = None
    KMAT_END_DF_PATH: Optional[str] = None
    KMAT_SIDE_DF_PATH: Optional[str] = None


def cast_player_id(df):
    # RAM消費を減らしたいので、Gを-1に置換して整数で持つ。
    if "nfl_player_id_2" in df.columns:
        df.loc[df["nfl_player_id_2"] == "G", "nfl_player_id_2"] = "-1"

    for c in ["nfl_player_id", "nfl_player_id_1", "nfl_player_id_2"]:
        if c in df.columns:
            df[c] = df[c].astype(np.int32)

    return df


def read_csv_with_cache(filename, cfg: Config, **kwargs):
    cache_filename = filename.split(".")[0] + ".f"
    cache_path = os.path.join(cfg.CACHE, cache_filename)
    if not os.path.exists(cache_path):
        df = pd.read_csv(os.path.join(cfg.INPUT, filename), **kwargs)
        df = reduce_dtype(cast_player_id(df))
        df.to_feather(cache_path)
    return pd.read_feather(cache_path)


def distance(x1, y1, x2, y2):
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))


def bbox_iou(df, view):
    xc1 = df[f"bbox_center_x_{view}_1"]
    yc1 = df[f"bbox_center_y_{view}_1"]
    xc2 = df[f"bbox_center_x_{view}_2"]
    yc2 = df[f"bbox_center_y_{view}_2"]
    w1 = df[f"width_{view}_1"]
    h1 = df[f"height_{view}_1"]
    w2 = df[f"width_{view}_2"]
    h2 = df[f"height_{view}_2"]

    overlap_x = np.minimum(xc1 + w1 / 2, xc2 + w2 / 2) - \
        np.maximum(xc1 - w1 / 2, xc2 - w2 / 2)
    overlap_y = np.minimum(yc1 + h1 / 2, yc2 + h2 / 2) - \
        np.maximum(yc1 - h1 / 2, yc2 - h2 / 2)

    intersection = overlap_x * overlap_y
    union = w1 * h1 + w2 * h2 - intersection
    iou = intersection / union

    iou.loc[(overlap_x < 0) | (overlap_y < 0)] = 0
    return iou


def merge_helmet(df, helmet, meta):
    start_times = meta[["game_play", "start_time"]].drop_duplicates()
    start_times["start_time"] = pd.to_datetime(start_times["start_time"])

    helmet = pd.merge(helmet,
                      start_times,
                      on="game_play",
                      how="left")

    # 追加
    helmet['xmin'] = helmet['left']
    helmet['ymin'] = helmet['top']
    helmet['xmax'] = helmet['left'] + helmet['width']
    helmet['ymax'] = helmet['top'] + helmet['height']
    bbox = np.array([helmet['xmin'],
                     helmet['ymin'],
                     helmet['xmax'],
                     helmet['ymax']]).T
    helmet['bbox_center_x'] = (bbox[:, 0] + bbox[:, 2]) / 2
    helmet['bbox_center_y'] = (bbox[:, 1] + bbox[:, 3]) / 2

    fps = 59.94
    helmet["datetime"] = helmet["start_time"] + \
        pd.to_timedelta(helmet["frame"] * (1 / fps), unit="s")
    helmet["datetime"] = pd.to_datetime(helmet["datetime"], utc=True)
    helmet["datetime_ngs"] = pd.DatetimeIndex(
        helmet["datetime"] + pd.to_timedelta(50, "ms")).floor("100ms").values
    helmet["datetime_ngs"] = pd.to_datetime(helmet["datetime_ngs"], utc=True)
    df["datetime_ngs"] = pd.to_datetime(df["datetime"], utc=True)

    # 追加
    feature_cols = ["width", "height", "bbox_center_x", "bbox_center_y"]

    helmet_agg = helmet.groupby(["datetime_ngs", "nfl_player_id", "view"]).agg({
        "width": "mean",
        "height": "mean",
        "bbox_center_x": "mean",
        "bbox_center_y": "mean",
    }).reset_index()

    for view in ["Sideline", "Endzone"]:
        helmet_ = helmet_agg[helmet_agg["view"] == view].drop("view", axis=1)

        helmet_global = helmet[helmet["view"] == view].groupby("datetime_ngs").agg(
            **{
                f"width_{view}_mean": pd.NamedAgg("width", "mean"),
                f"height_{view}_mean": pd.NamedAgg("height", "mean"),
                f"{view}_count": pd.NamedAgg("width", "count"),
            }).reset_index()

        helmet_global[f"aspect_{view}_mean"] = helmet_global[f"height_{view}_mean"] / \
            helmet_global[f"width_{view}_mean"]

        for postfix in ["_1", "_2"]:
            column_renames = {c: f"{c}_{view}{postfix}" for c in feature_cols}
            column_renames["nfl_player_id"] = f"nfl_player_id{postfix}"
            df = pd.merge(
                df,
                helmet_.rename(columns=column_renames),
                on=["datetime_ngs", f"nfl_player_id{postfix}"],
                how="left"
            )

        df = pd.merge(
            df,
            helmet_global,
            on="datetime_ngs",
            how="left"
        )

        df[f"bbox_iou_{view}"] = bbox_iou(df, view)

    # del df["datetime_ngs"]
    del df["datetime"]
    return reduce_dtype(df)


def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return cast_player_id(df)


def expand_helmet(cfg, df, phase="train"):
    helmet_cols = [
        "game_play",
        "view",
        "nfl_player_id",
        "frame",
        "left",
        "width",
        "top",
        "height"
    ]
    helmet = read_csv_with_cache(
        f"{phase}_baseline_helmets.csv", cfg, usecols=helmet_cols)
    meta = read_csv_with_cache(f"{phase}_video_metadata.csv", cfg)
    df = merge_helmet(df, helmet, meta)
    return df


def merge_tracking(df, tr, use_cols):
    key_cols = ["nfl_player_id", "step", "game_play"]
    use_cols = [c for c in use_cols if c in tr.columns]

    dst = pd.merge(
        df,
        tr[key_cols + use_cols].rename(columns={c: c + "_1" for c in use_cols}),
        left_on=["nfl_player_id_1", "step", "game_play"],
        right_on=key_cols,
        how="left"
    ).drop("nfl_player_id", axis=1)
    dst = pd.merge(
        dst,
        tr[key_cols + use_cols].rename(columns={c: c + "_2" for c in use_cols}),
        left_on=["nfl_player_id_2", "step", "game_play"],
        right_on=key_cols,
        how="left"
    ).drop("nfl_player_id", axis=1)

    return dst


def angle_diff(s1, s2):
    diff = s1 - s2

    return np.abs((diff + 180) % 360 - 180)


def add_contact_id(df):
    # Create contact ids
    df["contact_id"] = (
        df["game_play"] +
        "_" +
        df["step"].astype("str") +
        "_" +
        df["nfl_player_id_1"].astype("str") +
        "_" +
        df["nfl_player_id_2"].astype("str").str.replace(
            "-1",
            "G",
            regex=False))
    return df
