from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd

from utils.nfl import distance, angle_diff
from utils.general import reduce_dtype


def add_bbox_features(df: pd.DataFrame) -> pd.DataFrame:

    for view in ["Sideline", "Endzone"]:
        df = add_bbox_center_distance(df, view)
        df = add_bbox_from_step0_feature(df, view)
        df = add_shift_feature(df, view)
        df = add_diff_features(df, view)
    return df


def add_bbox_center_distance(df: pd.DataFrame, view: str) -> pd.DataFrame:
    '''
    二人のplayerの中心間距離
    '''
    df[f'bbox_center_{view}_distance'] = distance(
        df[f'bbox_center_x_{view}_1'],
        df[f'bbox_center_y_{view}_1'],
        df[f'bbox_center_x_{view}_2'],
        df[f'bbox_center_y_{view}_2'])

    return df


def add_bbox_from_step0_feature(df: pd.DataFrame, view: str) -> pd.DataFrame:
    for postfix in ["1", "2"]:
        step0_columns = [f'bbox_center_x_{view}_{postfix}',
                         f'bbox_center_y_{view}_{postfix}']
        df = add_bbox_step0(df, step0_columns, postfix)
        df = add_distance_step0(df, step0_columns, view, postfix)
    return df


def add_bbox_step0(
        df: pd.DataFrame,
        step0_columns: List[str],
        postfix: str) -> pd.DataFrame:
    merge_key_columns = ["game_play", f"nfl_player_id_{postfix}"]

    use_columns = deepcopy(merge_key_columns)
    use_columns.extend(step0_columns)

    _df = df[df['step'] == 0].copy()
    _df = _df[use_columns].drop_duplicates()

    rename_columns = [i + '_start' for i in step0_columns]

    _df.rename(columns=dict(zip(step0_columns, rename_columns)),
               inplace=True)

    df = pd.merge(df,
                  _df,
                  how='left',
                  on=merge_key_columns)
    return df


def add_distance_step0(
        df: pd.DataFrame,
        step0_columns: List[str],
        view,
        postfix) -> pd.DataFrame:

    for column in step0_columns:
        df['diff_step_0_' + column] = df[column] - df[column + '_start']
        df['diff_step_0_' + column] = df[column] - df[column + '_start']

    df[f'distance_from_step0_{view}_{postfix}'] = distance(df[step0_columns[0]],
                                                           df[step0_columns[1]],
                                                           df[step0_columns[0] +
                                                               '_start'],
                                                           df[step0_columns[1] + '_start'])

    # 邪魔なので削除する
    df.drop(columns=[i + '_start' for i in step0_columns], inplace=True)

    return df


def add_shift_feature(df: pd.DataFrame, view: str) -> pd.DataFrame:
    result = []

    shift_columns = [f'distance_from_step0_{view}_1',
                     f'distance_from_step0_{view}_2',
                     f'bbox_center_{view}_distance',
                     f'bbox_center_x_{view}_1',
                     f'bbox_center_y_{view}_1',
                     # f'bbox_center_x_{view}_2',
                     # f'bbox_center_y_{view}_2',
                     ]
    for i in [-5, -3, -1, 1, 3, 5]:

        _tmp = df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])[
            shift_columns].shift(i).add_prefix(f'shift_{i}_')
        result.append(_tmp)

    result = pd.concat(result, axis=1)
    df = pd.concat([df, result], axis=1)
    return df


def add_diff_feature(
        df: pd.DataFrame,
        column_name: str,
        shift_amount: List[int]) -> pd.DataFrame:
    for shift in shift_amount:
        df[f'diff_{shift}_current_{column_name}'] = df[column_name] - \
            df[f'shift_{shift}_{column_name}']
    return df


def add_diff_features(df: pd.DataFrame, view: str) -> pd.DataFrame:
    shift_columns = [f'distance_from_step0_{view}_1',
                     f'distance_from_step0_{view}_2',
                     f'bbox_center_{view}_distance',
                     f'bbox_center_x_{view}_1',
                     f'bbox_center_y_{view}_1',
                     # f'bbox_center_x_{view}_2',
                     # f'bbox_center_y_{view}_2',
                     ]

    for column in shift_columns:
        df = add_diff_feature(df, column, [-5, -3, -1, 1, 3, 5])

    return df


def add_basic_features(df):
    df["dx"] = df["x_position_1"] - df["x_position_2"]
    # df["dy"] = df["y_position_1"] - df["y_position_2"]
    df["distance"] = distance(
        df["x_position_1"],
        df["y_position_1"],
        df["x_position_2"],
        df["y_position_2"])
    df["different_team"] = df["team_1"] != df["team_2"]
    df["anglediff_dir1_dir2"] = angle_diff(
        df["direction_1"], df["direction_2"])
    df["anglediff_ori1_ori2"] = angle_diff(
        df["orientation_1"], df["orientation_2"])
    df["anglediff_dir1_ori1"] = angle_diff(
        df["direction_1"], df["orientation_1"])
    df["anglediff_dir2_ori2"] = angle_diff(
        df["direction_2"], df["orientation_2"])
    return df


def add_tracking_agg_features(df, tracking):
    """トラッキングデータのstep内での単純集計。他プレイヤーの動きを見る"""

    team_agg = tracking.groupby(["game_play", "step", "team"]).agg(
        x_position_team_mean=pd.NamedAgg("x_position", "mean"),
        y_position_team_mean=pd.NamedAgg("y_position", "mean"),
        speed_team_mean=pd.NamedAgg("speed", "mean"),
        acceleration_team_mean=pd.NamedAgg("acceleration", "mean"),
        sa_team_mean=pd.NamedAgg("sa", "mean")
    )
    agg = tracking.groupby(["game_play", "step"]).agg(
        x_position_mean=pd.NamedAgg("x_position", "mean"),
        y_position_mean=pd.NamedAgg("y_position", "mean"),
        speed_mean=pd.NamedAgg("speed", "mean"),
        acceleration_mean=pd.NamedAgg("acceleration", "mean"),
        sa_mean=pd.NamedAgg("sa", "mean")
    )
    player_agg = tracking[tracking["step"] >= 0].groupby(["game_play", "nfl_player_id"]).agg(
        sa_player_mean=pd.NamedAgg("sa", "mean"),
        sa_player_max=pd.NamedAgg("sa", "max"),
        acceleration_player_mean=pd.NamedAgg("acceleration", "mean"),
        acceleration_player_max=pd.NamedAgg("acceleration", "max"),
        speed_player_mean=pd.NamedAgg("speed", "mean"),
        speed_player_max=pd.NamedAgg("speed", "max"),
    )

    for postfix in ["_1", "_2"]:
        df = pd.merge(
            df,
            team_agg.rename(
                columns={
                    c: c +
                    postfix for c in team_agg.columns}).reset_index(),
            left_on=[
                "game_play",
                "step",
                f"team{postfix}"],
            right_on=[
                "game_play",
                "step",
                "team"],
            how="left").drop(
            "team",
            axis=1)

        player_agg_renames = {c: c + postfix for c in player_agg.columns}
        player_agg_renames["nfl_player_id"] = f"nfl_player_id{postfix}"

        df = pd.merge(
            df,
            player_agg.reset_index().rename(columns=player_agg_renames),
            on=["game_play", f"nfl_player_id{postfix}"],
            how="left"
        )

    df = pd.merge(df, agg, on=["game_play", "step"], how="left")
    return df


def add_distance_around_player(df):
    """距離など、pairwiseに計算された特徴量を集計しなおす
    対象のプレイヤーの周りに他プレイヤーが密集しているか？
    """
    feature_cols = [
        "mean_distance_around_player",
        "min_distance_around_player",
        "std_distance_around_player",
        "idxmin_distance_aronud_player"
    ]

    stacked = pd.concat([
        df[["game_play", "step", "nfl_player_id_1",
            "nfl_player_id_2", "distance", "different_team"]],
        df[["game_play", "step", "nfl_player_id_2", "nfl_player_id_1", "distance", "different_team"]].rename(
            columns={"nfl_player_id_2": "nfl_player_id_1", "nfl_player_id_1": "nfl_player_id_2"}),
    ])

    def _arg_min(s):
        try:
            return np.nanargmin(s.values)
        except ValueError:
            return np.nan

    def _merge_stacked_df(df, s, postfix=""):
        s = s.groupby(["game_play", "step", "nfl_player_id_1"]).agg({
            "distance": ["mean", "min", "std", _arg_min]
        }).reset_index()
        s = reduce_dtype(s)
        columns = ["nfl_player_id"] + [f"{f}{postfix}" for f in feature_cols]
        s.columns = ["game_play", "step"] + columns

        df = pd.merge(
            df,
            s.rename(columns={c: f"{c}_1" for c in columns}),
            on=["game_play", "step", "nfl_player_id_1"],
            how="left"
        )
        df = pd.merge(
            df,
            s.rename(columns={c: f"{c}_2" for c in columns}),
            on=["game_play", "step", "nfl_player_id_2"],
            how="left"
        )
        return df

    def _merge_stacked_df_pairwise(df, s):
        s = s.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"]).agg(
            {"distance": ["mean", "min", "std", _arg_min]}).reset_index()
        s = reduce_dtype(s)
        columns = [f"{f}_pair" for f in feature_cols]
        s.columns = ["game_play", "nfl_player_id_1",
                     "nfl_player_id_2"] + columns
        df = pd.merge(
            df,
            s,
            on=["game_play", "nfl_player_id_1", "nfl_player_id_2"],
            how="left"
        )
        return df

    df = _merge_stacked_df(df, stacked, "")
    df = _merge_stacked_df(
        df, stacked[stacked["different_team"]], "_different_team")
    df = _merge_stacked_df_pairwise(df, stacked)
    df["step_diff_to_min_distance"] = df["step"] - \
        df["idxmin_distance_aronud_player_pair"]
    return df


def add_step_feature(df, tracking):
    """stepの割合など
    １プレーの長さや、その中で前半後半のどの辺のステップなのかに意味がある
    （例えば、プレー開始直後に地面とコンタクトする可能性は低い）
    """

    # 予測対象フレーム
    df["step_max_1"] = df.groupby("game_play")["step"].transform("max")
    df["step_ratio_1"] = (df["step"] / df["step_max_1"]).astype(np.float32)

    # 全体
    #step_agg = tracking.groupby("game_play")["step"].agg(["min", "max"])
    #step_agg.columns = ["step_min_2", "step_max_2"]
    #df = pd.merge(df, step_agg, left_on="game_play", right_index=True, how="left")
    #df["step_ratio_2"] = df["step"] / df["step_max_2"]
    return df


def add_aspect_ratio_feature(df, drop_original=False):
    for postfix in ["_1", "_2"]:
        for view in ["Sideline", "Endzone"]:
            df[f"aspect_{view}{postfix}"] = df[f"height_{view}{postfix}"] / \
                df[f"width_{view}{postfix}"]

            if drop_original:
                del df[f"height_{view}{postfix}"]
                del df[f"width_{view}{postfix}"]
    return df


def add_misc_features_after_agg(df):
    """集約した特徴量とJoinした特徴量の比率や差など、他の特徴量を素材に作る特徴量"""
    df["distance_from_mean_1"] = distance(
        df["x_position_1"],
        df["y_position_1"],
        df["x_position_mean"],
        df["y_position_mean"],
    )
    df["distance_from_mean_2"] = distance(
        df["x_position_2"],
        df["y_position_2"],
        df["x_position_mean"],
        df["y_position_mean"],
    )
    df["distance_team2team"] = distance(
        df["x_position_team_mean_1"],
        df["y_position_team_mean_1"],
        df["x_position_team_mean_2"],
        df["y_position_team_mean_2"],
    )
    df["speed_diff_1_2"] = df["speed_1"] - df["speed_2"]
    df["speed_diff_1_team"] = df["speed_1"] - df["speed_team_mean_1"]
    df["speed_diff_2_team"] = df["speed_2"] - df["speed_team_mean_2"]
    df["distance_mean_in_play"] = df.groupby(
        "game_play")["distance"].transform("mean")
    #df["distance_std_in_play"] = df.groupby("game_play")["distance"].transform("std")
    #df["distance_team2team_mean_in_play"] = df.groupby("game_play")["distance_team2team"].transform("mean")

    # player pairが一番近づいた瞬間の距離と現在の距離の比
    df["distance_ratio_distance_to_min_pair_distance"] = df["distance"] / \
        df["min_distance_around_player_pair"]

    # player pairの距離と、現在一番player1の近くにいるplayerとの距離比
    df["distance_ratio_distance_to_min_distance_around_player_1"] = df["distance"] / \
        df["min_distance_around_player_1"]
    df["distance_ratio_distance_to_min_distance_around_player_2"] = df["distance"] / \
        df["min_distance_around_player_2"]
    df["distance_ratio_distance_to_min_distance_around_player_diffteam_1"] = df["distance"] / \
        df["min_distance_around_player_different_team_1"]
    df["distance_ratio_distance_to_min_distance_around_player_diffteam_2"] = df["distance"] / \
        df["min_distance_around_player_different_team_2"]

    # df["distance_ratio_distance_1"] = df["distance_1"] / df["distance"]
    # df["distance_ratio_distance_2"] = df["distance_2"] / df["distance"]

    # 進行方向以外の加速度成分
    # df["sa_ratio_1"] = np.abs(df["sa_1"] / df["acceleration_1"])
    # df["sa_ratio_2"] = np.abs(df["sa_2"] / df["acceleration_2"])
    return df


def add_t0_feature(df, tracking):
    # step=0時点での統計量を使った特徴量
    on_play_start = tracking[tracking["step"] == 0].reset_index(drop=True)
    on_play_start.rename(
        columns={
            "x_position": "x_position_start",
            "y_position": "y_position_start"},
        inplace=True)

    # step=0時点でのx位置は？
    mean_x_on_play = on_play_start.groupby(
        "game_play")["x_position_start"].mean()
    mean_x_on_play.name = "x_position_mean_on_start"

    feature_cols = ["nfl_player_id", "x_position_start", "y_position_start"]
    df = pd.merge(
        df,
        on_play_start[[
            "game_play"] + feature_cols].rename(columns={c: f"{c}_1" for c in feature_cols}),
        on=["game_play", "nfl_player_id_1"],
        how="left"
    )
    df = pd.merge(
        df,
        on_play_start[[
            "game_play"] + feature_cols].rename(columns={c: f"{c}_2" for c in feature_cols}),
        on=["game_play", "nfl_player_id_2"],
        how="left"
    )
    df = pd.merge(
        df,
        mean_x_on_play.reset_index(),
        on="game_play",
        how="left"
    )
    # step=0時点からどれくら移動しているか？
    df["distance_from_start_1"] = distance(
        df["x_position_start_1"],
        df["y_position_start_1"],
        df["x_position_1"],
        df["y_position_1"]
    )
    df["distance_from_start_2"] = distance(
        df["x_position_start_2"],
        df["y_position_start_2"],
        df["x_position_2"],
        df["y_position_2"]
    )
    return df


def add_shift_of_player(df, tracking, shifts, add_diff=False, player_id="1"):
    step_orig = tracking["step"].copy()
    feature_cols = [
        "x_position",
        "y_position",
        "speed",
        "orientation",
        "direction",
        "acceleration",
        "distance",
        "sa"]

    for shift in shifts:
        tracking["step"] = step_orig - shift
        f_or_p = "future" if shift > 0 else "past"
        abs_shift = np.abs(shift)

        renames = {
            c: f"{c}_{f_or_p}{abs_shift}_{player_id}" for c in feature_cols}
        renames["nfl_player_id"] = f"nfl_player_id_{player_id}"

        df = pd.merge(
            df,
            tracking[["step", "game_play", "nfl_player_id"] +
                     feature_cols].rename(columns=renames),
            on=["step", "game_play", f"nfl_player_id_{player_id}"],
            how="left"
        )
        if add_diff:
            for c in feature_cols:
                if c in ["orientation", "direction"]:
                    df[f"{c}_diff_{shift}_{player_id}"] = angle_diff(
                        df[f"{c}_{player_id}"], df[renames[c]])
                else:
                    df[f"{c}_diff_{shift}_{player_id}"] = df[f"{c}_{player_id}"] - \
                        df[renames[c]]

    tracking["step"] = step_orig

    return df


def tracking_prep(tracking):
    for c in [
        "direction",
        "orientation",
        "acceleration",
        "sa",
        "speed",
            "distance"]:
        tracking[f"{c}_p1"] = tracking.groupby(
            ["nfl_player_id", "game_play"])[c].shift(-1)
        tracking[f"{c}_m1"] = tracking.groupby(
            ["nfl_player_id", "game_play"])[c].shift(1)
        tracking[f"{c}_p1_diff"] = tracking[c] - tracking[f"{c}_p1"]
        tracking[f"{c}_m1_diff"] = tracking[f"{c}_m1"] - tracking[c]

        if c in ["direction", "orientation"]:
            tracking[f"{c}_p1_diff"] = np.abs(
                (tracking[f"{c}_p1_diff"] + 180) % 360 - 180)
            tracking[f"{c}_m1_diff"] = np.abs(
                (tracking[f"{c}_m1_diff"] + 180) % 360 - 180)

    return tracking


def select_close_example(df):
    # 間引く
    close_sample_index = np.logical_or(
        df["distance"] <= 3,
        df["nfl_player_id_2"] == -1).values
    df = df[close_sample_index]
    return df, close_sample_index


def add_interceptor_feature(df):
    # 1-2の間に別のプレイヤー(3)がいるかどうかを計算する。以下のどちらかに該当するケースを抽出。
    # - 角3-1-2が60度以下で、距離1-2より距離1-3のほうが短い場合
    # - 角3-2-1が60度以下で、距離2-1より距離2-3のほうが短い場合
    # 複数が該当する場合は角度が小さいものを優先する
    dy = df["y_position_2"] - df["y_position_1"]
    dx = df["x_position_2"] - df["x_position_1"]

    angle_th = 60

    df["angle_dxdy"] = np.rad2deg(np.arctan2(dy, dx))

    angles = df[["game_play", "step", "nfl_player_id_1", "nfl_player_id_2", "angle_dxdy", "distance"]].copy()
    angles = angles[angles["nfl_player_id_2"] != -1]

    def negate_angle(s):
        s = s + 180
        s.loc[s > 180] = s.loc[s > 180] - 360
        return s

    angles_ = angles.copy()
    angles_.columns = ["game_play", "step", "nfl_player_id_2", "nfl_player_id_1", "angle_dxdy", "distance"]
    angles_["angle_dxdy"] = negate_angle(angles_["angle_dxdy"])

    angles = pd.concat([angles, angles_[angles.columns]]).reset_index(drop=True)

    angles_triplet = pd.merge(
        angles, angles, left_on=[
            "game_play", "step", "nfl_player_id_2"], right_on=[
            "game_play", "step", "nfl_player_id_1"], how="left")

    del angles_triplet["nfl_player_id_1_y"]
    angles_triplet.columns = [
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "angle_2to1",
        "distance_2to1",
        "nfl_player_id_3",
        "angle_2to3",
        "distance_2to3"]
    angles_triplet["angle_2to1"] = negate_angle(angles_triplet["angle_2to1"])
    angles_triplet = angles_triplet[angles_triplet["nfl_player_id_1"] != angles_triplet["nfl_player_id_3"]]

    angles_triplet["angle_123"] = angle_diff(angles_triplet["angle_2to1"], angles_triplet["angle_2to3"])

    interceptors = angles_triplet[(angles_triplet["distance_2to3"] <= angles_triplet["distance_2to1"]) & (angles_triplet["angle_123"] <= angle_th)].sort_values(
        by="angle_123").drop_duplicates(subset=["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"])

    interceptor_player2 = interceptors[["game_play", "step", "nfl_player_id_1",
                                        "nfl_player_id_2", "distance_2to3", "angle_123", "nfl_player_id_3"]].copy()
    interceptor_player2.columns = [
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "distance_of_interceptor_2",
        "angle_interceptor_2",
        "nfl_player_id_interceptor_2"]

    interceptor_player1 = interceptors[["game_play", "step", "nfl_player_id_1",
                                        "nfl_player_id_2", "distance_2to3", "angle_123", "nfl_player_id_3"]].copy()
    interceptor_player1.columns = [
        "game_play",
        "step",
        "nfl_player_id_2",
        "nfl_player_id_1",
        "distance_of_interceptor_1",
        "angle_interceptor_1",
        "nfl_player_id_interceptor_1"]

    df = pd.merge(df, interceptor_player1, on=["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"], how="left")
    df = pd.merge(df, interceptor_player2, on=["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"], how="left")

    return df


def add_bbox_std_overlap_feature(df):
    for view in ["Sideline", "Endzone"]:
        xc1 = df[f"bbox_center_x_{view}_1"]
        yc1 = df[f"bbox_center_y_{view}_1"]
        xc2 = df[f"bbox_center_x_{view}_2"]
        yc2 = df[f"bbox_center_y_{view}_2"]
        w1 = df[f"width_{view}_1"]
        h1 = df[f"height_{view}_1"]
        w2 = df[f"width_{view}_2"]
        h2 = df[f"height_{view}_2"]

        df[f"bbox_x_std_overlap_{view}"] = (np.minimum(xc1 + w1 / 2, xc2 + w2 / 2) - np.maximum(xc1 - w1 / 2, xc2 - w2 / 2)) / (w1 + w2)
        df[f"bbox_y_std_overlap_{view}"] = (np.minimum(yc1 + h1 / 2, yc2 + h2 / 2) - np.maximum(yc1 - h1 / 2, yc2 - h2 / 2)) / (h1 + h2)

    return df
