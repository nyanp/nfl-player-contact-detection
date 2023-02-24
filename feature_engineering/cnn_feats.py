import gc
from typing import List
import pandas as pd
import numpy as np
import glob

from utils.general import reduce_dtype


def add_cnn_shift_diff_features(df: pd.DataFrame, shift_steps: List[int] = [-5, -3, -1, 1, 3, 5],
                                columns: List[str] = ["cnn_pred_Sideline", "cnn_pred_Endzone"]) -> pd.DataFrame:
    for col in columns:
        for shift_step in shift_steps:
            df[f'{col}_shift_{shift_step}'] = (df.sort_values('step')
                                               .groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])[col]
                                               .shift(shift_step).reset_index().sort_values('index').set_index('index'))
            df[f'{col}_diff_{shift_step}'] = df[col] - df[f'{col}_shift_{shift_step}']
    return df


def add_cnn_features(df, cnn_df_dict):
    base_feature_cols = [
        'cnn_pred_Sideline',
        'cnn_pred_Endzone',
        'camaro1_pred',
        # 'camaro1_any_pred',
        'camaro2_pred',
        # 'camaro1_any_pred',
    ]

    camaro1_df = cnn_df_dict.get('camaro1', pd.read_csv('../input/camaro-exp117/exp117_val_last_preds.csv'))
    # camaro1_any_df = cnn_df_dict.get('camaro1_any', pd.read_csv('../input/camaro-exp117/exp117_val_any_preds.csv'))
    camaro2_df = cnn_df_dict.get('camaro2', pd.read_csv('../input/nfl-exp048/val_df.csv'))
    # camaro2_any_df = cnn_df_dict.get('camaro2_any', pd.read_csv('../input/camaro-exp117/exp117_val_any_preds.csv'))

    camaro1_df['camaro1_pred'] = np.nan
    camaro1_df['camaro1_pred'] = camaro1_df['camaro1_pred'].astype(np.float32)
    camaro1_df.loc[camaro1_df['masks'], 'camaro1_pred'] = camaro1_df.loc[camaro1_df['masks'], 'preds']
    merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'camaro1_pred']
    df = df.merge(camaro1_df[merge_cols], how='left')
    del camaro1_df
    gc.collect()

    # camaro1_any_df['camaro1_any_pred'] = np.nan
    # camaro1_any_df['camaro1_any_pred'] = camaro1_any_df['camaro1_any_pred'].astype(np.float32)
    # camaro1_any_df.loc[camaro1_any_df['masks'], 'camaro1_any_pred'] = camaro1_any_df.loc[camaro1_any_df['masks'], 'preds']
    # merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'camaro1_any_pred']
    # df = df.merge(camaro1_any_df[merge_cols], how='left')

    camaro2_df['camaro2_pred'] = np.nan
    camaro2_df['camaro2_pred'] = camaro2_df['camaro2_pred'].astype(np.float32)
    camaro2_df.loc[camaro2_df['masks'], 'camaro2_pred'] = camaro2_df.loc[camaro2_df['masks'], 'preds']
    merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'camaro2_pred']
    df = df.merge(camaro2_df[merge_cols], how='left')
    del camaro2_df
    gc.collect()
    # camaro2_any_df['camaro2_any_pred'] = np.nan
    # camaro2_any_df['camaro2_any_pred'] = camaro2_any_df['camaro2_any_pred'].astype(np.float32)
    # camaro2_any_df.loc[camaro2_any_df['masks'], 'camaro2_any_pred'] = camaro2_any_df.loc[camaro2_any_df['masks'], 'preds']
    # merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'camaro2_any_pred']
    # df = df.merge(camaro2_any_df[merge_cols], how='left')

    kmat_end_df = cnn_df_dict.get('kmat_end', None)
    kmat_side_df = cnn_df_dict.get('kmat_side', None)
    kmat_end_map_df = cnn_df_dict.get('kmat_end_map', None)
    kmat_side_map_df = cnn_df_dict.get('kmat_side_map', None)

    if kmat_end_df is None:
        end_paths = sorted(glob.glob('../input/mfl2cnnkmat0219/output/fold*_cnn_pred_end.csv'))
        kmat_end_df = pd.concat([pd.read_csv(p) for p in end_paths]).reset_index(drop=True)

    if kmat_side_df is None:
        side_paths = sorted(glob.glob('../input/mfl2cnnkmat0219/output/fold*_cnn_pred_side.csv'))
        kmat_side_df = pd.concat([pd.read_csv(p) for p in side_paths]).reset_index(drop=True)

    if kmat_end_map_df is None:
        end_paths = sorted(glob.glob('../input/mfl2cnnkmat0219/output/fold*_map_pred_end.csv'))
        kmat_end_map_df = pd.concat([pd.read_csv(p) for p in end_paths]).reset_index(drop=True)

    if kmat_side_map_df is None:
        side_paths = sorted(glob.glob('../input/mfl2cnnkmat0219/output/fold*_map_pred_side.csv'))
        kmat_side_map_df = pd.concat([pd.read_csv(p) for p in side_paths]).reset_index(drop=True)

    kmat_end_df['step'] = kmat_end_df['step'].astype(int)
    kmat_end_df.loc[kmat_end_df['nfl_player_id_2'] == 0, 'nfl_player_id_2'] = -1
    kmat_end_df['cnn_pred_Endzone'] = kmat_end_df['cnn_pred_Endzone'].astype(np.float32)

    kmat_side_df['step'] = kmat_side_df['step'].astype(int)
    kmat_side_df.loc[kmat_side_df['nfl_player_id_2'] == 0, 'nfl_player_id_2'] = -1
    kmat_side_df['cnn_pred_Sideline'] = kmat_side_df['cnn_pred_Sideline'].astype(np.float32)

    merge_cols = ['step', 'game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'cnn_pred_Endzone']
    df = df.merge(kmat_end_df[merge_cols], how='left')

    merge_cols = ['step', 'game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'cnn_pred_Sideline']
    df = df.merge(kmat_side_df[merge_cols], how='left')
    del kmat_end_df, kmat_side_df
    gc.collect()

    kmat_end_map_df['step'] = kmat_end_map_df['step'].astype(int)
    kmat_side_map_df['step'] = kmat_side_map_df['step'].astype(int)

    # それ以外のシングルプレイヤー系の予測値。座標予測と、単独でのコンタクト予測
    new_columns = []
    for pid in [1, 2]:
        df = pd.merge(df,
                      kmat_end_map_df
                      [["game_play", "step", "nfl_player_id", "pred_coords_i_Endzone",
                        "pred_coords_j_Endzone", "player_single_contacts_Endzone"]],
                      left_on=["game_play", "step", f"nfl_player_id_{pid}"],
                      right_on=["game_play", "step", "nfl_player_id"], how="left")
        df.drop(columns=["nfl_player_id"], inpace=True)
        df.rename(columns={"pred_coords_i_Endzone": f"pred_coords_i_Endzone_pid{pid}",
                           "pred_coords_j_Endzone": f"pred_coords_j_Endzone_pid{pid}",
                           "player_single_contacts_Endzone": f"player_single_contacts_Endzone_pid{pid}"}, inpace=True)

        df = pd.merge(df,
                      kmat_side_map_df
                      [["game_play", "step", "nfl_player_id", "pred_coords_i_Sideline",
                        "pred_coords_j_Sideline", "player_single_contacts_Sideline"]],
                      left_on=["game_play", "step", f"nfl_player_id_{pid}"],
                      right_on=["game_play", "step", "nfl_player_id"], how="left")
        df.drop(columns=["nfl_player_id"], inpace=True)
        df.rename(columns={"pred_coords_i_Sideline": f"pred_coords_i_Sideline_pid{pid}",
                           "pred_coords_j_Sideline": f"pred_coords_j_Sideline_pid{pid}",
                           "player_single_contacts_Sideline": f"player_single_contacts_Sideline_pid{pid}"}, inpace=True)
        new_columns += [f"pred_coords_i_Endzone_pid{pid}", f"pred_coords_j_Endzone_pid{pid}", f"player_single_contacts_Endzone_pid{pid}",
                        f"pred_coords_i_Sideline_pid{pid}", f"pred_coords_j_Sideline_pid{pid}", f"player_single_contacts_Sideline_pid{pid}"]
    # add some features
    for view in ["Endzone", "Sideline"]:
        df[f"pred_coords_delta_i_{view}"] = np.abs(
            df[f"pred_coords_i_{view}_pid1"] - df[f"pred_coords_i_{view}_pid2"])
        df[f"pred_coords_delta_j_{view}"] = np.abs(
            df[f"pred_coords_j_{view}_pid1"] - df[f"pred_coords_j_{view}_pid2"])
        df[f"pred_coords_dist_{view}"] = np.sqrt(
            df[f"pred_coords_delta_i_{view}"]**2 + df[f"pred_coords_delta_j_{view}"]**2)
        df[f"player_single_contacts_multiply_{view}"] = df[f"player_single_contacts_{view}_pid1"] * \
            df[f"player_single_contacts_{view}_pid2"]
        df[f"pred_coords_distratio_{view}"] = df[f"pred_coords_dist_{view}"] / df["distance"]
        df[f"player_single_pair_relative_contacts_{view}_pid1"] = df[f"cnn_pred_{view}"] / \
            df[f"player_single_contacts_{view}_pid1"]
        df[f"player_single_pair_relative_contacts_{view}_pid2"] = df[f"cnn_pred_{view}"] / \
            df[f"player_single_contacts_{view}_pid2"]
        new_columns += [f"pred_coords_delta_i_{view}", f"pred_coords_delta_j_{view}",
                        f"pred_coords_dist_{view}", f"player_single_contacts_multiply_{view}",
                        f"pred_coords_distratio_{view}",
                        f"player_single_pair_relative_contacts_{view}_pid1", f"player_single_pair_relative_contacts_{view}_pid2"]

    del kmat_end_map_df, kmat_side_map_df
    gc.collect()

    return reduce_dtype(df), base_feature_cols


def add_cnn_agg_features(df, base_feature_cols):
    def g_con_around_feature(df_train, dist_thresh=1.5, columns=['cnn_pred_Sideline_roll11', 'cnn_pred_Endzone_roll11']):
        """
        周辺が倒れている場合、その人も倒れている。
        画像中の隠れているプレイヤに対するグラウンドコンタクト紐づけ　兼　アサインメントミスの補助
        """
        print(df_train.shape)
        add_columns = [n + "_g_contact_around" for n in columns]
        df_train_g = df_train.loc[df_train["nfl_player_id_2"] == -
                                  1, ["game_play", "step", "nfl_player_id_1"] +
                                  columns].rename(columns={c: a for c, a in zip(columns, add_columns)})
        p_pairs = df_train.loc[df_train["nfl_player_id_2"] != -1, ["game_play", "step", "nfl_player_id_1", "nfl_player_id_2", "distance"]]
        p_pairs = p_pairs[p_pairs["distance"] < dist_thresh]
        p_pairs = pd.concat([p_pairs, p_pairs.rename(columns={"nfl_player_id_1": "nfl_player_id_2", "nfl_player_id_2": "nfl_player_id_1"})], axis=0)
        p_pairs = pd.merge(p_pairs,
                           df_train_g,
                           on=["game_play", "step", "nfl_player_id_1"], how="left")
        # 周辺プレイヤの地面コンタクト(自分以外)をとりあえずsumる。
        p_pairs = p_pairs.groupby(["game_play", "step", "nfl_player_id_2"])[add_columns].sum().reset_index()  # ("sum", "mean")
        p_pairs = p_pairs.rename(columns={"nfl_player_id_2": "nfl_player_id_1"})

        df_train = pd.merge(df_train,
                            p_pairs,
                            on=["game_play", "step", "nfl_player_id_1"], how="left")

        # df_train.loc[df_train["nfl_player_id_2"]!=-1, add_columns] = np.nan

        return df_train

    def g_conact_as_condition(df_train, score_columns=['cnn_pred_Sideline_roll5', 'cnn_pred_Endzone_roll5'], dist_ratio=1.):
        """
        同じ状態(高さ姿勢)のものはコンタクトしやすい
        グラウンドコンタクトの発生状態を高さ姿勢と考えてペア間の高さ姿勢を比較する
        あわせて、疑似的な距離を算出する。
        """
        print(df_train.shape)
        temp_columns = [n + "_as_g_contact_cond" for n in score_columns]
        dev_columns = [n + "_dev_as_g_contact_cond" for n in score_columns]
        dist_columns = [n + "_dist_as_g_contact_cond" for n in score_columns]

        df_train_g = df_train.loc[df_train["nfl_player_id_2"] == -1, ["game_play", "step", "nfl_player_id_1"] +
                                  score_columns].rename(columns={c: a for c, a in zip(score_columns, temp_columns)})

        df_train = pd.merge(df_train,
                            df_train_g,
                            on=["game_play", "step", "nfl_player_id_1"], how="left").rename(columns={c: c + "_1" for c in temp_columns})
        df_train = pd.merge(df_train,
                            df_train_g.rename(columns={"nfl_player_id_1": "nfl_player_id_2"}),
                            on=["game_play", "step", "nfl_player_id_2"], how="left").rename(columns={c: c + "_2" for c in temp_columns})

        for dist_c, dev_c, temp_c in zip(dist_columns, dev_columns, temp_columns):
            df_train[dev_c] = np.abs(df_train[temp_c + "_2"] - df_train[temp_c + "_1"])
            df_train[dist_c] = np.sqrt((df_train[dev_c] * dist_ratio)**2 + df_train["distance"]**2)

        df_train = df_train.drop(columns=[c + "_1" for c in temp_columns] + [c + "_2" for c in temp_columns])

        return df_train

    def p_con_shift_feature(df_train, step_offset=5, score_columns=['cnn_pred_Sideline_roll11', 'cnn_pred_Endzone_roll11']):
        """
        過去にコンタクトがあった人は再度何かしらのコンタクト(特に地面と?)することが多い。
        (逆に地面コンタクトしている人はさかのぼると誰かにコンタクトしていると思う。TODO?)
        """
        print(df_train.shape)
        add_columns = [n + f"_p_contact_past{step_offset}" for n in score_columns]
        df_train_p = df_train.loc[df_train["nfl_player_id_2"] != -
                                  1, ["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"] +
                                  score_columns].rename(columns={c: a for c, a in zip(score_columns, add_columns)})
        df_train_p = pd.concat([df_train_p, df_train_p.rename(
            columns={"nfl_player_id_1": "nfl_player_id_2", "nfl_player_id_2": "nfl_player_id_1"})], axis=0)
        df_train_p = df_train_p.groupby(["game_play", "step", "nfl_player_id_1"])[add_columns].sum().reset_index()
        df_train_p["step"] = df_train_p["step"] + step_offset

        df_train = pd.merge(df_train,
                            df_train_p,
                            on=["game_play", "step", "nfl_player_id_1"], how="left").rename(columns={c: c + "_1" for c in add_columns})
        df_train = pd.merge(df_train,
                            df_train_p.rename(columns={"nfl_player_id_1": "nfl_player_id_2"}),
                            on=["game_play", "step", "nfl_player_id_2"], how="left").rename(columns={c: c + "_2" for c in add_columns})

        return df_train

    df = g_con_around_feature(
        df,
        dist_thresh=1.5,
        columns=[f'{col}_roll11' for col in base_feature_cols],
    )
    df = g_con_around_feature(
        df,
        dist_thresh=0.75,
        columns=[f'{col}_roll5' for col in base_feature_cols],
    )
    df = p_con_shift_feature(
        df,
        step_offset=5,
        score_columns=[f'{col}_roll5' for col in base_feature_cols],
    )
    df = p_con_shift_feature(
        df,
        step_offset=-5,
        score_columns=[f'{col}_roll5' for col in base_feature_cols]
    )
    # df = g_conact_as_condition(df, score_columns = ['cnn_pred_Sideline_roll11', 'cnn_pred_Endzone_roll11'])

    return reduce_dtype(df)


def agg_cnn_feature(df: pd.DataFrame, columns) -> pd.DataFrame:
    for column in columns:
        for agg in ['max', 'min', 'std']:
            df[f"{column}_{agg}_pair"] = df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])[column].transform(agg)
            df[f"{column}_{agg}_step"] = df.groupby(['game_play', 'step'])[column].transform(agg)
    return reduce_dtype(df)
