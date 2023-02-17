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


def add_cnn_features(df, camaro_df=None, kmat_end_df=None, kmat_side_df=None, camaro_any_df=None, camaro_df3=None, camaro_df4=None):
    base_feature_cols = [
        'cnn_pred_Sideline',
        'cnn_pred_Endzone',
        'camaro_pred',
        'camaro_any_pred',
        # 'camaro_pred3',
        # 'camaro_pred4',
    ]

    if camaro_df is None:
        # camaro_df = pd.read_csv('../input/nfl-exp048/val_df.csv')
        camaro_df = pd.read_csv('../input/camaro-exp128/exp128_val_preds.csv')
    camaro_df['camaro_pred'] = np.nan  # np.nanじゃないとroll feature作れなかった
    camaro_df['camaro_pred'] = camaro_df['camaro_pred'].astype(np.float32)
    camaro_df.loc[camaro_df['masks'], 'camaro_pred'] = camaro_df.loc[camaro_df['masks'], 'preds']
    merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'camaro_pred']
    df = df.merge(camaro_df[merge_cols], how='left')

    if camaro_any_df is None:
        camaro_any_df = pd.read_csv('../input/camaro-exp128/exp128_val_preds.csv')
    camaro_any_df['camaro_any_pred'] = np.nan  # np.nanじゃないとroll feature作れなかった
    camaro_any_df['camaro_any_pred'] = camaro_any_df['camaro_any_pred'].astype(np.float32)
    camaro_any_df.loc[camaro_any_df['masks'], 'camaro_any_pred'] = camaro_any_df.loc[camaro_any_df['masks'], 'preds']
    merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'camaro_any_pred']
    df = df.merge(camaro_any_df[merge_cols], how='left')

    # if camaro_df3 is None:
    #     camaro_df3 = pd.read_csv('../input/camaro-exp125/exp125_val_any_preds.csv')
    # camaro_df3 = camaro_df3.rename(columns={'preds': 'camaro_pred3'})
    # camaro_df3['camaro_pred3'] = camaro_df3['camaro_pred3'].astype(np.float32)
    # merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'camaro_pred3']
    # df = df.merge(camaro_df3[merge_cols], how='left')

    # if camaro_df4 is None:
    #     camaro_df4 = pd.read_csv('../input/camaro-exp123-exp124/exp123_exp124_val_preds.csv')
    # camaro_df4 = camaro_df4.rename(columns={'preds': 'camaro_pred4'})
    # camaro_df4['camaro_pred4'] = camaro_df4['camaro_pred4'].astype(np.float32)
    # merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'camaro_pred4']
    # df = df.merge(camaro_df4[merge_cols], how='left')

    if kmat_end_df is None:
        end_paths = sorted(glob.glob('../input/mfl2cnnkmat0121/output/fold*_cnn_pred_end.csv'))
        kmat_end_df = pd.concat([pd.read_csv(p) for p in end_paths]).reset_index(drop=True)

    if kmat_side_df is None:
        side_paths = sorted(glob.glob('../input/mfl2cnnkmat0121/output/fold*_cnn_pred_side.csv'))
        kmat_side_df = pd.concat([pd.read_csv(p) for p in side_paths]).reset_index(drop=True)

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
