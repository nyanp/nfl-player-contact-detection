from typing import List
import pandas as pd
import numpy as np
import glob


def add_cnn_shift_diff_feature(df: pd.DataFrame, shift_steps: List[int] = [-5, -3, -1, 1, 3, 5],
                               columns: List[str] = ["cnn_pred_Sideline", "cnn_pred_Endzone"]) -> pd.DataFrame:
    for col in columns:
        for shift_step in shift_steps:
            df[f'{col}_shift_{shift_step}'] = (df.sort_values('step')
                                               .groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2'])[col]
                                               .shift(shift_step).reset_index().sort_values('index').set_index('index'))
            df[f'{col}_diff_{shift_step}'] = df[col] - df[f'{col}_shift_{shift_step}']
    return df


def add_cnn_features(df, camaro_df=None, kmat_end_df=None, kmat_side_df=None):
    if camaro_df is None:
        camaro_df = pd.read_csv('../pipeline/output/exp048_both_ext_blur_dynamic_normalize_coords_fix_frame_noise/val_df.csv')

    camaro_df['camaro_pred'] = np.nan  # np.nanじゃないとroll feature作れなかった
    camaro_df.loc[camaro_df['masks'], 'camaro_pred'] = camaro_df.loc[camaro_df['masks'], 'preds']
    # camaro_df = camaro_df.rename(columns={'preds': 'camaro_pred'})

    merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'camaro_pred']
    df = df.merge(camaro_df[merge_cols], how='left')

    camaro_agg_df = pd.read_csv('../pipeline/output/exp048_both_ext_blur_dynamic_normalize_coords_fix_frame_noise/oof_val_preds_agg_df.csv')
    merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'preds_max', 'preds_min', 'preds_std', 'preds_mean', 'preds_count']
    df = df.merge(camaro_agg_df[merge_cols], how='left')

    if kmat_end_df is None:
        end_paths = sorted(glob.glob('../input/mfl2cnnkmat1225/output/fold*_cnn_pred_end.csv'))
        kmat_end_df = pd.concat([pd.read_csv(p) for p in end_paths]).reset_index(drop=True)

    if kmat_side_df is None:
        side_paths = sorted(glob.glob('../input/mfl2cnnkmat1225/output/fold*_cnn_pred_side.csv'))
        kmat_side_df = pd.concat([pd.read_csv(p) for p in side_paths]).reset_index(drop=True)

    kmat_end_df['step'] = kmat_end_df['step'].astype(int)
    kmat_end_df.loc[kmat_end_df['nfl_player_id_2'] == 0, 'nfl_player_id_2'] = -1

    kmat_side_df['step'] = kmat_side_df['step'].astype(int)
    kmat_side_df.loc[kmat_side_df['nfl_player_id_2'] == 0, 'nfl_player_id_2'] = -1

    merge_cols = ['step', 'game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'cnn_pred_Endzone']
    df = df.merge(kmat_end_df[merge_cols], how='left')

    merge_cols = ['step', 'game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'cnn_pred_Sideline']
    df = df.merge(kmat_side_df[merge_cols], how='left')

    return df
