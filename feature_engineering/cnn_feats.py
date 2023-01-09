import pandas as pd
import numpy as np
import glob


def add_cnn_features(df, camaro_df=None, kmat_end_df=None, kmat_side_df=None):
    if camaro_df is None:
        camaro_df = pd.read_csv('../pipeline/output/exp048_both_ext_blur_dynamic_normalize_coords_fix_frame_noise/oof_val_preds_agg_df.csv')

    camaro_df['masks'] = camaro_df['masks'].fillna(False)
    camaro_df['camaro_pred'] = np.nan  # np.nanじゃないとroll feature作れなかった
    camaro_df.loc[camaro_df['masks'], 'camaro_pred'] = camaro_df.loc[camaro_df['masks'], 'preds']

    agg_cols = ['preds_max', 'preds_min', 'preds_std', 'preds_mean', 'preds_count']
    merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'camaro_pred'] + agg_cols
    df = df.merge(camaro_df[merge_cols], how='left')

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
