import pandas as pd
import glob


def add_cnn_features(df):
    exp028_df = pd.read_csv(
        '../pipeline/output/exp028_g_only_simple_org_size_image_aug_v3_frame_noise/val_df.csv')
    exp028_df = exp028_df.rename(columns={'preds': 'exp028_ground'})

    end_paths = sorted(glob.glob('../input/mfl2cnnkmat1225/output/fold*_cnn_pred_end.csv'))
    end_preds = pd.concat([pd.read_csv(p) for p in end_paths]).reset_index(drop=True)
    end_preds['step'] = end_preds['step'].astype(int)
    end_preds.loc[end_preds['nfl_player_id_2'] == 0, 'nfl_player_id_2'] = -1

    side_paths = sorted(glob.glob('../input/mfl2cnnkmat1225/output/fold*_cnn_pred_side.csv'))
    side_preds = pd.concat([pd.read_csv(p) for p in side_paths]).reset_index(drop=True)
    side_preds['step'] = side_preds['step'].astype(int)
    side_preds.loc[side_preds['nfl_player_id_2'] == 0, 'nfl_player_id_2'] = -1

    merge_cols = ['game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'exp028_ground']
    df = df.merge(exp028_df[merge_cols], how='left')

    merge_cols = ['step', 'game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'cnn_pred_Endzone']
    df = df.merge(end_preds[merge_cols], how='left')

    merge_cols = ['step', 'game_play', 'nfl_player_id_1', 'nfl_player_id_2', 'cnn_pred_Sideline']
    df = df.merge(side_preds[merge_cols], how='left')

    return df
