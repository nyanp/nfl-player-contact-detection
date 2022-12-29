import pandas as pd


def add_camaro_features(df):
    exp028_df = pd.read_csv(
        '../pipeline/output/exp028_g_only_simple_org_size_image_aug_v3_frame_noise/val_df.csv')
    exp028_df = exp028_df.rename(columns={'preds': 'exp028_ground'})
    merge_cols = [
        'game_play',
        'step',
        'nfl_player_id_1',
        'nfl_player_id_2',
        'exp028_ground']
    df = df.merge(exp028_df[merge_cols], how='left')
    return df
