import pickle
import numpy as np
import pandas as pd
from datasets.common import setup_df

VIEWS = ['Endzone', 'Sideline']


def load_track_df(track_path):
    tr_tracking = pd.read_csv(track_path, parse_dates=["datetime"])
    tr_tracking['x_position'] /= 120  # 最初にやってしまったけどxとyのスケールが変わってよくなかった
    tr_tracking['y_position'] /= 53.3  # 最初にやってしまったけどxとyのスケールが変わってよくなかった
    tr_tracking['sin_direction'] = tr_tracking['direction'].map(lambda x: np.sin(x * np.pi / 180))
    tr_tracking['cos_direction'] = tr_tracking['direction'].map(lambda x: np.cos(x * np.pi / 180))
    tr_tracking['sin_orientation'] = tr_tracking['orientation'].map(lambda x: np.sin(x * np.pi / 180))
    tr_tracking['cos_orientation'] = tr_tracking['orientation'].map(lambda x: np.cos(x * np.pi / 180))
    tr_tracking = tr_tracking[['game_play', 'nfl_player_id', 'step', 'x_position', 'y_position', 'speed',
                               'distance', 'acceleration', 'sa', 'sin_direction', 'cos_direction', 'sin_orientation', 'cos_orientation']]

    label_df_path = '../input/train_labels_with_folds.csv'
    labels = pd.read_csv(label_df_path)

    g_df = labels.query('contact == 1 & nfl_player_id_2 == 0')
    g_df = g_df.rename(columns={'nfl_player_id_1': 'nfl_player_id'}).merge(tr_tracking, how='left')
    return g_df


def get_track_dict(track_df):
    gb_cols = ['game_play', 'frame']
    track_cols = ['x_position', 'y_position', 'speed', 'distance',
                  'acceleration', 'sa', 'sin_direction', 'cos_direction',
                  'sin_orientation', 'cos_orientation', 'nfl_player_id']

    track_cols
    gb = track_df.groupby(gb_cols)
    track_dict = {k: v[track_cols].values for k, v in gb}
    return track_dict


def get_track_dict_by_step(track_df):
    gb_cols = ['game_play', 'step']
    track_cols = ['x_position', 'y_position', 'speed', 'distance',
                  'acceleration', 'sa', 'sin_direction', 'cos_direction',
                  'sin_orientation', 'cos_orientation', 'nfl_player_id']

    track_cols
    gb = track_df.groupby(gb_cols)
    track_dict = {k: v[track_cols].values for k, v in gb}
    return track_dict


def get_unique_ids_dict(label_df):
    unique_ids_dict = label_df.groupby(['game_play', 'frame'])['nfl_player_id_1'].unique().to_dict()

    for k, v in unique_ids_dict.items():
        v = sorted(v)
        if len(v) < 22:
            v = v + [-1] * (22 - len(v))  # padding
        unique_ids_dict[k] = v
    return unique_ids_dict


def get_inter_contact_dict(label_df):
    gb = label_df.query('contact == 1 & nfl_player_id_2 > 0').groupby(['game_play', 'frame'])[['nfl_player_id_1', 'nfl_player_id_2']]
    contact_dict = {k: v[['nfl_player_id_1', 'nfl_player_id_2']].values for k, v in gb}
    return contact_dict


def get_ground_contact_dict(label_df):
    gb = label_df.query('contact == 1 & nfl_player_id_2 == 0').groupby(['game_play', 'frame'])[['nfl_player_id_1', 'nfl_player_id_2']]
    contact_dict = {k: v['nfl_player_id_1'].values for k, v in gb}
    return contact_dict


def get_helmet_dict(helmet_df):
    cols = ['left', 'top', 'width', 'height', 'nfl_player_id']
    gb = helmet_df.groupby(['game_play', 'frame', 'view'])
    helmet_dict = {k: v[cols].values for k, v in gb}
    return helmet_dict


df_path = '../input/folds.csv'
label_df_path = '../input/train_labels_with_folds.csv'
helmet_df_path = '../input/train_helmets_with_folds.csv'
interpolated_helmet_df_path = '../input/interpolated_train_helmets_with_folds.csv'
tracking_df_path = '../input/tracking_with_folds.csv'
image_dir = '../input/train_frames_half'


for fold in range(4):
    for mode in ['train', 'valid']:
        print('start', mode, fold)
        df = setup_df(df_path, fold, mode)
        label_df = setup_df(label_df_path, fold, mode)
        helmet_df = setup_df(helmet_df_path, fold, mode)
        interpolated_helmet_df = setup_df(interpolated_helmet_df_path, fold, mode)
        tracking_df = setup_df(tracking_df_path, fold, mode)

        unique_ids_dict = get_unique_ids_dict(label_df)
        inter_contact_dict = get_inter_contact_dict(label_df)
        ground_contact_dict = get_ground_contact_dict(label_df)
        helmet_dict = get_helmet_dict(helmet_df)
        interpolated_helmet_dict = get_helmet_dict(interpolated_helmet_df)
        track_dict = get_track_dict(tracking_df)
        track_dict_by_step = get_track_dict_by_step(tracking_df)

        with open(f'../input/unique_ids_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(unique_ids_dict, f)
        with open(f'../input/inter_contact_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(inter_contact_dict, f)
        with open(f'../input/ground_contact_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(ground_contact_dict, f)
        with open(f'../input/helmet_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(helmet_dict, f)
        with open(f'../input/interpolated_helmet_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(interpolated_helmet_dict, f)
        with open(f'../input/track_dict_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(track_dict, f)
        with open(f'../input/track_dict_by_step_{mode}_fold{fold}.pkl', 'wb') as f:
            pickle.dump(track_dict_by_step, f)
