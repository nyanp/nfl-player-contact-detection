import pandas as pd


def step_to_frame(df):
    step0_frame = 300
    fps_frame = 59.95
    fps_step = 10
    step_max = 200
    convert_dict = {}
    for step in range(step_max):
        convert_dict[step] = step * fps_frame / fps_step + step0_frame
    df['frame'] = df['step'].map(convert_dict)
    df['frame'] = df['frame'].round().astype(int)
    return df


unique_data_cols = ['game_play', 'step', 'frame', 'fold']
label_cols = ['game_play', 'step', 'nfl_player_id_1',
              'nfl_player_id_2', 'contact', 'frame', 'fold']
tracking_cols = ['game_play', 'nfl_player_id', 'step',
                 'team', 'position', 'jersey_number', 'x_position', 'y_position',
                 'speed', 'distance', 'direction', 'orientation', 'acceleration', 'sa',
                 'fold']

tr_helmets_cols = ['game_play', 'view', 'frame',
                   'nfl_player_id', 'player_label', 'left', 'width', 'top', 'height',
                   'fold']

if __name__ == "__main__":
    print('load dataframes...')
    train_tracking = pd.read_csv('../input/nfl-player-contact-detection/train_player_tracking.csv')
    train_labels = pd.read_csv('../input/nfl-player-contact-detection/train_labels.csv')
    train_helmets = pd.read_csv('../input/nfl-player-contact-detection/train_baseline_helmets.csv')
    train_meta = pd.read_csv('../input/nfl-player-contact-detection/train_video_metadata.csv')
    folds = pd.read_csv("../input/nfl-game-fold/game_fold.csv")

    print('preprocess dataframes...')
    train_labels['nfl_player_id_2'] = train_labels['nfl_player_id_2'].replace('G', 0).astype(int)
    train_labels = step_to_frame(train_labels)
    train_labels['game'] = train_labels['game_play'].map(lambda x: int(x.split('_')[0]))
    unique_data = train_labels[['game', 'game_play', 'step', 'frame']].drop_duplicates().reset_index(drop=True)

    print('merge dataframes...')
    unique_data = unique_data.merge(folds)
    train_labels = train_labels.merge(folds)
    train_tracking = train_tracking.merge(folds.rename(columns={'game': 'game_key'}))
    train_helmets = train_helmets.merge(folds.rename(columns={'game': 'game_key'}))

    print('save dataframes...')
    unique_data[unique_data_cols].to_csv('../input/folds.csv', index=False)
    train_labels[label_cols].to_csv('../input/train_labels_with_folds.csv', index=False)
    train_tracking[tracking_cols].to_csv('../input/train_tracking_with_folds.csv', index=False)
    train_helmets[tr_helmets_cols].to_csv('../input/train_helmets_with_folds.csv', index=False)
