import os
import pandas as pd

from camaro.utils.nfl import frame_to_step, step_to_frame, expand_contact_id

VIEWS = ['Endzone', 'Sideline']


def both_image_exists(game_play, frame, image_dir):
    for view in VIEWS:
        img_path = f'{image_dir}/{game_play}_{view}/{frame:06}.jpg'
        if not os.path.exists(img_path):
            return False
    return True


def get_sample_df(df, helmet_df, image_dir):
    all_intervals = df.groupby(['game_play'])['frame'].agg(['min', 'max'])

    samples = []
    for (idx, game_play, min_frame, max_frame) in all_intervals.reset_index().itertuples():
        for frame in range(min_frame, max_frame + 1):
            samples.append((game_play, frame))
    sample_df = pd.DataFrame(samples, columns=['game_play', 'frame'])
    sample_df['has_both_images'] = sample_df.apply(lambda row: both_image_exists(row.game_play, row.frame, image_dir), axis=1)

    end_helmet_count = helmet_df.query('view == "Endzone"').groupby(['game_play', 'frame']).size().rename('end_count').reset_index()
    side_helmet_count = helmet_df.query('view == "Sideline"').groupby(['game_play', 'frame']).size().rename('side_count').reset_index()

    sample_df = sample_df.merge(end_helmet_count, how='left')
    sample_df = sample_df.merge(side_helmet_count, how='left')

    sample_df = sample_df.query('has_both_images == True & end_count > 0 & side_count > 0').reset_index(drop=True)
    sample_df['step'] = sample_df['frame'].map(frame_to_step)
    return sample_df


def get_sample_df_by_step(df, helmet_df, image_dir):
    all_intervals = df.groupby(['game_play'])['step'].agg(['min', 'max'])

    samples = []
    for (idx, game_play, min_step, max_step) in all_intervals.reset_index().itertuples():
        for step in range(min_step, max_step + 1):
            samples.append((game_play, step))
    sample_df = pd.DataFrame(samples, columns=['game_play', 'step'])
    sample_df['frame'] = sample_df['step'].map(step_to_frame)
    sample_df['has_both_images'] = sample_df.apply(lambda row: both_image_exists(row.game_play, row.frame, image_dir), axis=1)

    end_helmet_count = helmet_df.query('view == "Endzone"').groupby(['game_play', 'frame']).size().rename('end_count').reset_index()
    side_helmet_count = helmet_df.query('view == "Sideline"').groupby(['game_play', 'frame']).size().rename('side_count').reset_index()

    sample_df = sample_df.merge(end_helmet_count, how='left')
    sample_df = sample_df.merge(side_helmet_count, how='left')

    sample_df = sample_df.query('has_both_images == True & end_count > 0 & side_count > 0').reset_index(drop=True)
    sample_df['step'] = sample_df['frame'].map(frame_to_step)
    return sample_df


def merge_fold(df):
    fold_path = "../input/nfl-game-fold/game_fold.csv"
    fold_df = pd.read_csv(fold_path)

    df["game"] = df["game_play"].str[:5].astype(int)

    return df.merge(fold_df, how='left')


def main():
    train_path = '../input/nfl-player-contact-detection/train_labels.csv'
    train_helmet_path = '../input/interpolated/interpolated_train_helmets.csv'

    train_df = pd.read_csv(train_path)
    train_df = expand_contact_id(train_df)
    # train_df = merge_fold(train_df)
    train_df['frame'] = train_df['step'].map(step_to_frame)

    train_helmet_df = pd.read_csv(train_helmet_path)
    if 'game_play' not in train_helmet_df.columns:
        train_helmet_df['game_play'] = train_helmet_df['game_key'].astype(
            str) + '_' + train_helmet_df['play_id'].astype(str).str.zfill(6)

    train_sample_df = get_sample_df(train_df, train_helmet_df, image_dir='../input/train_frames/')
    train_sample_df = merge_fold(train_sample_df)
    train_sample_df.to_csv('../input/train_sample_df.csv', index=False)

    train_sample_df_by_step = get_sample_df_by_step(train_df, train_helmet_df, image_dir='../input/train_frames/')
    train_sample_df_by_step = merge_fold(train_sample_df_by_step)
    train_sample_df_by_step.to_csv('../input/train_sample_df_by_step.csv', index=False)


if __name__ == "__main__":
    main()
