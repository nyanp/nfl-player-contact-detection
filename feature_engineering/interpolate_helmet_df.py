from multiprocessing import Pool, cpu_count

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.auto import tqdm

STRING_COLS = ['game_play', 'game_key', 'view', 'nfl_player_id', 'player_label']
SAVE_COLS = ['game_play', 'view', 'frame', 'nfl_player_id', 'player_label', 'left', 'width', 'top', 'height']

INTERPOLATE_LIMIT = 11
DIST_LIMIT = 3  # 3 times larger than average bbox size

TRAIN_HELMET_PATH = '../input/nfl-player-contact-detection/train_baseline_helmets.csv'
TEST_HELMET_PATH = '../input/nfl-player-contact-detection/test_baseline_helmets.csv'
FOLD_PATH = "../input/nfl-game-fold/game_fold.csv"
TRAIN_SAVE_PATH = './data/train_interpolated_helmets.csv'
TEST_SAVE_PATH = './data/test_interpolated_helmets.csv'


def check_has_missing(df):
    col = 'left'  # anything is ok
    min_frame = df['frame'].min()
    max_frame = df['frame'].max()
    frames = np.arange(min_frame, max_frame + 1)
    merged = pd.DataFrame({'frame': frames}).merge(df[['frame', col]], how='left')
    return merged[col].isnull().sum() > 0


def get_center(bbox):
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    return cx, cy


def bbox_dist(bbox1, bbox2):
    cx1, cy1 = get_center(bbox1)
    cx2, cy2 = get_center(bbox2)
    return ((cx2 - cx1)**2 + (cy2 - cy1)**2)**0.5


def interpolate_bboxes(prev_bbox, nxt_bbox, prev_frame, curr_frame, nxt_frame):
    x, y, w, h = (curr_frame - prev_frame) * (nxt_bbox - prev_bbox) / (nxt_frame - prev_frame) + prev_bbox
    is_interpolated = 1  # True
    curr_bbox = np.array([x, y, w, h, curr_frame, is_interpolated])
    return curr_bbox


def add_not_interpolated_flag(bbox):
    x, y, w, h, frame = bbox
    is_interpolated = 0  # False
    return np.array([x, y, w, h, frame, is_interpolated])


def interpolate_missing_frame_bboxes(bboxes):
    interpolated = []

    avg_size = bboxes[:, 2:4].max(1).mean()

    prev = None
    for i, nxt in enumerate(bboxes):
        # first frame
        if prev is None:
            prev = nxt
            interpolated.append(add_not_interpolated_flag(nxt))
            continue

        prev_frame = prev[-1]
        nxt_frame = nxt[-1]

        # continuous frame
        if nxt_frame - prev_frame == 1:
            pass

        # too far in time space
        elif nxt_frame - prev_frame > INTERPOLATE_LIMIT:
            pass

        # too far in image space
        elif bbox_dist(nxt, prev) > avg_size * DIST_LIMIT:
            pass

        else:
            prev_bbox = prev[:-1]
            nxt_bbox = nxt[:-1]
            for curr_frame in range(prev_frame + 1, nxt_frame):
                curr_bbox = interpolate_bboxes(prev_bbox, nxt_bbox, prev_frame, curr_frame, nxt_frame)
                # print('prev:', prev)
                # print('curr:', curr_bbox)
                # print('next:', nxt)
                interpolated.append(curr_bbox)

        prev = nxt
        interpolated.append(add_not_interpolated_flag(nxt))

    return np.stack(interpolated)


def interpolate_video_player_df(df):
    bboxes = df[['left', 'top', 'width', 'height', 'frame', ]].values
    interpolated_bboxes = interpolate_missing_frame_bboxes(bboxes)
    interpolated_df = pd.DataFrame(data=np.stack(interpolated_bboxes), columns=[
                                   'left', 'top', 'width', 'height', 'frame', 'is_interpolated'])
    interpolated_df['frame'] = interpolated_df['frame'].astype(int)
    interpolated_df['is_interpolated'] = interpolated_df['is_interpolated'].astype(bool)
    for col in STRING_COLS:
        interpolated_df[col] = df[col].iloc[0]
    return interpolated_df


def load_helmet_df(helmet_df_path):
    helmet_df = pd.read_csv(helmet_df_path)
    helmet_df['view'] = helmet_df['view'].str.replace('Endzone2', 'Endzone')
    return helmet_df


def add_fold(df):
    folds = pd.read_csv(FOLD_PATH)
    df = df.merge(folds.rename(columns={'game': 'game_key'}))
    return df


def _wrapper_func(args):
    _, df = args
    df = interpolate_video_player_df(df)
    return df


def interpolate_helmet_df(helmet_path):
    helmet_df = load_helmet_df(helmet_path)
    helmet_gb = helmet_df.groupby(['game_play', 'view', 'nfl_player_id'])

    dfs = []
    pool = Pool(processes=cpu_count())
    with tqdm(total=len(helmet_gb)) as t:
        for df in pool.imap_unordered(_wrapper_func, helmet_gb):
            dfs.append(df)
            t.update(1)

    interpolated_df = pd.concat(dfs).reset_index(drop=True)
    bbox_cols = ['left', 'width', 'top', 'height']
    interpolated_df[bbox_cols] = interpolated_df[bbox_cols].round().astype(int)

    print('before:', helmet_df.shape)
    print('after :', interpolated_df.shape)

    return interpolated_df


def main():
    train_interpolated_df = interpolate_helmet_df(TRAIN_HELMET_PATH)
    train_interpolated_df = add_fold(train_interpolated_df)
    train_interpolated_df.to_csv(TRAIN_SAVE_PATH, index=False)

    test_interpolated_df = interpolate_helmet_df(TEST_HELMET_PATH)
    test_interpolated_df.to_csv(TEST_SAVE_PATH, index=False)


def test():
    # some tests
    bbox1 = np.array([100, 100, 10, 10])
    bbox2 = np.array([110, 100, 10, 10])
    assert get_center(bbox1) == (105, 105)
    assert get_center(bbox2) == (115, 105)
    assert bbox_dist(bbox1, bbox2) == 10.0

    bbox1 = np.array([100, 100, 10, 10])
    bbox2 = np.array([100, 110, 10, 10])
    assert get_center(bbox1) == (105, 105)
    assert get_center(bbox2) == (105, 115)
    assert bbox_dist(bbox1, bbox2) == 10.0

    bbox1 = np.array([100, 100, 10, 10])
    bbox2 = np.array([110, 110, 10, 10])
    assert get_center(bbox1) == (105, 105)
    assert get_center(bbox2) == (115, 115)
    assert bbox_dist(bbox1, bbox2) == 2**0.5 * 10.0

    prev_frame, curr_frame, nxt_frame = 100, 109, 110
    prev_bbox = np.array([0, 0, 0, 0])
    nxt_bbox = np.array([10, 20, 30, 40])
    interpolate_bbox = interpolate_bboxes(prev_bbox, nxt_bbox, prev_frame, curr_frame, nxt_frame)
    np.testing.assert_almost_equal(interpolate_bbox, np.array([9., 18., 27., 36., 109., 1]))


if __name__ == "__main__":
    # test()
    main()
