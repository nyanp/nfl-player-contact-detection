import os
import subprocess

import cv2
import numpy as np

IMG_HEIGHT, IMG_WIDTH = 720, 1280
HELMET_COLOR = (0, 0, 0)  # black
CORRECT_COLOR_IMPACT = (0, 255, 0)  # Green
FP_COLOR = (0, 0, 255)  # Red -1
FN_COLOR = (255, 0, 0)  # BLUE -2


TEMP_PATH = 'temp.mp4'
FPS = 30
IMG_WIDTH, IMG_HEIGHT = 1280, 720

COLOR_MAP = {
    -3: HELMET_COLOR,
    -2: FN_COLOR,
    -1: FP_COLOR,
    0: HELMET_COLOR,
    1: CORRECT_COLOR_IMPACT,
}

LABEL_MAP = {
    -3: None,  # 何？
    -2: "FN",
    -1: "FP",
    0: "TN",
    1: "TP",
}


def put_text(img, text, x_ratio, y_ratio, ):
    black = (0, 0, 0)
    white = (255, 255, 255)

    cv2.putText(img, text,
                (int(x_ratio * IMG_WIDTH), int(y_ratio * IMG_HEIGHT)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                black,
                3
                )
    cv2.putText(img, text,
                (int(x_ratio * IMG_WIDTH), int(y_ratio * IMG_HEIGHT)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                white,
                1
                )


def draw_helmet_per_frame(img, play_name, frame, frame_helmets):
    # add play_name text
    put_text(img, play_name, 0.01, 0.05)
    put_text(img, 'red:FP, blue:FN, green:TP', 0.01, 0.95)
    put_text(img, "Frame: " + str(frame), 0.75, 0.05)

    if len(frame_helmets) == 0:
        return img

    # adding helmet bounding boxes and player tags
    first_row = frame_helmets.iloc[0]
    put_text(img, f"gt:{first_row['contact']}", 0.30, 0.05)
    put_text(img, f"pred:{first_row['oof']:.3f}", 0.40, 0.05)
    put_text(img, LABEL_MAP[first_row['miss_label']], 0.55, 0.05)
    put_text(img, f"Step:{first_row['step']}", 0.65, 0.05)

    cols = ['player_label', 'left', 'width', 'top', 'height', 'miss_label']
    for row in frame_helmets[cols].values:
        player_label, bbox_left, bbox_width, bbox_top, bbox_height, miss_label = row
        # add helmet bbox
        color = COLOR_MAP[miss_label]
        cv2.rectangle(img,
                      (bbox_left, bbox_top),
                      (bbox_left + bbox_width, bbox_top + bbox_height),
                      color,
                      1
                      )
        # add player label
        cv2.putText(img,
                    str(player_label),
                    (bbox_left + 10, bbox_top + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    1
                    )
    return img


def finalize(tmp_output_path, output_path):
    # Not all browsers support the codec, we will re-load the file at tmp_output_path
    # and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            tmp_output_path,
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-hide_banner",
            "-loglevel",
            "error",
            "-vcodec",
            "libx264",
            output_path,
        ]
    )
    os.remove(tmp_output_path)


def step_to_frame(step):
    step0_frame = 300
    fps_frame = 59.94
    fps_step = 10
    return int(round(step * fps_frame / fps_step + step0_frame))


def frame_to_step(frame):
    step0_frame = 300
    fps_frame = 59.94
    fps_step = 10
    return int(round((frame - step0_frame) * fps_step / fps_frame))


def get_frame_range(df):
    min_step = df['step'].min()
    max_step = df['step'].max()

    return step_to_frame(min_step), step_to_frame(max_step)


def load_image(game_play, view, frame):
    image_path = f'../input/train_frames/{game_play}_{view}/{frame:06}.jpg'
    return cv2.imread(image_path)


def add_miss_label(df):
    df['miss_label'] = 0
    df.loc[(df['y_pred'] == 1) & (df['contact'] == 1), 'miss_label'] = 1
    df.loc[(df['y_pred'] == 0) & (df['contact'] == 0), 'miss_label'] = 0
    df.loc[(df['y_pred'] == 1) & (df['contact'] == 0), 'miss_label'] = -1  # FP
    df.loc[(df['y_pred'] == 0) & (df['contact'] == 1), 'miss_label'] = -2  # FN
    return df


def save_separate_videos(labels, helmet, game_play, nfl_player_id_1, nfl_player_id_2):
    single_labels = labels.query('game_play==@game_play & nfl_player_id_1==@nfl_player_id_1  & nfl_player_id_2==@nfl_player_id_2')
    single_labels = single_labels.fillna(0)  # fill easy sample pred with 0
    single_labels = add_miss_label(single_labels)
    min_frame, max_frame = get_frame_range(single_labels)
    single_helmet = helmet.query(
        'game_play==@game_play & nfl_player_id in [@nfl_player_id_1, @nfl_player_id_2] & frame >= @min_frame & frame <= @max_frame')
    single_helmet['step'] = single_helmet['frame'].map(frame_to_step)

    output_paths = []
    for view in ['Endzone', 'Sideline']:
        out = cv2.VideoWriter(TEMP_PATH,
                              cv2.VideoWriter_fourcc(*'MP4V'),
                              FPS, (IMG_WIDTH, IMG_HEIGHT))
        for frame in range(min_frame, max_frame + 1):
            view_image = load_image(game_play, view, frame)
            frame_helmet = single_helmet.query('view == @view & frame == @frame')
            frame_helmet = frame_helmet.merge(single_labels[['step', 'contact', 'y_pred', 'oof', 'miss_label']], on='step', how='left')
            view_image = draw_helmet_per_frame(view_image, game_play, frame, frame_helmet)
            out.write(view_image)
        out.release()
        output_path = f'output/{game_play}_{view}_{nfl_player_id_1}_{nfl_player_id_2}.mp4'
        finalize(TEMP_PATH, output_path)
        output_paths.append(output_path)
    return output_paths


def save_concat_video(labels, helmet, game_play, nfl_player_id_1, nfl_player_id_2):
    single_labels = labels.query('game_play==@game_play & nfl_player_id_1==@nfl_player_id_1  & nfl_player_id_2==@nfl_player_id_2')
    single_labels = single_labels.fillna(0)  # fill easy sample pred with 0
    single_labels = add_miss_label(single_labels)
    min_frame, max_frame = get_frame_range(single_labels)
    single_helmet = helmet.query(
        'game_play==@game_play & nfl_player_id in [@nfl_player_id_1, @nfl_player_id_2] & frame >= @min_frame & frame <= @max_frame')
    single_helmet['step'] = single_helmet['frame'].map(frame_to_step)

    out = cv2.VideoWriter(TEMP_PATH,
                          cv2.VideoWriter_fourcc(*'MP4V'),
                          FPS, (IMG_WIDTH, IMG_HEIGHT * 2))
    for frame in range(min_frame, max_frame + 1):
        images = []
        for view in ['Endzone', 'Sideline']:
            view_image = load_image(game_play, view, frame)
            frame_helmet = single_helmet.query('view == @view & frame == @frame')
            frame_helmet = frame_helmet.merge(single_labels[['step', 'contact', 'y_pred', 'oof', 'miss_label']], on='step', how='left')
            view_image = draw_helmet_per_frame(view_image, game_play, frame, frame_helmet)
            if view_image is not None:
                images.append(view_image)
            else:
                print(game_play, view, frame)
        if len(images):
            out.write(np.concatenate(images, axis=0))
    out.release()
    output_path = f'output/{game_play}_{nfl_player_id_1}_{nfl_player_id_2}.mp4'
    finalize(TEMP_PATH, output_path)
    return output_path
