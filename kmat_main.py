
"""
ロード及び前処理まわり。
"""
import gc
import glob
import json
import os
import pickle
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from threading import Thread
from turtle import distance
from typing import List

import cv2
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from feature_engineering.table import add_basic_features, tracking_prep
from kmat.model.model import matthews_correlation_fixed
from kmat.train_contact_det import NFLContact, view_contact_mask
from kmat.train_utils.dataloader import inference_preprocess
from kmat.train_utils.tf_Augmentations_detection import Center_Crop, Compose
from scipy.optimize import minimize
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import (GroupKFold, PredefinedSplit,
                                     StratifiedGroupKFold)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
from utils.nfl import (TRACK_COLS, Config, ModelSize, expand_contact_id, expand_helmet, merge_tracking,
                       read_csv_with_cache)


def cv2bgr_to_tf32(img):
    # return tf.cast(img[:,:,::-1], tf.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return tf.cast(img, tf.float32)


def make_rectangle(df):
    top = df.top.values.reshape(-1)
    left = df.left.values.reshape(-1)
    width = df.width.values.reshape(-1)
    height = df.height.values.reshape(-1)
    bottom = top + height
    right = left + width
    top = top.tolist()
    left = left.tolist()
    bottom = bottom.tolist()
    right = right.tolist()
    rectangles = []
    for i in range(len(top)):
        rectangle = [[left[i], top[i]], [right[i], top[i]],
                     [right[i], bottom[i]], [left[i], bottom[i]]]
        rectangles.append(rectangle)
    return rectangles


def prepare_matching_dataframe(game_play, tr_tracking, helmets, meta, view="Sideline", fps=59.94, only_center_of_step=True):
    tr_tracking = tr_tracking.query("game_play == @game_play").copy()
    gp_helms = helmets.query("game_play == @game_play").copy()

    start_time = meta.query("game_play == @game_play and view == @view")[
        "start_time"
    ].values[0]

    gp_helms["datetime"] = (
        pd.to_timedelta(gp_helms["frame"] * (1 / fps), unit="s") + start_time
    )
    gp_helms["datetime"] = pd.to_datetime(gp_helms["datetime"], utc=True)
    gp_helms["datetime_ngs"] = (
        pd.DatetimeIndex(gp_helms["datetime"] + pd.to_timedelta(50, "ms"))
        .floor("100ms")
        .values
    )
    gp_helms["datetime_ngs"] = pd.to_datetime(gp_helms["datetime_ngs"], utc=True)
    gp_helms["delta_from_round_val"] = (gp_helms["datetime"] - gp_helms["datetime_ngs"]).dt.total_seconds()

    tr_tracking["datetime_ngs"] = pd.to_datetime(tr_tracking["datetime"], utc=True)
    gp_helms = gp_helms.merge(
        tr_tracking[["datetime_ngs", "step", "x_position", "y_position", "nfl_player_id"]],
        left_on=["datetime_ngs", "nfl_player_id"],
        right_on=["datetime_ngs", "nfl_player_id"],
        how="left",
    )
    gp_helms["center_frame_of_step"] = np.abs(gp_helms["delta_from_round_val"])
    gp_helms["center_frame_of_step"] = gp_helms["center_frame_of_step"].values == gp_helms.groupby(
        "datetime_ngs")["center_frame_of_step"].transform("min").values
    # 複数minimumが存在するケースもあるので気を付ける(あとでグループ平均とるなど)
    if only_center_of_step:
        gp_helms = gp_helms[gp_helms["center_frame_of_step"]].drop(columns=["center_frame_of_step"])
    # 追加した。ok?
    gp_helms = gp_helms[gp_helms["view"] == view]

    return gp_helms


def prepare_cnn_dataframe(game_play, labels, helmets, meta, view="Sideline", fps=59.94):
    gp_labs = labels.query("game_play == @game_play").copy()
    gp_helms = helmets.query("game_play == @game_play").copy()

    start_time = meta.query("game_play == @game_play and view == @view")[
        "start_time"
    ].values[0]

    gp_helms["datetime"] = (
        pd.to_timedelta(gp_helms["frame"] * (1 / fps), unit="s") + start_time
    )
    gp_helms["datetime"] = pd.to_datetime(gp_helms["datetime"], utc=True)
    gp_helms["datetime_ngs"] = (
        pd.DatetimeIndex(gp_helms["datetime"] + pd.to_timedelta(50, "ms"))
        .floor("100ms")
        .values
    )
    gp_helms["datetime_ngs"] = pd.to_datetime(gp_helms["datetime_ngs"], utc=True)
    gp_helms["delta_from_round_val"] = (gp_helms["datetime"] - gp_helms["datetime_ngs"]).dt.total_seconds()

    if "datetime_ngs" not in gp_labs.columns:
        gp_labs["datetime_ngs"] = pd.to_datetime(gp_labs["datetime"], utc=True)
    gp_helms = gp_helms.merge(
        gp_labs[["datetime_ngs", "step"]].drop_duplicates(),
        on=["datetime_ngs"],
        how="left",
    )
    gp_helms["center_frame_of_step"] = np.abs(gp_helms["delta_from_round_val"])
    gp_helms["center_frame_of_step"] = gp_helms["center_frame_of_step"].values == gp_helms.groupby(
        "step")["center_frame_of_step"].transform("min").values
    # 複数minimumが存在するケースもあるので気を付ける(あとでグループ平均とるなど)

    # 追加した。ok?
    gp_helms = gp_helms[gp_helms["view"] == view]

    return gp_labs, gp_helms


"""
- Camaro-san Reader
"""
# import the necessary packages

if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue


class FileVideoStream:
    def __init__(self, cv2_stream, transform=None, queue_size=64):
        self.stream = cv2_stream  # cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            if not self.Q.full():
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    self.stopped = True

                if grabbed and self.transform:
                    frame = self.transform(frame)
                    # add the frame to the queue
                    self.Q.put(frame)
            else:
                time.sleep(0.01)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()


"""
video loader for cnn
"""


class Video2Input():
    def __init__(self,
                 game_play_view,
                 video_path,
                 labels,
                 helms,
                 frame_interval=1,
                 only_center_frame_of_step=True):

        VIDEO_CODEC = "MP4V"
        self.video_name = os.path.basename(video_path)
        print(f"Preparing {self.video_name}")
        self.labels = labels.copy()  # .query("video == @self.video_name").copy()
        self.helms = helms.query("video == @self.video_name").copy()
        min_frame = self.helms["frame"].values[np.argmin(self.helms["step"].values)]
        self.start_frame = max(0, min_frame - 5)
        self.vidcap = cv2.VideoCapture(video_path)
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        #self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = self.start_frame
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.interval = frame_interval

        self.vidcap = FileVideoStream(self.vidcap,
                                      # transform=cv2bgr_to_tf32,
                                      transform=lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                                      ).start()

    def get_next(self, only_center_frame_of_step=True):
        """
        only_center_frame_of_step:
            if True, use only center_frame_of_step and neglect other frames.
        """
        while True:
            #it_worked, img = self.vidcap.read()
            img = self.vidcap.read()
            # if not it_worked:
            if not self.vidcap.running():
                self.vidcap.stop()
                return self.current_frame, None
            # We need to add 1 to the frame count to match the label frame index
            # that starts at 1
            self.current_frame += 1
            if self.current_frame % self.interval != 0:
                continue

            # Now, add the boxes
            df_frame_helms = self.helms[self.helms["frame"] == self.current_frame]  # uery("frame == @self.current_frame")
            if len(df_frame_helms) == 0:
                continue
            step_no = df_frame_helms["step"].iloc[0]
            if only_center_frame_of_step:
                if df_frame_helms["center_frame_of_step"].iloc[0] == False:
                    continue
            df_frame_label = self.labels[self.labels["step"] == step_no]
            if len(df_frame_label) == 0:
                continue

            rectangles = make_rectangle(df_frame_helms)
            player_id = df_frame_helms["nfl_player_id"].values.astype(int).tolist()
            player_id_1 = df_frame_label["nfl_player_id_1"].values.astype(int)  # .tolist()
            player_id_2 = df_frame_label["nfl_player_id_2"].values.astype(int)  # .tolist()

            # remove players not in the image
            set_player = set([0] + player_id)
            player_1_exist = [p in set_player for p in player_id_1]
            player_2_exist = [p in set_player for p in player_id_2]
            players_exist = np.logical_and(player_1_exist, player_2_exist)

            player_id_1 = player_id_1[players_exist]
            player_id_2 = player_id_2[players_exist]
            contact_labels = df_frame_label["contact"].values[players_exist]  # .tolist()
            num_contact_labels = len(contact_labels)
            num_player = len(set_player) - 1

            contact_pairlabels = np.vstack([player_id_1, player_id_2, contact_labels]).T

            if num_player == 0:
                continue

            data = {"rgb": tf.cast(img, tf.float32),  # img,#cv2bgr_to_tf32(img),
                    "rectangles": tf.cast(rectangles, tf.float32),
                    "player_id": tf.cast(player_id, tf.int32),
                    "contact_pairlabels": tf.cast(contact_pairlabels, tf.int32),
                    "num_labels": tf.cast(num_contact_labels, tf.int32),
                    "num_player": num_player,
                    "img_height": 720,
                    "img_width": 1280,
                    }
            step_frame_no = [step_no, self.current_frame]  # single step(100ms) contains multiple frames
            return step_frame_no, data


class DataStacker:
    def __init__(self, batch_size=2, as_list=False):
        self.stacked = None
        self.length = 0
        self.batch_size = batch_size
        self.as_list = as_list

    def add(self, data):
        if self.stacked is None:
            self.stacked = {k: [data[k]] for k in data.keys()}
        else:
            self.stacked = {k: self.stacked[k] + [data[k]] for k in data.keys()}
        self.length += 1
        if self.length > self.batch_size:
            raise Exception("stacked too much")

    def get_if_ready(self, reset=True, neglect_readiness=False):
        if self.length == self.batch_size or neglect_readiness:
            is_ready = True
            if self.as_list:
                stacked = self.stacked
            else:
                stacked = {k: tf.stack(self.stacked[k]) for k in self.stacked.keys()}
            if reset:
                self.reset()
        else:
            stacked = None
            is_ready = False
        return is_ready, stacked

    def is_ready(self):
        return self.length == self.batch_size

    def reset(self):
        self.stacked = None
        self.length = 0


"""
CNN main実行系関数。
foldとるところ少し一時的。。。
"""


def build_model(input_shape, output_shape, load_path, num_max_pair=153):

    model_params = {"input_shape": input_shape,
                    "output_shape": output_shape,
                    "weight_file": load_path,
                    "is_train_model": False,
                    }
    model = NFLContact(**model_params)

    transforms = [
        Center_Crop(p=1, min_height=input_shape[0], min_width=input_shape[1]),
    ]
    transforms = Compose(transforms)

    def preprocessor(x): return inference_preprocess(x,
                                                     transforms=transforms,
                                                     max_box_num=23,
                                                     max_pair_num=num_max_pair,  # enough large
                                                     padding=True)
    return model, preprocessor


def get_foldcnntrain_set(train_df, train_01_val_23=False):
    # get game_play
    # fold_info = pd.read_csv(f"{KMAT_PATH}/output/game_fold.csv")
    # fold_01_game = fold_info.loc[np.logical_or(fold_info["fold"] == 0, fold_info["fold"] == 1), "game"].values
    # fold_23_game = fold_info.loc[np.logical_or(fold_info["fold"] == 2, fold_info["fold"] == 3), "game"].values

    # game_play_names = list(train_df["game_play"].unique())
    # game_names = [int(gp.rsplit("_", 1)[0]) for gp in game_play_names]
    # mask_fold_01 = [name in fold_01_game for name in game_names]
    # mask_fold_23 = [name in fold_23_game for name in game_names]

    # game_play_fold_23 = np.array(game_play_names)[np.array(mask_fold_23)]
    # game_play_fold_01 = np.array(game_play_names)[np.array(mask_fold_01)]

    train_title = "fold01" if train_01_val_23 else "fold23"
    # val_title = "fold23" if train_01_val_23 else "fold01"

    # val_set = game_play_fold_23 if train_01_val_23 else game_play_fold_01
    # print(f"train by {train_title}, val by {len(val_set)} {val_title}")

    load_path = f"{KMAT_PATH}/model/weights/ex000_contdet_run022_{train_title}train_ground_othermask/final_weights.h5"
    input_shape = (704, 1280, 3)
    output_shape = (352, 640)
    num_max_pair = train_df.groupby(["game_play", "step"])["nfl_player_id_2"].size().max()
    print(f"cnn model predict {num_max_pair} pairs at max")
    model, preprocessor = build_model(input_shape, output_shape, load_path, num_max_pair)
    return model, preprocessor, val_set

def load_model(train_df, load_path):
    # load_path = f"{KMAT_PATH}/model/weights/ex000_contdet_run022_{train_title}train_ground_othermask/final_weights.h5"
    input_shape = (704, 1280, 3)
    output_shape = (352, 640)
    num_max_pair = train_df.groupby(["game_play", "step"])["nfl_player_id_2"].size().max()
    print(f"cnn model predict {num_max_pair} pairs at max")
    model, preprocessor = build_model(input_shape, output_shape, load_path, num_max_pair)
    return model, preprocessor


class CNNEnsembler:
    def __init__(self, models, num_output_items=2):
        self.models = models
        self.num_models = len(models)
        self.num_output_items = num_output_items

    def predict(self, *args, **kargs):
        predictions = [[] for _ in range(self.num_output_items)]
        for model in self.models:
            preds = model.predict(*args, **kargs)
            if len(preds) != self.num_output_items:
                raise Exception("set num_output_item correctly")
            for i in range(self.num_output_items):
                predictions[i].append(preds[i])
        predictions = [tf.reduce_mean(tf.stack(preds), axis=0) for preds in predictions]
        return predictions


def pred_by_cnn(train_df, tr_tracking, tr_helmets, tr_video_metadata, game_playes_to_pred, model, preprocessor, is_train_dataset=True):

    only_center_frame_of_step = True
    batch_size = 4
    draw_pred = False

    if only_center_frame_of_step:  # every six frames(60fps->10fps)
        frame_interval = 1  # must be set 1
    else:
        frame_interval = 1  # 中央フレームだけでは見えにくいフレームもあると思うので、理想的には1step予測に数フレーム使う方がいいと思う。

    df_cnn_preds_end = []
    df_cnn_preds_side = []
    for game_play in game_playes_to_pred:  # labels_mini["game_play"].unique():game_play_names
        print(game_play)
        for view in ["Sideline", "Endzone"]:
            gp_labs, gp_helms = prepare_cnn_dataframe(game_play, train_df, tr_helmets,
                                                      tr_video_metadata, view,
                                                      )
            if is_train_dataset:
                video_path = f"{cfg.INPUT}/train/{game_play}_{view}.mp4"
            else:
                video_path = f"{cfg.INPUT}/test/{game_play}_{view}.mp4"

            game_play_view = f"{game_play}_{view}"

            vi = Video2Input(game_play_view,
                             video_path,
                             gp_labs, gp_helms,
                             frame_interval=frame_interval)

            stacked_inputs = DataStacker(batch_size)
            stacked_targets = DataStacker(batch_size)
            stacked_info = DataStacker(batch_size, as_list=False)
            start_time = time.time()
            counter = 0
            d_counter = 0
            predicted_labels = []
            gt_labels = []
            players_1 = []
            players_2 = []
            step_numbers = []
            frame_numbers = []
            is_last_batch = False
            while True:
                step_frame_no, data = vi.get_next(only_center_frame_of_step=only_center_frame_of_step)

                if data is None:
                    is_last_batch = True
                    is_ready = True
                    if stacked_inputs.length == 0:
                        break
                else:
                    d_counter += 1

                    try:
                        inputs, targets, info = preprocessor(data)
                        # 今少し良くないやり方.要修正、このプリ処理で消えるやつがおる。数変わるのでちゅうい
                    except BaseException:  # クロップレンジ外。一時的
                        print("ラベル無し")
                        continue
                    step_numbers += [step_frame_no[0]] * int(info["num_labels"])
                    frame_numbers += [step_frame_no[1]] * int(info["num_labels"])

                    stacked_inputs.add(inputs)
                    stacked_targets.add(targets)
                    stacked_info.add(info)
                    is_ready = stacked_inputs.is_ready()

                if is_ready:

                    _, inp = stacked_inputs.get_if_ready(neglect_readiness=is_last_batch)
                    _, targ = stacked_targets.get_if_ready(neglect_readiness=is_last_batch)
                    _, info = stacked_info.get_if_ready(neglect_readiness=is_last_batch)

                    preds = model.predict(**inp)
                    pred_mask, pred_label = preds

                    if draw_pred:
                        view_contact_mask(inp["input_rgb"].numpy()[0],
                                          inp["input_pairs"].numpy()[0],
                                          pred_mask.numpy()[0, :, :, :, 0],
                                          pred_label.numpy()[0],
                                          gt_label=targ["output_contact_label"].numpy()[0],
                                          title_epoch="")

                    for i, [p, gt, num, pairs] in enumerate(zip(pred_label.numpy(), targ["output_contact_label"].numpy(),
                                                                info["num_labels"], info["contact_pairlabels"])):

                        predicted_labels += list(p[:num])
                        gt_labels += list(gt[:num])
                        players_1 += list(pairs[:num, 0].numpy())
                        players_2 += list(pairs[:num, 1].numpy())

                    counter += batch_size
                    time_elapsed = time.time() - start_time
                    fps_inference = counter / time_elapsed
                    print(f"\r{round(fps_inference, 1)} fps, at {vi.current_frame} / {vi.total_frames} in video", end="")

                if is_last_batch:
                    break

            df_pred = pd.DataFrame(np.array(predicted_labels).reshape(-1, 1), columns=[f"cnn_pred_{view}"])
            df_pred["nfl_player_id_1"] = players_1
            df_pred["nfl_player_id_2"] = players_2
            df_pred["game_play"] = game_play
            df_pred["step"] = step_numbers
            df_pred["frame"] = frame_numbers
            df_pred["gt_tmp"] = gt_labels

            df_pred = df_pred.groupby(["step", "game_play", "nfl_player_id_1", "nfl_player_id_2"]).mean().reset_index()
            # min, max, mean
            if view == "Sideline":
                df_cnn_preds_side.append(df_pred)
            else:
                df_cnn_preds_end.append(df_pred)

            show_each_matthews = False
            if show_each_matthews:
                gt_labels = tf.cast(gt_labels, tf.float32)
                predicted_labels = tf.cast(predicted_labels, tf.float32)
                for th in np.linspace(0.1, 0.9, 9):
                    print(th, matthews_correlation_fixed(gt_labels, predicted_labels, threshold=th))

    df_cnn_preds_end = pd.concat(df_cnn_preds_end, axis=0)
    df_cnn_preds_side = pd.concat(df_cnn_preds_side, axis=0)
    return df_cnn_preds_end, df_cnn_preds_side


def cnn_features_val(train_df, tr_tracking, tr_helmets, tr_video_metadata, cnn_pred_path="/kaggle/input/nfl2cnnpred1218"):

    if "distance" in train_df.columns:
        dist_btw_players = train_df["distance"].copy()
    else:
        dist_btw_players = distance(train_df["x_position_1"], train_df["y_position_1"], train_df["x_position_2"], train_df["y_position_2"])
    dist_thresh = 3
    ground_id = train_df["nfl_player_id_2"].min()
    train_df_mini = train_df[np.logical_or(dist_btw_players < dist_thresh, train_df["nfl_player_id_2"] == ground_id)].copy()
    train_df_mini['nfl_player_id_2'] = train_df_mini['nfl_player_id_2'].replace(ground_id, 0).astype(int)

    if os.path.exists(cnn_pred_path):
        print("load existing pred file")
        df_cnn_preds_end_01 = pd.read_csv(os.path.join(cnn_pred_path, "fold01_cnn_pred_end.csv"))
        df_cnn_preds_side_01 = pd.read_csv(os.path.join(cnn_pred_path, "fold01_cnn_pred_side.csv"))
        df_cnn_preds_end_23 = pd.read_csv(os.path.join(cnn_pred_path, "fold23_cnn_pred_end.csv"))
        df_cnn_preds_side_23 = pd.read_csv(os.path.join(cnn_pred_path, "fold23_cnn_pred_side.csv"))

        df_cnn_preds_end_01[["nfl_player_id_1", "nfl_player_id_2"]] = df_cnn_preds_end_01[["nfl_player_id_1", "nfl_player_id_2"]].astype(int)
        df_cnn_preds_side_01[["nfl_player_id_1", "nfl_player_id_2"]] = df_cnn_preds_side_01[["nfl_player_id_1", "nfl_player_id_2"]].astype(int)
        df_cnn_preds_end_23[["nfl_player_id_1", "nfl_player_id_2"]] = df_cnn_preds_end_23[["nfl_player_id_1", "nfl_player_id_2"]].astype(int)
        df_cnn_preds_side_23[["nfl_player_id_1", "nfl_player_id_2"]] = df_cnn_preds_side_23[["nfl_player_id_1", "nfl_player_id_2"]].astype(int)

    else:
        print("start CNN validation pred")

        train_01_val_23 = False
        model, preprocessor, val_set = get_foldcnntrain_set(train_df_mini, train_01_val_23=train_01_val_23)
        df_cnn_preds_end_01, df_cnn_preds_side_01 = pred_by_cnn(
            train_df_mini, tr_tracking, tr_helmets, tr_video_metadata, val_set, model, preprocessor, is_train_dataset=True)

        train_01_val_23 = True
        model, preprocessor, val_set = get_foldcnntrain_set(train_df_mini, train_01_val_23=train_01_val_23)
        df_cnn_preds_end_23, df_cnn_preds_side_23 = pred_by_cnn(
            train_df_mini, tr_tracking, tr_helmets, tr_video_metadata, val_set, model, preprocessor, is_train_dataset=True)

        os.makedirs(cnn_pred_path, exist_ok=True)
        df_cnn_preds_end_01.to_csv(os.path.join(cnn_pred_path, "fold01_cnn_pred_end.csv"), index=False)
        df_cnn_preds_side_01.to_csv(os.path.join(cnn_pred_path, "fold01_cnn_pred_side.csv"), index=False)
        df_cnn_preds_end_23.to_csv(os.path.join(cnn_pred_path, "fold23_cnn_pred_end.csv"), index=False)
        df_cnn_preds_side_23.to_csv(os.path.join(cnn_pred_path, "fold23_cnn_pred_side.csv"), index=False)

    df_side = pd.concat([df_cnn_preds_side_01, df_cnn_preds_side_23], axis=0)
    df_end = pd.concat([df_cnn_preds_end_01, df_cnn_preds_end_23], axis=0)

    #df_side = df_cnn_preds_side_01
    #df_end = df_cnn_preds_end_01

    df_end['nfl_player_id_2'] = df_end['nfl_player_id_2'].replace(0, ground_id).astype(int)
    df_side['nfl_player_id_2'] = df_side['nfl_player_id_2'].replace(0, ground_id).astype(int)

    return df_side, df_end


def cnn_features_test(train_df, tr_tracking, tr_helmets, tr_video_metadata, ):
    if "distance" in train_df.columns:
        dist_btw_players = train_df["distance"].copy()
    else:
        dist_btw_players = distance(train_df["x_position_1"], train_df["y_position_1"], train_df["x_position_2"], train_df["y_position_2"])

    dist_thresh = 3
    ground_id = train_df["nfl_player_id_2"].min()  # kmat modelでは0を使用していたのでid No調整。関数出る前に戻す。
    train_df_mini = train_df[np.logical_or(dist_btw_players < dist_thresh, train_df["nfl_player_id_2"] == ground_id)].copy()
    train_df_mini['nfl_player_id_2'] = train_df_mini['nfl_player_id_2'].replace(ground_id, 0).astype(int)
    train_df_mini["contact"] = 1.  # not use

    game_plays = list(train_df_mini["game_play"].unique())
    # べた書き
    model_01, preprocessor = load_model(train_df, "../input/mfl2cnnkmat1225/model/weights/ex000_contdet_run022_fold01train_ground_othermask/final_weights.h5")
    model_23, preprocessor = load_model(train_df, "../input/mfl2cnnkmat1225/model/weights/ex000_contdet_run022_fold23train_ground_othermask/final_weights.h5")
    # model_01, preprocessor, _ = get_foldcnntrain_set(train_df, train_01_val_23=True)
    # model_23, preprocessor, _ = get_foldcnntrain_set(train_df, train_01_val_23=False)
    model = CNNEnsembler([model_01, model_23], num_output_items=2)
    df_end, df_side = pred_by_cnn(train_df_mini, tr_tracking, tr_helmets, tr_video_metadata, game_plays, model, preprocessor, is_train_dataset=False)

    df_end['nfl_player_id_2'] = df_end['nfl_player_id_2'].replace(0, ground_id).astype(int)
    df_side['nfl_player_id_2'] = df_side['nfl_player_id_2'].replace(0, ground_id).astype(int)
    return df_side, df_end


if __name__ == "__main__":
    phase = 'test'

    cfg = Config(
        EXP_NAME='exp007_remove_hard_example_large_camaro_kmat_cnn_feats_p2p_interpolate',
        PRETRAINED_MODEL_PATH='./',
        MODEL_SIZE=ModelSize.LARGE)

    # 重複した前処理になるが一旦許容する
    te_tracking = read_csv_with_cache(
        "test_player_tracking.csv", cfg, usecols=TRACK_COLS)

    sub = read_csv_with_cache("sample_submission.csv", cfg)
    test_df = expand_contact_id(sub)
    test_df = pd.merge(test_df,
                       te_tracking[["step", "game_play", "datetime"]].drop_duplicates(),
                       on=["game_play", "step"], how="left")
    test_df = expand_helmet(cfg, test_df, "test")

    te_tracking = tracking_prep(te_tracking)
    test_df = merge_tracking(
        test_df,
        te_tracking,
        [
            "team", "position", "x_position", "y_position",
            "speed", "distance", "direction", "orientation", "acceleration",
            "sa",
            # "direction_p1_diff", "direction_m1_diff",
            # "orientation_p1_diff", "orientation_m1_diff",
            # "distance_p1", "distance_m1"
        ]
    )
    test_df = add_basic_features(test_df)

    # te_helmets = read_csv_with_cache("test_baseline_helmets.csv", cfg)
    te_helmets = pd.read_csv('../input/nfl-player-contact-detection/test_baseline_helmets.csv')
    te_meta = pd.read_csv(os.path.join(cfg.INPUT, "test_video_metadata.csv"),
                          parse_dates=["start_time", "end_time", "snap_time"])

    df_side, df_end = cnn_features_test(test_df, te_tracking, te_helmets, te_meta)
    df_side.to_csv('kmat_side_df.csv', index=False)
    df_end.to_csv('kmat_end_df.csv', index=False)
