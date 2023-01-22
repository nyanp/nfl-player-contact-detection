# -*- coding: utf-8 -*-
"""
@author: k_mat
data preparation for training NFL models
"""


import os
import glob
import json
import argparse

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_distance(df, tr_tracking, merge_col="datetime"):
    """
    Merges tracking data on player1 and 2 and computes the distance.
    """
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id", "x_position", "y_position"]
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={"x_position": "x_position_1", "y_position": "y_position_1"})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id", "x_position", "y_position"]
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={"x_position": "x_position_2", "y_position": "y_position_2"})
        .copy()
    )

    df_combo["distance"] = np.sqrt(
        np.square(df_combo["x_position_1"] - df_combo["x_position_2"])
        + np.square(df_combo["y_position_1"] - df_combo["y_position_2"])
    )
    return df_combo

def join_helmets_contact(game_play, labels, helmets, meta, view="Sideline", fps=59.94, only_center_of_step=False):
    """
    Joins helmets and labels for a given game_play. Results can be used for visualizing labels.
    Returns a dataframe with the joint dataframe, duplicating rows if multiple contacts occur.
    """
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

    gp_labs["datetime_ngs"] = pd.to_datetime(gp_labs["datetime"], utc=True)

    gp = gp_helms.merge(
        gp_labs[
            #["datetime_ngs", "nfl_player_id_1", "nfl_player_id_2", "contact_id"]
            ["datetime_ngs", "nfl_player_id_1", "nfl_player_id_2", "contact", "step"]
        ],
        left_on=["datetime_ngs", "nfl_player_id"],
        right_on=["datetime_ngs", "nfl_player_id_1"],
        how="left",
    )
    
    gp["center_frame_of_step"] = np.abs(gp["delta_from_round_val"]) 
    gp["center_frame_of_step"] = gp["center_frame_of_step"].values==gp.groupby("datetime_ngs")["center_frame_of_step"].transform("min").values
    if only_center_of_step:
        gp = gp[gp["center_frame_of_step"]]#.drop(columns=["center_frame_of_step"])
    return gp


def make_annotation_file(img_path, 
                         ann_path, 
                         img, 
                         ann,
                         save_img=True):
    if save_img:
        cv2.imwrite(img_path, img)
    # save ann
    with open(ann_path, 'w') as f:
        json.dump(ann, f)

def chain_list(inputs):
    outputs = []
    for inp_list in inputs:
        outputs += inp_list
    return outputs


def make_dataset_from_video(game_play_view,
                            video_path,# = "../data/nfl-impact-detection/train/*.mp4",
                            labels,# = "../data/train_label_tracking_merged_interp_w_motion.csv",
                            save_path,# = "../data/train_img_interp/",
                                 #not_interp = False,
                                 interval=1,
                                 only_center_of_step=False):
    os.makedirs(save_path, exist_ok=True)
    VIDEO_CODEC = "MP4V"
    video_name = os.path.basename(video_path)
    print(f"Running for {video_name}")
    labels = labels.query("video == @video_name").copy()
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame = 0
    mistake = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        if frame > total_frames:
            break
        # We need to add 1 to the frame count to match the label frame index
        # that starts at 1
        print("\r----- frame no. {}/{} in {}.mp4-----".format(frame, total_frames, video_name), end="")
        frame += 1        
        if frame % interval != 0:
            continue

        # Now, add the boxes
        df_frame =  labels.query("frame == @frame")
        #for df_frame in df_video.itertuples(index=False):
        if len(df_frame)==0:
            continue
        
        step_no = df_frame["step"].iloc[0]
        if only_center_of_step:
            if df_frame["center_frame_of_step"].iloc[0]==False:
                continue
        #df_frame =  labels[labels["step"]==step_no]
        #if len(df_frame)==0:
        #    continue
        
        
        img_path, ann_path = get_file_name(save_path, game_play_view, frame)
        rectangles = make_rectangle(df_frame)
        player_id_1 = df_frame["nfl_player_id"].values.astype(int).tolist()
        set_1 = set([0]+player_id_1)
        player_id_2 = df_frame["nfl_player_id_2"].values.tolist()
        contact_labels = df_frame["contact"].values.tolist()
        
        # players not in the image. very fool
        remaked_pid2 = []
        remaked_cont = []
        for pid2, cont in zip(player_id_2, contact_labels):
            bool_list = [id2 in set_1 for id2 in pid2]
            remaked_pid2.append([p for p, b in zip(pid2, bool_list) if b])
            remaked_cont.append([c for c, b in zip(cont, bool_list) if b])
        
        ##### 一時的
        
        set_1 = set([0]+player_id_1)
        set_2 = set(chain_list(remaked_pid2))
        #print(len(set_1), len(set_2))
        if len(set_1) < len(set_2):
            mistake += 1    
            
        
        #"""
        num_contact_labels = [len(ids) for ids in remaked_cont]
        num_player = len(player_id_1)
        ann = {"file": img_path, 
               "rectangles": rectangles,
               "player_id_1": player_id_1,
               "player_id_2": remaked_pid2,
               "contact_labels": remaked_cont,
               "num_contact_labels": num_contact_labels,
               "num_player": num_player,
               }
        
        make_annotation_file(img_path, 
                             ann_path, 
                             img, 
                             ann,
                             save_img=True)
        #"""
    print(mistake)



"""
merge dataset
"""

def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.
    """
    tracks = tracks.copy()
    tracks["game_play"] = (
        tracks["game_key"].astype("str")
        + "_"
        + tracks["play_id"].astype("str").str.zfill(6)
    )
    tracks["time"] = pd.to_datetime(tracks["time"])
    snap_dict = (
        tracks.query('event == "ball_snap"')
        .groupby("game_play")["time"]
        .first()
        .to_dict()
    )
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).astype(
        "timedelta64[ms]"
    ) / 1_000
    # Estimated video frame
    tracks["est_frame"] = (
        ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
    )
    return tracks

def make_interpolated_tracking(df_tracking, df_helmet):
    
    df_ref_play_frame = pd.DataFrame(df_helmet["video_frame"].unique())[0].str.rsplit('_', n=2, expand=True).rename(columns={0: 'game_play', 1: 'view', 2:"frame"}).drop("view",axis=1).drop_duplicates()
    df_ref_play_frame["frame"] = df_ref_play_frame["frame"].astype("int")
    df_ref_play_frame = df_ref_play_frame.sort_values(['game_play', "frame"])
    
    df_list = []

    for keys, _df_tracking in df_tracking.groupby(["player", "game_play"]):
        # skip because there are sideline player
        if keys[0] == "H00" or keys[0] == "V00":
            continue
        _df_ref_play_frame = df_ref_play_frame[df_ref_play_frame["game_play"]==keys[1]].copy()
        _df_ref_play_frame = _df_ref_play_frame.drop("game_play",axis=1)
        
        _df_tracking = _df_tracking.sort_values("est_frame")
        _df_tracking_copy = _df_tracking[["est_frame", "x", "y"]].copy().rename(columns={"est_frame": "next_est_frame", "x":"next_x", "y":"next_y"}).shift(-1).interpolate()
        #あとの線形補間の計算上add 1しておく
        _df_tracking_copy.iloc[-1, 0] += 1
        _df_tracking = pd.concat([_df_tracking, _df_tracking_copy], axis=1)
        

        # merge with frame and est_frame
        merged_df = pd.merge_asof(
                _df_ref_play_frame.copy(),
                _df_tracking,
                left_on="frame",
                right_on="est_frame",
                direction="backward",#'nearest',
            )
        df_list.append(merged_df)

    all_merged_df = pd.concat(df_list)
    w_1 = all_merged_df[["x", "y"]].values * ((all_merged_df["next_est_frame"].values-all_merged_df["frame"].values)/(all_merged_df["next_est_frame"].values-all_merged_df["est_frame"].values))[:,np.newaxis]
    w_2 = all_merged_df[["next_x", "next_y"]].values * ((all_merged_df["frame"].values-all_merged_df["est_frame"].values)/(all_merged_df["next_est_frame"].values-all_merged_df["est_frame"].values))[:,np.newaxis]
    motion = all_merged_df[["next_x", "next_y"]].values - all_merged_df[["x", "y"]].values
    ##interpolated_xy = pd.DataFrame(w_1 + w_2, columns={"x_interp", "y_interp"})
    ##tr_tracking = pd.concat([tr_tracking, interpolated_xy], axis=1).drop(["next_est_frame", "next_x", "next_y"],axis=1)
    all_merged_df["x_interp"] = w_1[:,0] + w_2[:,0]
    all_merged_df["y_interp"] = w_1[:,1] + w_2[:,1]
    all_merged_df["motion_x"] = motion[:,0]
    all_merged_df["motion_y"] = motion[:,1]
    
    all_merged_df = all_merged_df.drop(["next_est_frame", "next_x", "next_y"],axis=1)
    return all_merged_df

def merge_label_and_tracking_interp(tracking_df, label_df):

    tracking_with_game_index = tracking_df.set_index(["game_key", "play_id", "player"])

    df_list = []

    for key, _label_df in label_df.groupby(["game_key", "play_id", "view", "label"]):
        # skip because there are sideline player
        if key[3] == "H00" or key[3] == "V00":
            continue

        tracking_data = tracking_with_game_index.loc[(key[0], key[1], key[3])]
        _label_df = _label_df.sort_values("frame")
        _label_df["frame"] = _label_df["frame"].astype("int")

        # merge with frame and est_frame
        merged_df = pd.merge_asof(
            _label_df,
            tracking_data,
            left_on="frame",
            right_on="frame",
            direction='nearest',
        )
        df_list.append(merged_df)

    all_merged_df = pd.concat(df_list)
    all_merged_df = all_merged_df.sort_values(["video_frame", "label"], ignore_index=True)
    
    return all_merged_df




"""
extract images from movies
"""

def get_file_name(save_path, video_name, num_frame):
    #img_path = save_path + video_name + "_{:05}.jpg".format(num_frame)
    #img_path = save_path + video_name + "_{:05}.png".format(num_frame)
    #ann_path_train = save_path + video_name + "_{:05}_train.json".format(num_frame)
    #ann_path_test = save_path + video_name + "_{:05}_test.json".format(num_frame)
    img_path = save_path + "{:05}.jpg".format(num_frame)
    #img_path = save_path + video_name + "_{:05}.png".format(num_frame)
    ann_path_train = save_path + "{:05}_label.json".format(num_frame)
    return img_path, ann_path_train

def __make_annotation_file(img_path, ann_path, img, rectangles, location, player,
                         motions,
                         is_impact=None,
                         #from_current_to_points, 
                         save_img=True):
    if save_img:
        cv2.imwrite(img_path, img)
    # Save Log
    ann = {"file": img_path, 
           "rectangles": rectangles,
           "motions": motions,
           "location": location,
           "player": player,
           }
    if is_impact is not None:
        ann.update({"is_impact":is_impact})
    with open(ann_path, 'w') as f:
        json.dump(ann, f)

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
    rectangles=[]
    for i in range(len(top)):
        rectangle = [[left[i], top[i]],[right[i], top[i]],
                     [right[i], bottom[i]],[left[i], bottom[i]]]
        rectangles.append(rectangle)
    return rectangles

def make_locations(df):
    xs = df.x.values.reshape(-1)
    ys = df.y.values.reshape(-1)
    locations=[]
    for x,y in zip(xs,ys):
        loc = [[x, y]]
        locations.append(loc)
    return locations

def make_motions(df):
    xs = df.motion_x.values.reshape(-1)
    ys = df.motion_y.values.reshape(-1)
    motions=[]
    for x,y in zip(xs,ys):
        mot = [[x, y]]
        motions.append(mot)
    return motions

    
def main_make_dataset_from_video_w_motion(interval=1,
                                          video_path = "../data/nfl-impact-detection/train/*.mp4",
                                          track_path = "../data/train_player_tracking_interp_w_motion.csv",
                                          new_label_path = "../data/train_label_tracking_merged_interp_w_motion.csv",
                                          save_base = "../data/train_img_interp/",
                                          not_interp = False):
    if not os.path.exists(save_base): os.mkdir(save_base)
    videos = sorted(glob.glob(video_path))
    if DEBUG:
        videos = videos[::15]
    #video_labels = pd.read_csv(label_path)
    video_tracks = pd.read_csv(track_path)
    video_tracks["time"] = pd.to_datetime(video_tracks["time"])
    video_labels = pd.read_csv(new_label_path)
    plt.hist(video_labels["snap_offset"].values, bins=50)
    plt.show()
    if not_interp:
        video_labels = video_labels[np.abs((video_labels["frame"]-video_labels["est_frame"]).values)<0.5]
    #print(len(video_labels))
    
    for video_file in videos:
        num_frame = 1
        video_name = os.path.basename(video_file).split(".mp4")[0]
        save_path = os.path.join(save_base, video_name) + "/"
        if not os.path.exists(save_path): os.mkdir(save_path)
        single_video_labels = video_labels[video_labels["video"]==video_name+".mp4"]
        game_key = single_video_labels["game_key"].values[0]
        play_id = single_video_labels["play_id"].values[0]
        single_video_tracks = video_tracks[np.logical_and(video_tracks["game_key"]==game_key, video_tracks["play_id"]==play_id)]
        #return single_video_labels
        #single_video_unique_frames = single_video_labels["est_frame"].unique()#not interp
        single_video_unique_frames = single_video_labels["frame"].unique()
        #print(single_video_unique_frames)
        cap = cv2.VideoCapture(video_file)
        #if not cap.isOpened():
        #    return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        while ret:
            if num_frame%interval==0:
                if num_frame%60==1:
                    print("\r----- frame no. {}/{} in {}.mp4-----".format(num_frame, total_frames, video_name), end="")
                if num_frame in single_video_unique_frames:
                    img_path, ann_path, ann_path_test = get_file_name(save_path, video_name, num_frame)
                    single_frame_labels = single_video_labels[single_video_labels["frame"]==num_frame]
                    rectangles = make_rectangle(single_frame_labels)
                    locations = make_locations(single_frame_labels)
                    motions = make_motions(single_frame_labels)
                    players = single_frame_labels["label"].values.tolist()
                    #TODO
                    is_impact = single_frame_labels["isDefinitiveImpact"].values.astype(np.int32).tolist()
                    single_frame_track = single_video_tracks[single_video_tracks["time"]==single_frame_labels["time"].values[0]]
                    all_locations = make_locations(single_frame_track)
                    all_motions = make_motions(single_frame_track)
                    all_players = single_frame_track["player"].values.tolist()
                    #return locations, all_locations
                    #impacts = list(single_frame_labels.impact.fillna(0).values)        
                    make_annotation_file(img_path, ann_path, frame, rectangles, locations, players, motions=motions, is_impact=is_impact, save_img=False)
                    make_annotation_file(img_path, ann_path_test, frame, rectangles, all_locations, all_players, motions=all_motions, save_img=True)

            ret, frame = cap.read()
            num_frame += 1
    return video_labels


if __name__=="__main__":
    # Read in data files
    #"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    DEBUG = args.debug
    
    only_center_of_step = True
    
    setting_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"SETTINGS.json")

    DIRS = json.load(open(setting_file))#os.path.join(os.getcwd(), 'SETTINGS.json'), 'r'))
    BASE_DIR = DIRS["RAW_DATA_DIR"]
    SAVE_DIR = DIRS["TRAIN_DATA_DIR"]
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    
    
    # Labels and sample submission
    labels = pd.read_csv(f"{BASE_DIR}/train_labels.csv", parse_dates=["datetime"])
    
    ss = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
    
    # Player tracking data
    tr_tracking = pd.read_csv(
        f"{BASE_DIR}/train_player_tracking.csv", parse_dates=["datetime"]
    )
    te_tracking = pd.read_csv(
        f"{BASE_DIR}/test_player_tracking.csv", parse_dates=["datetime"]
    )
    
    # Baseline helmet detection labels
    tr_helmets = pd.read_csv(f"{BASE_DIR}/train_baseline_helmets.csv")
    te_helmets = pd.read_csv(f"{BASE_DIR}/test_baseline_helmets.csv")
    
    # Video metadata with start/stop timestamps
    tr_video_metadata = pd.read_csv(
        f"{BASE_DIR}/train_video_metadata.csv",
        parse_dates=["start_time", "end_time", "snap_time"],
    )
    
    
    
    th = 3.
    labels = pd.read_csv(f"{BASE_DIR}/train_labels.csv", parse_dates=["datetime"])
    # use only near pairs or with ground
    labels = compute_distance(labels, tr_tracking)
    labels = labels[np.logical_or(labels["distance"]<th, labels["nfl_player_id_2"]=="G")]
    print(labels.head())
    
    # label "G" -> 0
    labels['nfl_player_id_1'] = labels['nfl_player_id_1'].str.replace("G","0").astype(int)
    labels['nfl_player_id_2'] = labels['nfl_player_id_2'].str.replace("G","0").astype(int)
    #labels['nfl_player_id_1']#.astype(int)
    
    # add flipped label
    labels_flip = labels.copy().rename(columns={'nfl_player_id_1': 'nfl_player_id_2', 'nfl_player_id_2': 'nfl_player_id_1'})
    labels = pd.concat([labels,labels_flip], axis=0)
    print(labels.shape)
    
    labels_mini = pd.DataFrame(labels.groupby(["game_play", "datetime", "step","nfl_player_id_1"])["nfl_player_id_2"].apply(list))
    labels_mini["contact"] = labels.groupby(["game_play", "datetime", "step","nfl_player_id_1"])["contact"].apply(list)
    labels_mini["num_labels"] = labels_mini["contact"].apply(len)
    labels_mini["num_labels"].hist()
    labels_mini = labels_mini.reset_index()
    #"""
    for game_play in labels_mini["game_play"].unique():
        for view in ["Sideline", "Endzone"]:
            gp = join_helmets_contact(game_play, labels_mini, tr_helmets, tr_video_metadata, view, only_center_of_step=only_center_of_step)
            gp["contact"] = gp["contact"].apply(lambda d: d if isinstance(d, list) else [])
            gp["nfl_player_id_2"] = gp["nfl_player_id_2"].apply(lambda d: d if isinstance(d, list) else [])
            
            video_path = f"{BASE_DIR}/train/{game_play}_{view}.mp4"
            save_path = f"{SAVE_DIR}/train_img_10fps/{game_play}_{view}/" if only_center_of_step else f"{SAVE_DIR}/train_img/{game_play}_{view}/"
            game_play_view = f"{game_play}_{view}"
                                                  
            make_dataset_from_video(game_play_view,
                                    video_path,# = "../data/nfl-impact-detection/train/*.mp4",
                                    gp,# = "../data/train_label_tracking_merged_interp_w_motion.csv",
                                    save_path,# = "../data/train_img_interp/",
                                    #not_interp = False,
                                    interval=1 if only_center_of_step else 3,
                                    only_center_of_step=only_center_of_step)


    




    """
    # Labels and sample submission
    #labels = pd.read_csv(f'{BASE_DIR}/labels.csv')
    labels = pd.read_csv(f"{BASE_DIR}/train_labels.csv", parse_dates=["datetime"])

    ##ss = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')
    
    # Player tracking data
    tr_tracking = pd.read_csv(f'{BASE_DIR}/train_player_tracking.csv')
    ##te_tracking = pd.read_csv(f'{BASE_DIR}/test_player_tracking.csv')
    
    # Baseline helmet detection labels
    tr_helmets = pd.read_csv(f'{BASE_DIR}/train_baseline_helmets.csv')
    ##te_helmets = pd.read_csv(f'{BASE_DIR}/test_baseline_helmets.csv')
    
    # Extra image labels
    #img_labels = pd.read_csv(f'{BASE_DIR}/image_labels.csv')
    
 
    # Remake dataset for training.
    tr_tracking = add_track_features(tr_tracking)
    #tr_tracking.to_csv(f"{SAVE_DIR}/train_player_tracking_w_motion.csv", index=False)
    
    tr_tracking = make_interpolated_tracking(tr_tracking, tr_helmets)
    tr_tracking.to_csv(f"{SAVE_DIR}/train_player_tracking_interp_w_motion.csv", index=False)
    """
    """
    merged_df = merge_label_and_tracking_interp(tr_tracking, labels)
    merged_df.to_csv(f"{SAVE_DIR}/train_label_tracking_merged_interp_w_motion.csv", index=False)
        

    main_make_dataset_from_video_w_motion(video_path = f"{BASE_DIR}/train/*.mp4",
                                          track_path = f"{SAVE_DIR}/train_player_tracking_interp_w_motion.csv",
                                          new_label_path = f"{SAVE_DIR}/train_label_tracking_merged_interp_w_motion.csv",
                                          save_base = f"{SAVE_DIR}/train_img/",
                                          not_interp=True)
    
    main_make_dataset_from_video_w_motion(video_path = f"{BASE_DIR}/train/*.mp4",
                                          track_path = f"{SAVE_DIR}/train_player_tracking_interp_w_motion.csv",
                                          new_label_path = f"{SAVE_DIR}/train_label_tracking_merged_interp_w_motion.csv",
                                          save_base = f"{SAVE_DIR}/train_img_interp/",)
    """


    
    
