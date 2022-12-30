# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 23:35:29 2022

@author: kmat
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 21:38:45 2022

@author: kmat
"""


import os
import glob
import json
import warnings
import argparse
import sys
import random
import time


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, CSVLogger, ModelCheckpoint
#from tensorflow.keras.utils import multi_gpu_model
import numpy as np
import matplotlib.pyplot as plt
#import pickle
#from PIL import Image
import pandas as pd
#import cv2
#import mlflow

from train_utils.tf_Augmentations_detection import Compose, Oneof, HorizontalFlip, VerticalFlip, Crop, Center_Crop, Resize, BrightnessContrast, CoarseDropout, HueShift, ToGlay, Blur, PertialBrightnessContrast, Shadow, GaussianNoise, Rotation
from train_utils.tf_Augmentations_detection import Center_Crop_by_box_shape, Crop_by_box_shape
from train_utils.scheduler import lrs_wrapper, lrs_wrapper_cos
from train_utils.dataloader import load_dataset, get_tf_dataset_gcn, preprop_inference
from model.model_gnn import build_gcn, build_gcn_1dcnn, build_dense
from model.model import matthews_correlation_fixed, matthews_correlation_best


        
class NFLGNN():
    def __init__(self, 
                 num_players=22, 
                 num_input_feature=8, 
                 num_adj=2,
                 sequence_length=7,
                 weight_file=None, 
                 is_train_model=False):
        
        print("\rLoading Models...", end="")
        
        self.num_players = num_players
        self.num_input_feature = num_input_feature
        self.num_adj = num_adj
        self.sequence_length = sequence_length
        self.is_train_model = is_train_model
        self.weight_file = weight_file
        self.load_model(weight_file, is_train_model)
        print("Loading Models......Finish")
        
    def load_model(self, weight_file=None, is_train_model=False):
        """build model and load weights"""
        #self.model, self.losses, self.loss_weights, self.metrics = build_gcn(self.num_players, self.num_input_feature, self.num_adj)
        self.model, self.losses, self.loss_weights, self.metrics = build_gcn_1dcnn(self.num_players, self.num_input_feature, self.num_adj)
        #self.model, self.losses, self.loss_weights, self.metrics = build_dense(self.num_players, self.num_input_feature, self.num_adj)
        if not weight_file is None:
            self.model.load_weights(weight_file)#, by_name=True, skip_mismatch=True)
            self.model.trainable=False
        if not is_train_model:
            self.tf_model = tf.function(lambda x: self.model(x))

    def train(self, train_dataset, val_dataset, save_dir, num_data, 
              learning_rate=0.002, n_epoch=150, batch_size=32, 
              ):
        if not self.is_train_model:
            raise ValueError("Model must be loaded as is_train_model=True")
        
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        
        lr_schedule = LearningRateScheduler(lrs_wrapper_cos(learning_rate, n_epoch, epoch_st=5))
        logger = CSVLogger(save_dir + 'log.csv')
        weight_file = "{epoch:02d}.hdf5"
        cp_callback = ModelCheckpoint(save_dir + weight_file, 
                                      monitor = 'val_loss', 
                                      save_weights_only = True,
                                      save_best_only = True,
                                      period = 100,
                                      verbose = 1)
        
                
        optim = Adam(lr=learning_rate, clipnorm=0.001)
        self.model.compile(loss = self.losses,
                           loss_weights = self.loss_weights, 
                           metrics = self.metrics,
                           optimizer = optim,
                           )
        
        
        """
        # test run
        tfd = get_tf_dataset_gcn(train_dataset, 
                                 batch_size=batch_size, 
                                 sequence_length=self.sequence_length,
                                 num_players=self.num_players,
                                 is_train=True,)
        for inp, targ in tfd.take(5):
            #print("run model")
            print("FFFFFFFFFFFFFFFF")
            print(inp["input_features"])
            print("AAAAAAAAAAAAAA")
            #print(inp["input_adjacency_matrix"])
            print(self.model(inp))
            #print(targ)
            #print(tf.math.reduce_std(inp["input_rgb"][...,3:], axis=[0,1,2]))
            #print(tf.reduce_max(inp["input_rgb"][...,3:], axis=[0,1,2]))
            #print(tf.reduce_min(inp["input_rgb"][...,3:], axis=[0,1,2]))
            #mean_size = tf.reduce_mean(inp["input_boxes"][0,...,2] - inp["input_boxes"][0,...,0])
            #plt.imshow(inp["input_rgb"][0,...,:3])
            #plt.title(mean_size.numpy())
            #plt.show()
            #print(tf.reduce_sum(tf.cast(targ["output_contact_label"][0]>-1, tf.float32)), "HAVELABELL")
            #plt.imshow(inp["input_rgb"][0,...,3:4])
            #plt.show()
            #plt.imshow(inp["input_rgb"][0,...,5:6])
            #plt.show()
            #plt.imshow(inp["input_rgb"][0,...,4:5])
            #plt.show()
            #plt.imshow(inp["input_rgb"][0,...,6:7])
            #plt.show()
            
        raise Exception
        #"""
        
        
        print("step per epoch", num_data[0]//batch_size, num_data[1]//batch_size)
        self.hist = self.model.fit(get_tf_dataset_gcn(train_dataset, 
                                                      batch_size=batch_size, 
                                                      sequence_length=self.sequence_length,
                                                      num_players=self.num_players,
                                                      is_train=True,), 
                    steps_per_epoch=num_data[0]//batch_size, 
                    epochs=n_epoch, 
                    validation_data=get_tf_dataset_gcn(val_dataset, 
                                               batch_size=batch_size, 
                                               sequence_length=self.sequence_length,
                                               num_players=self.num_players,
                                               is_train=False,
                                               ),
                    validation_steps=num_data[1]//batch_size,
                    callbacks=[lr_schedule, logger, cp_callback],
                    )
        print("Saving weights and results...")
        self.model.save_weights(save_dir + "final_weights.h5")
        csv_hist = save_dir + "hist.csv"
        pd.DataFrame(self.hist.history).to_csv(csv_hist, index=False)
        print("Done")
    
    def predict(self, player_3d_num_matrix,adj_matrix, step_range):
        inputs = [player_3d_num_matrix,adj_matrix, step_range]
        preds = self.tf_model(inputs)
        return preds
    
        
def set_seeds(num=111):
    tf.random.set_seed(num)
    np.random.seed(num)
    random.seed(num)
    os.environ["PYTHONHASHSEED"] = str(num)
    

def run_training_main(epochs=20, 
                      batch_size=4,
                      num_players=22, 
                      num_input_feature=8, 
                      num_adj=2,
                      sequence_length=7,
                      learning_ratio=0.005,
                      load_path="", 
                      save_path="", 
                      train_all=False):
    
    K.clear_session()
    set_seeds(111)
    
    
    """paths_endzone = sorted(glob.glob(DATA_PATH + "*Endzone"))
    paths_sideline = sorted(glob.glob(DATA_PATH + "*Sideline"))
    np.random.shuffle(paths_endzone)
    np.random.shuffle(paths_sideline)
    """    
    path_all_gameplay = sorted(glob.glob(os.path.join(DATA_PATH, "*")))
    #game_names = [int(os.path.basename(p)) for p in path_all_gameplay]
    game_names = [int(os.path.basename(p).split("_",1)[0]) for p in path_all_gameplay]
    print("ALL", len(game_names))
    fold_info = pd.read_csv(FOLD_PATH)
    fold_01_game = fold_info.loc[np.logical_or(fold_info["fold"]==0, fold_info["fold"]==1), "game"].values
    fold_23_game = fold_info.loc[np.logical_or(fold_info["fold"]==2, fold_info["fold"]==3), "game"].values
    mask_fold_01 = [name in fold_01_game for name in game_names]
    mask_fold_23 = [name in fold_23_game for name in game_names]
    path_fold_01 = list(np.array(path_all_gameplay)[mask_fold_01])# + list(np.array(side_path)[mask_fold_01])
    path_fold_23 = list(np.array(path_all_gameplay)[mask_fold_23])# + list(np.array(side_path)[mask_fold_23])
    
    if VAL23:
        train_path = path_fold_01
        val_path = path_fold_23
    else:
        train_path = path_fold_23
        val_path = path_fold_01
    print("train, val", len(train_path), len(val_path))
    
    #shuffle_indices = np.arange(len(end_path))
    np.random.shuffle(train_path)
    np.random.shuffle(val_path)
    #end_path = [end_path[idx] for idx in shuffle_indices]
    #side_path = [side_path[idx] for idx in shuffle_indices]
    

    #end_path_fold_12 = [p for p in end_path if p]
    #path = os.path.join(DATA_PATH, "58168_003392_Endzone/")
    #print(end_path[:120] + side_path[:120])
    #print([os.path.basename(p) for p in end_path[120:] + side_path[120:]])
    #train_files = load_dataset(end_path[:120] + side_path[:120])
    ###train_files = load_dataset(end_path[120:] + side_path[120:])
    #val_files = load_dataset(end_path[120:] + side_path[120:])[::5]
    ###val_files = load_dataset(end_path[:120] + side_path[:120])[::5]
    
    train_files = load_dataset(train_path, gcn_model=True)
    val_files = load_dataset(val_path, gcn_model=True)
    
    #step_rate = 6
    #step_rate_val = 6
    
    np.random.shuffle(train_files)
    np.random.shuffle(val_files)
    
    #num_data = [len(train_files)*step_rate, len(val_files)*step_rate_val]
    num_data = [sum([data["num_steps"]//sequence_length for data in train_files]),
                sum([data["num_steps"]//sequence_length-1 for data in val_files]),
                ]
    
    print(num_data)
    
    
    model_params = {"num_players": num_players,
                    "num_input_feature": num_input_feature,  
                    "sequence_length": sequence_length,
                    "num_adj": num_adj,
                    "weight_file": load_path,
                    "is_train_model": True,
                    }
    
    train_params = {"train_dataset": train_files,
                    "val_dataset": val_files,
                    "num_data": num_data,
                    "save_dir": save_path,
                    "learning_rate": learning_ratio, 
                    "n_epoch": epochs, 
                    "batch_size": batch_size,
                    }  
    
    #with tf.device('/device:GPU:0'):
    nfl = NFLGNN(**model_params)
    nfl.train(**train_params)      

def run_validation_predict(load_path,
                           num_players=22, 
                           num_input_feature=8, 
                           num_adj=2,
                           sequence_length=7,
                           
                           save_csv="output/pred_1dcnn.csv"):
    K.clear_session()
    set_seeds(111)
    
    
    """paths_endzone = sorted(glob.glob(DATA_PATH + "*Endzone"))
    paths_sideline = sorted(glob.glob(DATA_PATH + "*Sideline"))
    np.random.shuffle(paths_endzone)
    np.random.shuffle(paths_sideline)
    """    
    path_all_gameplay = sorted(glob.glob(os.path.join(DATA_PATH, "*")))
    #game_names = [int(os.path.basename(p)) for p in path_all_gameplay]
    game_names = [int(os.path.basename(p).split("_",1)[0]) for p in path_all_gameplay]
    print("ALL", len(game_names))
    fold_info = pd.read_csv(FOLD_PATH)
    fold_01_game = fold_info.loc[np.logical_or(fold_info["fold"]==0, fold_info["fold"]==1), "game"].values
    fold_23_game = fold_info.loc[np.logical_or(fold_info["fold"]==2, fold_info["fold"]==3), "game"].values
    mask_fold_01 = [name in fold_01_game for name in game_names]
    mask_fold_23 = [name in fold_23_game for name in game_names]
    path_fold_01 = list(np.array(path_all_gameplay)[mask_fold_01])# + list(np.array(side_path)[mask_fold_01])
    path_fold_23 = list(np.array(path_all_gameplay)[mask_fold_23])# + list(np.array(side_path)[mask_fold_23])
    
    if VAL23:
        train_path = path_fold_01
        val_path = path_fold_23
    else:
        train_path = path_fold_23
        val_path = path_fold_01
    print("train, val", len(train_path), len(val_path))
    
    #train_files = load_dataset(train_path, gcn_model=True)
    val_dataset = load_dataset(val_path, gcn_model=True)[:]
    val_game_plays = [os.path.basename(p) for p in val_path][:]
    
    model_params = {"num_players": num_players,
                    "num_input_feature": num_input_feature,  
                    "sequence_length": sequence_length,
                    "num_adj": num_adj,
                    "weight_file": load_path,
                    "is_train_model": False,
                    }
    
    
    nfl = NFLGNN(**model_params)
    
    
    
    start_time = time.time()
    all_outputs = []
    
    
    
    
    
    #TODO ラベルは.Tでマックスとるようにする。フリップはいってねえええ
    # それはそれでフリップ不成立はまずくないか？
    
    
    """
    for inp, targ in get_tf_dataset_gcn(val_dataset, 
                               batch_size=8, 
                               sequence_length=sequence_length,
                               num_players=num_players,
                               is_train=False,
                               ).take(20):
        
        #pred_g, pred_p = nfl.predict(inp["input_features"], inp["input_adjacency_matrix"], inp["step_range"])
        pred_g, pred_p = nfl.model(inp, training=False)
        print(pred_g.shape, pred_p.shape)
        pred_g = pred_g.numpy().flatten()
        pred_p = pred_p.numpy().flatten()
        
        label_g = targ["g_contact"].numpy().flatten()
        label_p = targ["p_contact"].numpy().flatten()
        print((targ["p_contact"].numpy()[1,1,:,:]>-0.5).sum())
        
        print("P------------")
        gt = label_p.reshape(-1)[label_p.reshape(-1)>-0.5]#label_p * (label_p.reshape(-1)>-0.5).astype(float)
        pr = pred_p.flatten()[label_p.reshape(-1)>-0.5]
        tf_gt_labels = tf.cast(np.array(gt), tf.float32)
        tf_predicted_labels = tf.cast(pr.flatten(), tf.float32)
        for th in np.linspace(0.1, 0.9, 9):
            print(th, matthews_correlation_fixed(tf_gt_labels, tf_predicted_labels, threshold=th))
        
        print("G------------")
        gt = label_g * (label_g.reshape(-1)>-0.5).astype(float)
        pr = pred_g.flatten()
        tf_gt_labels = tf.cast(np.array(gt), tf.float32)
        tf_predicted_labels = tf.cast(pr.flatten()[:len(gt)], tf.float32)
        for th in np.linspace(0.1, 0.9, 9):
            print(th, matthews_correlation_fixed(tf_gt_labels, tf_predicted_labels, threshold=th))
        
    
    
    raise Exception()
    """
    
    
    for i, [game_play, data] in enumerate(zip(val_game_plays, val_dataset)):#.take(1000):
        print("\r----- running {}/{} -----".format(i+1, len(val_dataset)), end="")
        preprocessed = preprop_inference(data)
        inp = [preprocessed["player_3d_num_matrix"], preprocessed["adj_matrix"], preprocessed["step_range_norm"][...,tf.newaxis]]
        pred_g, pred_p = nfl.predict(*inp)
        
        num_players = int(preprocessed["num_players"])
        steps = np.stack([data["step_range"].numpy()] * (num_players + 1) * num_players, axis=-1)
        pred_g = pred_g[:,:,:num_players].numpy().reshape(-1, 1, num_players)
        pred_p = pred_p[:,:,:num_players, :num_players].numpy().reshape(-1, num_players, num_players)
        pred_all = np.concatenate([pred_g, pred_p], axis=1)
        num_total_steps = pred_all.shape[0]
        player_1 = np.concatenate([preprocessed["unique_players"]] * (num_players + 1), axis=0)
        player_2 = np.stack([np.pad(preprocessed["unique_players"], [[1,0]])] * num_players).T.flatten()
        player_1 = np.tile(player_1[tf.newaxis, :], [num_total_steps, 1])
        player_2 = np.tile(player_2[tf.newaxis, :], [num_total_steps, 1])
        
        #data["label_p_contact"] = data["label_p_contact"] - 2 * tf.cast(data["p2p_adj_dist_matrix"] >= 3, tf.int32).numpy().reshape(-1,22,22)
        #label_p = data["label_p_contact"].reshape(-1, num_players, num_players)
        #label_g = data["label_g_contact"].reshape(-1, 1, num_players)
        
        output_df = pd.DataFrame(steps.flatten(), columns=["step"])
        output_df["nfl_player_id_1"] = player_1.flatten()
        output_df["nfl_player_id_2"] = player_2.flatten()
        output_df["pred_1dcnn"] = pred_all.flatten()
        output_df["game_play"] = game_play
        output_df = output_df.groupby(["game_play","step","nfl_player_id_1","nfl_player_id_2"])["pred_1dcnn"].mean().reset_index()
        
        all_outputs.append(output_df)
        
        show_results = False
        if show_results:
            label_p = data["label_p_contact"].numpy().reshape(-1, num_players, num_players)
            label_g = data["label_g_contact"].numpy().reshape(-1, 1, num_players)
            #raise Exception()
            label_all = np.concatenate([label_g,label_p], axis=1).reshape(-1)
            
            print("P------------")
            #gt = (label_p.reshape(-1)>0.5).astype(float)
            #pr = pred_p.flatten()
            gt = label_p.reshape(-1)[label_p.reshape(-1)>-0.5]#label_p * (label_p.reshape(-1)>-0.5).astype(float)
            pr = pred_p.flatten()[label_p.reshape(-1)>-0.5]
            
            
            tf_gt_labels = tf.cast(np.array(gt), tf.float32)
            tf_predicted_labels = tf.cast(pr.flatten()[:len(gt)], tf.float32)
            for th in np.linspace(0.1, 0.9, 9):
                print(th, matthews_correlation_fixed(tf_gt_labels, tf_predicted_labels, threshold=th))
            
            print("G------------")
            #gt = (label_g.reshape(-1)>0.5).astype(float)
            #pr = pred_g.flatten()
            gt = label_g.reshape(-1)[label_g.reshape(-1)>-0.5]#label_p * (label_p.reshape(-1)>-0.5).astype(float)
            pr = pred_g.flatten()[label_g.reshape(-1)>-0.5]
            
            tf_gt_labels = tf.cast(np.array(gt), tf.float32)
            tf_predicted_labels = tf.cast(pr.flatten()[:len(gt)], tf.float32)
            for th in np.linspace(0.1, 0.9, 9):
                print(th, matthews_correlation_fixed(tf_gt_labels, tf_predicted_labels, threshold=th))
            
            
        
        #raise Exception()
    pd.concat(all_outputs, axis=0).to_csv(save_csv, index=False)
    """
        #mean_size = tf.reduce_mean(inp["input_boxes"][0,...,2] - inp["input_boxes"][0,...,0])
        #plt.imshow(inp["input_rgb"][0,...,:3])
        #plt.title(mean_size.numpy())
        #plt.show()
        
        
        preds = nfl.predict(**inp)
        pred_mask, pred_label = preds
        
        
        if draw_pred:
            dev = abs(pred_label.numpy()[0] - targ["output_contact_label"].numpy()[0]) * (targ["output_contact_label"].numpy()[0]>0)
            argmax_idx = np.argmax(dev)
            #pred_label = pred_label.numpy()[0]
            #pred_mask = pred_mask.numpy()[0, :, :, :, 0]
            if dev[argmax_idx] > 0.8:
                img = inp["input_rgb"].numpy()[0]
                pairs = inp["input_pairs"].numpy()[0]
                label = targ["output_contact_label"].numpy()[0]
                view_contact_mask(img, pairs, 
                                      pred_mask.numpy()[0, :, :, :, 0], 
                                      pred_label.numpy()[0], 
                                      gt_label=label,
                                      title_epoch="",
                                      idx= argmax_idx)
        
        for pairs, p, gt, num in zip(inp["input_pairs"].numpy(), pred_label.numpy(), targ["output_contact_label"].numpy(), info["num_labels"]):
            predicted_labels += list(p[:num])
            gt_labels += list(gt[:num])
            ground_labels += list((pairs[:num,1]==0))
        counter += batch_size
        time_elapsed = time.time() - start_time
        fps_inference = counter / time_elapsed
        print(f"\r{round(fps_inference, 1)} fps, at {counter} / {len(val_files)}data", end="")
    print(np.sum(gt_labels))
    print(np.sum(predicted_labels))
    print(len(predicted_labels))
    #print(predicted_labels)
    #print(gt_labels)
    
    print("ground------")
    tf_gt_labels = tf.cast(np.array(gt_labels)[np.array(ground_labels)], tf.float32)
    tf_predicted_labels = tf.cast(np.array(predicted_labels)[np.array(ground_labels)], tf.float32)
    for th in np.linspace(0.1, 0.9, 9):
        print(th, matthews_correlation_fixed(tf_gt_labels, tf_predicted_labels, threshold=th))
    print("not ground------")
    tf_gt_labels = tf.cast(np.array(gt_labels)[~np.array(ground_labels)], tf.float32)
    tf_predicted_labels = tf.cast(np.array(predicted_labels)[~np.array(ground_labels)], tf.float32)
    for th in np.linspace(0.1, 0.9, 9):
        print(th, matthews_correlation_fixed(tf_gt_labels, tf_predicted_labels, threshold=th))
    
    print("full------")
    
    gt_labels_w_easy_sample = gt_labels + [0] * (len(gt_labels)*8)
    predicted_labels_w_easy_sample = predicted_labels + [1e-7] * (len(predicted_labels)*8)
    
    gt_labels = tf.cast(gt_labels, tf.float32)
    predicted_labels = tf.cast(predicted_labels, tf.float32)
    gt_labels_w_easy_sample = tf.cast(gt_labels_w_easy_sample, tf.float32)
    predicted_labels_w_easy_sample = tf.cast(predicted_labels_w_easy_sample, tf.float32)
    for th in np.linspace(0.1, 0.9, 9):
        print(th, matthews_correlation_fixed(gt_labels, predicted_labels, threshold=th))
        print(th, matthews_correlation_fixed(gt_labels_w_easy_sample, predicted_labels_w_easy_sample, threshold=th))
    """

# metricsじっそう！！！ LR大き目。　少し解像度小さいかも。切り取るのは要検討。いらないならcoordsはずして任意サイズにするのも手だと思う
# とりあえずアテンションモデルをトライ。
"""
ハイパラ：
プレイヤクロップの範囲・解像度・畳み込み回数や方法
回転系augまだいれてない

オクルージョンにおけるコンタクトをどのように定義するのか。
各プレイヤ、手前と奥のmodeling
ボックスの座標で判断してもいいけど、しゃがんでるやつがまざると苦しそう。。
奥にある場合は前後差し替え、同程度の場合はどっちも、などでもいいし、重みづけ（これ自体を学習）でもいい。

ベクトル画像、正しく方向出せてるかチェックしたほうがいいかも…

グランドコンタクトとコンタクトマップ別でもいいかも？

"""
if __name__=="__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_rate', type=float, default=1.0)
    args = parser.parse_args()
    DEBUG = args.debug
    batch_rate = args.batch_rate
    """
    DEBUG = False
    num_epoch =  2 if DEBUG else 30

    setting_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"SETTINGS.json")
    DIRS = json.load(open(setting_file))
    BASE_DIR = DIRS["RAW_DATA_DIR"]
    TRAIN_DIR = DIRS["TRAIN_DATA_DIR"]
    WEIGHT_DIR = DIRS["WEIGHT_DIR"]# model/weights/
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    
    #DATA_PATH_EXT = os.path.join(BASE_DIR, "images/")
    #ANNOTATINO_PATH_EXT = os.path.join(BASE_DIR, "image_labels.csv")
    DATA_PATH = os.path.join(TRAIN_DIR, "gp_data_for_gcn/")
    FOLD_PATH = os.path.join(TRAIN_DIR, "game_fold.csv")
    #TARGET_SIZE = 25

    # normal resolution model
    #FIXED_SIZE_DETECTION = False
    
    #トラック-画像差分系特徴も入れてみる
    VAL23 = True
    if VAL23:
        NAME = "fold01"
    else:
        NAME = "fold23"
        
    run_train = False
    if run_train:
        #CROP_SHAPE=(432, 768, 3)
        save_path = os.path.join(WEIGHT_DIR, f"ex002_gnn_run003_{NAME}train_1dcnn/")
        run_training_main(epochs=int(num_epoch*1.2), 
                              batch_size=int(64/4),
                              num_players=22, 
                              num_input_feature=8, 
                              num_adj=2+8,
                              sequence_length=18, #18
                              learning_ratio=0.0075,
                         load_path=None,#save_path+"final_weights.h5",#os.path.join(WEIGHT_DIR, "map_model_final_weights.h5"),
                         save_path=save_path)
        
    else:
        # ex000_contdet_run015_fold01train_fixed_size
        # ex000_contdet_run016_fold01train_not_fixed_size
        run_validation_predict(os.path.join(WEIGHT_DIR, f"ex002_gnn_run002_{NAME}train_1dcnn/final_weights.h5"),
                                   num_players=22, 
                                   num_input_feature=8, 
                                   num_adj=2+8,
                                   sequence_length=18,
                                   save_csv=f"output/pred_1dcnn_{NAME}.csv")
    
    
    
    """
    VAL23 = False
    if VAL23:
        name = "fold01"
    else:
        name = "fold23"
    run_train = True
    AUTO_RESIZE = False
    if run_train:
        FIXED_SIZE_DETECTION = False
        save_path = os.path.join(WEIGHT_DIR, f"ex000_contdet_run019_{name}train_fixed_size_retry/")
        run_training_main(epochs=num_epoch, 
                          batch_size=int(8),#int(12), 
                         #input_shape=(512+64, 896+128, 3),#(384, 640, 3), 
                         #output_shape=(256+32, 448+64),
                         input_shape=(512, 896, 3),#(384, 640, 3), 
                         output_shape=(256, 448),
                         #(192, 320), 
                         load_path=None,#os.path.join(WEIGHT_DIR, "ex000_contdet_run010_120videos_w_flow/final_weights.h5"),
                         save_path=save_path)
    
    else:
        # ex000_contdet_run015_fold01train_fixed_size
        # ex000_contdet_run016_fold01train_not_fixed_size
        run_validation_predict(os.path.join(WEIGHT_DIR, "ex000_contdet_run015_fold01train_fixed_size/final_weights.h5"),
                                   input_shape=(704, 1280, 3), 
                                   output_shape=(352, 640),
                                   draw_pred=False)
    
    
    """
