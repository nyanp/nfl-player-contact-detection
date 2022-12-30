# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 21:38:45 2022

@author: kmat
"""


import argparse
import glob
import json
import os
import random
import sys
import time
import warnings

import matplotlib.pyplot as plt
# from tensorflow.keras.utils import multi_gpu_model
import numpy as np
# import pickle
# from PIL import Image
import pandas as pd
import tensorflow as tf
from kmat.model.model import (build_model, build_model_nomask,
                         matthews_correlation_fixed)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (Callback, CSVLogger,
                                        LearningRateScheduler, ModelCheckpoint)
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from kmat.train_utils.dataloader import (get_tf_dataset, get_tf_dataset_inference,
                                         get_tf_dataset_inference_auto_resize,
                                         inference_preprocess, load_dataset)
from kmat.train_utils.scheduler import lrs_wrapper, lrs_wrapper_cos
from kmat.train_utils.tf_Augmentations_detection import (Blur, BrightnessContrast,
                                                         Center_Crop,
                                                         Center_Crop_by_box_shape,
                                                         CoarseDropout, Compose,
                                                         Crop, Crop_by_box_shape,
                                                         GaussianNoise,
                                                         HorizontalFlip, HueShift,
                                                         Oneof,
                                                         PertialBrightnessContrast,
                                                         Resize, Rotation, Shadow,
                                                         ToGlay, VerticalFlip)

#import cv2
#import mlflow


def view_contact_mask(img, pairs,
                      pred_mask,
                      pred_label,
                      gt_label=None,
                      title_epoch="",
                      idx=None):
    if gt_label is not None:
        max_label_ind = idx or np.argmax(gt_label)
        is_ground = pairs[max_label_ind].min() == 0
        gt = gt_label[max_label_ind]
    else:
        max_label_ind = idx or np.argmax(pred_label)
        is_ground = pairs[max_label_ind].min() == 0
        gt = "test"

    pred_label = pred_label[max_label_ind]
    pred_mask = pred_mask[max_label_ind]
    pred_mask_double = tf.image.resize(pred_mask[np.newaxis, :, :, np.newaxis], size=img.shape[:2]).numpy()[0, :, :, 0]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title(f"Original Image: {title_epoch} {gt}, ground:{is_ground}")

    ax[1].imshow(pred_mask)
    ax[1].set_title(f"Predicted Mask: {title_epoch} {pred_label}")

    blend_img = img * (1. - pred_mask_double[:, :, np.newaxis])
    blend_img[:, :, 0] += pred_mask_double
    ax[2].imshow(blend_img)
    ax[2].set_title("Blended Image")
    # fig.savefig(f"{SAVE_PATH}{epoch:03d}.png")
    plt.show()
    plt.close()


class TrainMonitor(tf.keras.callbacks.Callback):
    """
    TODO add bbox on rgb image
    """

    def __init__(self, mini_dataset, mask_model, num_check=10, num_freq=2):
        super(TrainMonitor, self).__init__()
        self.dataset = mini_dataset
        self.mask_model = mask_model
        self.num_check = num_check
        self.num_freq = num_freq

    def on_epoch_end(self, epoch, logs=None):

        self.mask_model.trainable = False
        if epoch % self.num_freq == 0:
            for inp, targ in self.dataset.take(self.num_check):
                pred_mask_raw, pred_label = self.mask_model(inp, training=False)
                pred_label = pred_label.numpy()[0]
                pred_mask = pred_mask_raw.numpy()[0, :, :, :, 0]

                img = inp["input_rgb"].numpy()[0, ..., :3]
                pairs = inp["input_pairs"].numpy()[0]
                label = targ["output_contact_label"].numpy()[0]

                view_contact_mask(img, pairs,
                                  pred_mask,
                                  pred_label,
                                  gt_label=label,
                                  title_epoch=epoch)

                """
                max_label_ind = np.argmax(label)
                is_ground = pairs[max_label_ind].min()==0

                gt = label[max_label_ind]
                pred_label = pred_label[max_label_ind]
                pred_mask = pred_mask[max_label_ind]
                pred_mask_double = tf.image.resize(pred_mask_raw[:, max_label_ind], size=img.shape[:2]).numpy()[0, :, :, 0]

                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                ax[0].imshow(img)
                ax[0].set_title(f"Original Image: {epoch:03d} {gt}, ground:{is_ground}")

                ax[1].imshow(pred_mask)
                ax[1].set_title(f"Predicted Mask: {epoch:03d} {pred_label}")

                blend_img = img * (1. - pred_mask_double[:,:,np.newaxis])
                blend_img[:,:,0] += pred_mask_double
                ax[2].imshow(blend_img)
                ax[2].set_title("Blended Image")
                #fig.savefig(f"{SAVE_PATH}{epoch:03d}.png")
                plt.show()
                plt.close()
                #plt.figure(figsize=(10,10))
                #plt.imshow(keras.preprocessing.image.array_to_img(test_recons_images))
                #plt.show()
                """

        self.mask_model.trainable = True


class NFLContact():
    def __init__(self,
                 input_shape=(512, 512, 4),
                 output_shape=(256, 256),
                 weight_file=None,
                 is_train_model=False):

        print("\rLoading Models...", end="")

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.is_train_model = is_train_model
        self.weight_file = weight_file
        self.load_model(weight_file, is_train_model)
        print("Loading Models......Finish")

    def load_model(self, weight_file=None, is_train_model=False):
        """build model and load weights"""
        self.model, self.sub_model, self.losses, self.loss_weights, self.metrics = build_model(self.input_shape,
                                                                                               minimum_stride=self.input_shape[0] // self.output_shape[0],
                                                                                               is_train=self.is_train_model,
                                                                                               backbone="effv2s",
                                                                                               from_scratch=False)
        if not weight_file is None:
            self.model.load_weights(weight_file)  # , by_name=True, skip_mismatch=True)
        if not is_train_model:
            self.sub_model.trainable = False
            for layer in self.model.layers:
                layer.trainable = False
                if "efficient" in layer.name:
                    for l in layer.layers:
                        l.trainable = False
            self.tf_model = tf.function(lambda x: self.sub_model(x))  # , experimental_relax_shapes=True)
            # self.tf_model = lambda x: self.sub_model(x)#, experimental_relax_shapes=True)

    def train(self, train_dataset, val_dataset, save_dir, num_data,
              learning_rate=0.002, n_epoch=150, batch_size=32,
              ):
        if not self.is_train_model:
            raise ValueError("Model must be loaded as is_train_model=True")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        lr_schedule = LearningRateScheduler(lrs_wrapper_cos(learning_rate, n_epoch, epoch_st=5))
        logger = CSVLogger(save_dir + 'log.csv')
        weight_file = "{epoch:02d}.hdf5"
        cp_callback = ModelCheckpoint(save_dir + weight_file,
                                      monitor='val_loss',
                                      save_weights_only=True,
                                      save_best_only=True,
                                      period=10,
                                      verbose=1)

        optim = Adam(lr=learning_rate, clipnorm=0.001)
        self.model.compile(loss=self.losses,
                           loss_weights=self.loss_weights,
                           metrics=self.metrics,
                           optimizer=optim,
                           )

        if FIXED_SIZE_DETECTION:
            print("define scale by boxshape. effective??")
            crop_helmet_target_size = 20
            transforms_train = [
                HorizontalFlip(p=0.5),
                Crop_by_box_shape(
                    p=1,
                    target_box_length=crop_helmet_target_size,
                    target_img_height=self.input_shape[0],
                    target_img_width=self.input_shape[1],
                    img_height=720,
                    img_width=1280),
                Resize(height=self.input_shape[0], width=self.input_shape[1], target_height=self.output_shape[0], target_width=self.output_shape[1]),
                BrightnessContrast(p=1.0),
                HueShift(p=0.8, min_offset=-0.25, max_offset=0.25),
                Oneof(
                    transforms=[PertialBrightnessContrast(p=0.2,
                                                          max_holes=3, max_height=80, max_width=80,
                                                          min_holes=1, min_height=30, min_width=30,
                                                          min_offset=0, max_offset=30,
                                                          min_multiply=1.0, max_multiply=1.5),
                                Shadow(p=0.4,
                                       max_holes=3, max_height=120, max_width=120,
                                       min_holes=1, min_height=50, min_width=50,
                                       min_strength=0.2, max_strength=0.8, shadow_color=0.0)
                                ],
                    probs=[0.2, 0.2]
                ),
                Blur(p=0.1),
                GaussianNoise(p=0.1, min_var=10, max_var=40),
                ToGlay(p=0.1),
            ]
            transforms_val = [
                Center_Crop_by_box_shape(
                    p=1,
                    target_box_length=crop_helmet_target_size,
                    target_img_height=self.input_shape[0],
                    target_img_width=self.input_shape[1],
                    img_height=720,
                    img_width=1280),
                Resize(height=self.input_shape[0], width=self.input_shape[1], target_height=self.output_shape[0], target_width=self.output_shape[1]),
            ]
        else:
            transforms_train = [
                HorizontalFlip(p=0.5),
                Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                #Crop(p=1, min_height=CROP_SHAPE[0], min_width=CROP_SHAPE[1]),
                #Resize(height=self.input_shape[0], width=self.input_shape[1], target_height=self.output_shape[0], target_width=self.output_shape[1]),
                BrightnessContrast(p=1.0),
                HueShift(p=0.8, min_offset=-0.25, max_offset=0.25),
                Oneof(
                    transforms=[PertialBrightnessContrast(p=0.2,
                                                          max_holes=3, max_height=80, max_width=80,
                                                          min_holes=1, min_height=30, min_width=30,
                                                          min_offset=0, max_offset=30,
                                                          min_multiply=1.0, max_multiply=1.5),
                                Shadow(p=0.4,
                                       max_holes=3, max_height=120, max_width=120,
                                       min_holes=1, min_height=50, min_width=50,
                                       min_strength=0.2, max_strength=0.8, shadow_color=0.0)
                                ],
                    probs=[0.2, 0.2]
                ),

                Blur(p=0.1),
                GaussianNoise(p=0.1, min_var=10, max_var=40),
                ToGlay(p=0.1),
            ]

            transforms_val = [
                Center_Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                #Center_Crop(p=1, min_height=CROP_SHAPE[0], min_width=CROP_SHAPE[1]),
                #Resize(height=self.input_shape[0], width=self.input_shape[1], target_height=self.output_shape[0], target_width=self.output_shape[1]),
            ]

        train_transforms = Compose(transforms_train)
        val_transforms = Compose(transforms_val)
        """
        # test run
        tfd = get_tf_dataset(train_dataset,
                       batch_size=1,
                       transforms=train_transforms,
                       is_train=True,
                       max_pair_num=120,)
        for inp, targ in tfd.take(50):
            #print("run model")
            #print(self.model(inp).shape)
            print(tf.math.reduce_std(inp["input_rgb"][...,3:], axis=[0,1,2]))
            print(tf.reduce_max(inp["input_rgb"][...,3:], axis=[0,1,2]))
            print(tf.reduce_min(inp["input_rgb"][...,3:], axis=[0,1,2]))
            mean_size = tf.reduce_mean(inp["input_boxes"][0,...,2] - inp["input_boxes"][0,...,0])
            plt.imshow(inp["input_rgb"][0,...,:3])
            plt.title(mean_size.numpy())
            plt.show()
            print(tf.reduce_sum(tf.cast(targ["output_contact_label"][0]>-1, tf.float32)), "HAVELABELL")
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
        monitor_dataset = get_tf_dataset(val_dataset[:100],
                                         batch_size=1,
                                         transforms=val_transforms,
                                         is_train=False,
                                         )
        monitor = TrainMonitor(monitor_dataset, self.sub_model)

        print("step per epoch", num_data[0] // batch_size, num_data[1] // batch_size)
        self.hist = self.model.fit(get_tf_dataset(train_dataset,
                                                  # self.input_shape,
                                                  # self.output_shape,
                                                  batch_size=batch_size,
                                                  transforms=train_transforms,
                                                  is_train=True,
                                                  max_pair_num=50,
                                                  # use_cut_mix=False,
                                                  ),
                                   steps_per_epoch=num_data[0] // batch_size,
                                   epochs=n_epoch,
                                   validation_data=get_tf_dataset(val_dataset,
                                                                  # self.input_shape,
                                                                  # self.output_shape,
                                                                  batch_size=batch_size,
                                                                  transforms=val_transforms,
                                                                  max_pair_num=50,
                                                                  is_train=False,
                                                                  ),
                                   validation_steps=num_data[1] // batch_size,
                                   callbacks=[lr_schedule, logger, cp_callback, monitor],
                                   )
        print("Saving weights and results...")
        self.model.save_weights(save_dir + "final_weights.h5")
        csv_hist = save_dir + "hist.csv"
        pd.DataFrame(self.hist.history).to_csv(csv_hist, index=False)
        print("Done")

    def predict(self, input_rgb, input_boxes, input_pairs):
        inputs = [input_rgb, input_boxes, input_pairs]
        preds = self.tf_model(inputs)
        return preds


def set_seeds(num=111):
    tf.random.set_seed(num)
    np.random.seed(num)
    random.seed(num)
    os.environ["PYTHONHASHSEED"] = str(num)


def run_training_main(epochs=20,
                      batch_size=4,
                      input_shape=(448, 768, 3),
                      output_shape=(224, 384),
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
    end_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Endzone")))
    side_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Sideline")))
    game_names = [int(os.path.basename(p).split("_", 1)[0]) for p in end_path]
    print("ALL", len(end_path), len(side_path))

    fold_info = pd.read_csv(FOLD_PATH)
    fold_01_game = fold_info.loc[np.logical_or(fold_info["fold"] == 0, fold_info["fold"] == 1), "game"].values
    fold_23_game = fold_info.loc[np.logical_or(fold_info["fold"] == 2, fold_info["fold"] == 3), "game"].values
    mask_fold_01 = [name in fold_01_game for name in game_names]
    mask_fold_23 = [name in fold_23_game for name in game_names]

    path_fold_01 = list(np.array(end_path)[mask_fold_01]) + list(np.array(side_path)[mask_fold_01])
    path_fold_23 = list(np.array(end_path)[mask_fold_23]) + list(np.array(side_path)[mask_fold_23])

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

    train_files = load_dataset(train_path)
    val_files = load_dataset(val_path)[::10]

    step_rate = 6
    step_rate_val = 1

    np.random.shuffle(train_files)
    np.random.shuffle(val_files)

    num_data = [len(train_files) // step_rate, len(val_files) // step_rate_val]
    print(num_data)

    print(input_shape)
    model_params = {"input_shape": input_shape,
                    "output_shape": output_shape,
                    "weight_file": load_path,
                    "is_train_model": True,
                    }

    train_params = {"train_dataset": train_files,
                    "val_dataset": val_files,
                    "num_data": num_data,
                    "save_dir": save_path,
                    "learning_rate": 0.0004 * batch_size / 8,
                    "n_epoch": epochs,
                    "batch_size": batch_size,
                    }

    # with tf.device('/device:GPU:0'):
    nfl = NFLContact(**model_params)
    nfl.train(**train_params)


def run_validation_predict(load_path,
                           input_shape=(704, 1280, 3),
                           output_shape=(352, 640),
                           draw_pred=False):
    K.clear_session()
    set_seeds(111)
    """
    paths_endzone = sorted(glob.glob(DATA_PATH + "*Endzone"))
    paths_sideline = sorted(glob.glob(DATA_PATH + "*Sideline"))
    np.random.shuffle(paths_endzone)
    np.random.shuffle(paths_sideline)

    end_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Endzone")))
    side_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Sideline")))
    shuffle_indices = np.arange(len(end_path))
    np.random.shuffle(shuffle_indices)
    end_path = [end_path[idx] for idx in shuffle_indices]
    side_path = [side_path[idx] for idx in shuffle_indices]
    print("ALL", len(end_path), len(side_path))
    #path = os.path.join(DATA_PATH, "58168_003392_Endzone/")
    #train_files = load_dataset(end_path[:40] + side_path[:40])
    val_files = load_dataset(end_path[-8:] + side_path[-8:])
    print(len(val_files))
    np.random.shuffle(val_files)
    """

    end_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Endzone")))
    side_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Sideline")))
    game_names = [int(os.path.basename(p).split("_", 1)[0]) for p in end_path]
    print("ALL", len(end_path), len(side_path))

    fold_info = pd.read_csv(FOLD_PATH)
    fold_01_game = fold_info.loc[np.logical_or(fold_info["fold"] == 0, fold_info["fold"] == 1), "game"].values
    fold_23_game = fold_info.loc[np.logical_or(fold_info["fold"] == 2, fold_info["fold"] == 3), "game"].values
    mask_fold_01 = [name in fold_01_game for name in game_names]
    mask_fold_23 = [name in fold_23_game for name in game_names]

    path_fold_01 = list(np.array(end_path)[mask_fold_01]) + list(np.array(side_path)[mask_fold_01])
    path_fold_23 = list(np.array(end_path)[mask_fold_23]) + list(np.array(side_path)[mask_fold_23])

    if VAL23:
        train_path = path_fold_01
        val_path = path_fold_23
    else:
        train_path = path_fold_23
        val_path = path_fold_01

    val_files = load_dataset(val_path[:])[::5]
    print(len(val_files))
    np.random.shuffle(val_files)

    max_pair = max([v['num_labels'] for v in val_files])
    print("maxpairs", max_pair)

    if AUTO_RESIZE:
        batch_size = 1
        tf_dataset = get_tf_dataset_inference_auto_resize(val_files,
                                                          batch_size=batch_size,
                                                          max_box_num=23,
                                                          max_pair_num=max_pair,  # enough large
                                                          )
        model_params = {"input_shape": [None, None, 3],
                        "output_shape": [None, None],
                        "weight_file": load_path,
                        "is_train_model": False,
                        }
    else:
        batch_size = 1
        transforms = [
            #Center_Crop(p=1, min_height=input_shape[0], min_width=input_shape[1]),
            #Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0], target_width=output_shape[1]),
            Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0], target_width=output_shape[1]),

        ]
        transforms = Compose(transforms)
        tf_dataset = get_tf_dataset_inference(val_files,
                                              transforms,
                                              batch_size=batch_size,
                                              max_box_num=23,
                                              max_pair_num=max_pair,  # enough large
                                              )

        model_params = {"input_shape": input_shape,
                        "output_shape": output_shape,
                        "weight_file": load_path,
                        "is_train_model": False,
                        }
    nfl = NFLContact(**model_params)

    """
    inp, targ, info = inference_preprocess(val_files[-1],
                      transforms, # maybe crop
                      max_box_num=25, #
                      max_pair_num=100, # enough large
                      padding=True)
    print(targ["output_contact_label"].numpy(), info["num_labels"])
    """
    start_time = time.time()
    counter = 0
    predicted_labels = []
    gt_labels = []
    ground_labels = []
    for inp, targ, info in tf_dataset:  # .take(1000):
        #mean_size = tf.reduce_mean(inp["input_boxes"][0,...,2] - inp["input_boxes"][0,...,0])
        # plt.imshow(inp["input_rgb"][0,...,:3])
        # plt.title(mean_size.numpy())
        # plt.show()

        preds = nfl.predict(**inp)
        pred_mask, pred_label = preds

        if draw_pred:
            dev = abs(pred_label.numpy()[0] - targ["output_contact_label"].numpy()[0]) * (targ["output_contact_label"].numpy()[0] > 0)
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
                                  idx=argmax_idx)

        for pairs, p, gt, num in zip(inp["input_pairs"].numpy(), pred_label.numpy(), targ["output_contact_label"].numpy(), info["num_labels"]):
            predicted_labels += list(p[:num])
            gt_labels += list(gt[:num])
            ground_labels += list((pairs[:num, 1] == 0))
        counter += batch_size
        time_elapsed = time.time() - start_time
        fps_inference = counter / time_elapsed
        print(f"\r{round(fps_inference, 1)} fps, at {counter} / {len(val_files)}data", end="")
    print(np.sum(gt_labels))
    print(np.sum(predicted_labels))
    print(len(predicted_labels))
    # print(predicted_labels)
    # print(gt_labels)

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

    gt_labels_w_easy_sample = gt_labels + [0] * (len(gt_labels) * 8)
    predicted_labels_w_easy_sample = predicted_labels + [1e-7] * (len(predicted_labels) * 8)

    gt_labels = tf.cast(gt_labels, tf.float32)
    predicted_labels = tf.cast(predicted_labels, tf.float32)
    gt_labels_w_easy_sample = tf.cast(gt_labels_w_easy_sample, tf.float32)
    predicted_labels_w_easy_sample = tf.cast(predicted_labels_w_easy_sample, tf.float32)
    for th in np.linspace(0.1, 0.9, 9):
        print(th, matthews_correlation_fixed(gt_labels, predicted_labels, threshold=th))
        print(th, matthews_correlation_fixed(gt_labels_w_easy_sample, predicted_labels_w_easy_sample, threshold=th))


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
if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_rate', type=float, default=1.0)
    args = parser.parse_args()
    DEBUG = args.debug
    batch_rate = args.batch_rate
    """
    DEBUG = False
    num_epoch = 2 if DEBUG else 20

    setting_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SETTINGS.json")
    DIRS = json.load(open(setting_file))
    BASE_DIR = DIRS["RAW_DATA_DIR"]
    TRAIN_DIR = DIRS["TRAIN_DATA_DIR"]
    WEIGHT_DIR = DIRS["WEIGHT_DIR"]  # model/weights/
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    #DATA_PATH_EXT = os.path.join(BASE_DIR, "images/")
    #ANNOTATINO_PATH_EXT = os.path.join(BASE_DIR, "image_labels.csv")
    DATA_PATH = os.path.join(TRAIN_DIR, "train_img/")
    FOLD_PATH = os.path.join(TRAIN_DIR, "game_fold.csv")
    #TARGET_SIZE = 25

    # normal resolution model
    #FIXED_SIZE_DETECTION = False
    VAL23 = False
    if VAL23:
        name = "fold01"
    else:
        name = "fold23"

    run_train = True
    AUTO_RESIZE = False
    if run_train:
        #CROP_SHAPE=(432, 768, 3)
        FIXED_SIZE_DETECTION = False
        save_path = os.path.join(WEIGHT_DIR, f"ex000_contdet_run024_{name}train_ground_othermask/")
        run_training_main(epochs=num_epoch,
                          batch_size=int(6),  # int(12),
                          # input_shape=(512+64, 896+128, 3),#(384, 640, 3),
                          ##output_shape=(256+32, 448+64),
                          input_shape=(512, 896, 3),  # (384, 640, 3),
                          output_shape=(256, 448),
                          #(192, 320),
                          load_path=None,  # os.path.join(WEIGHT_DIR, "map_model_final_weights.h5"),
                          save_path=save_path)

    else:
        print("RESIZE modeに注意")
        # ex000_contdet_run015_fold01train_fixed_size
        # ex000_contdet_run016_fold01train_not_fixed_size
        run_validation_predict(os.path.join(WEIGHT_DIR, f"ex000_contdet_run021_{name}train_zoom/final_weights.h5"),
                               input_shape=(960, 1728, 3),
                               output_shape=(480, 864),
                               #input_shape=(704, 1280, 3),
                               #output_shape=(352, 640),
                               draw_pred=True)

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
