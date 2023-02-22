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
#from tensorflow.keras.utils import multi_gpu_model
import numpy as np
#import pickle
#from PIL import Image
import pandas as pd
import tensorflow as tf
#import cv2
#import mlflow
import tensorflow_addons as tfa
from kmat.model.model import (build_model, build_model_explicit_distance_shift,
                         map_inference_func_wrapper, map_test,
                         matthews_correlation_fixed)
from kmat.model.model_mappingnet import build_model_map_previous_competition
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (Callback, CSVLogger,
                                        LearningRateScheduler, ModelCheckpoint)
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from kmat.train_utils.scheduler import lrs_wrapper, lrs_wrapper_cos

STACKFRAME = False
if not STACKFRAME:
    from kmat.train_utils.tf_Augmentations_detection import (
        Blur, BrightnessContrast, Center_Crop, Center_Crop_by_box_shape,
        CoarseDropout, Compose, Crop, Crop_by_box_shape, GaussianNoise,
        HorizontalFlip, HueShift, Oneof, PertialBrightnessContrast, Resize,
        Rotation, Shadow, ToGlay, VerticalFlip)

else:  # 5FRAME
    from kmat.train_utils.tf_Augmentations_3inputs import (
        Blur, BrightnessContrast, Center_Crop, Center_Crop_by_box_shape,
        CoarseDropout, Compose, Crop, Crop_by_box_shape, GaussianNoise,
        HorizontalFlip, HueShift, Oneof, PertialBrightnessContrast, Resize,
        Rotation, Shadow, ToGlay, VerticalFlip)
# 5FRAME
if not STACKFRAME:
    from kmat.train_utils.dataloader import (get_tf_dataset,
                                        get_tf_dataset_inference,
                                        get_tf_dataset_inference_auto_resize,
                                        inference_preprocess, load_dataset,
                                        load_dataset_se, load_train_data_all29)
else:
    from kmat.train_utils.dataloader_3inputs import (
        get_tf_dataset, get_tf_dataset_inference,
        get_tf_dataset_inference_auto_resize, inference_preprocess,
        load_dataset)


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

    def __init__(self, mini_dataset, mask_model, num_check=15, num_freq=2):
        super(TrainMonitor, self).__init__()
        self.dataset = mini_dataset
        self.mask_model = mask_model
        self.num_check = num_check
        self.num_freq = num_freq

    def on_epoch_end(self, epoch, logs=None):

        self.mask_model.trainable = False
        if epoch % self.num_freq == 0:
            for inp, targ in self.dataset.take(self.num_check):
                pred_mask_raw, _, pred_label = self.mask_model(inp, training=False)
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
                 is_train_model=False,
                 num_cls=None

                 ):

        print("\rLoading Models...", end="")

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.num_cls = num_cls
        self.is_train_model = is_train_model
        self.weight_file = weight_file
        self.load_model(weight_file, is_train_model)
        self.use_map_model = False
        print("Loading Models......Finish")

    def load_model(self, weight_file=None, is_train_model=False):
        """build model and load weights"""
        #self.map_model = build_model_map_previous_competition(os.path.join(WEIGHT_DIR, "map_model_final_weights_r.h5"))

        builder = build_model_explicit_distance_shift
        self.model, self.sub_model, self.losses, self.loss_weights, self.metrics = builder(self.input_shape,
                                                                                           backbone="effv2s",
                                                                                           minimum_stride=self.input_shape[0] // self.output_shape[0],
                                                                                           max_stride=64,
                                                                                           is_train=self.is_train_model,
                                                                                           num_boxes=None,
                                                                                           size="SM",
                                                                                           #num_cls = self.num_cls,
                                                                                           #feature_ext_weight=os.path.join(WEIGHT_DIR, f"ex000_contdet_run43_{name}train_72crop6cbr_concat_dist/final_weights.h5"),
                                                                                           output_single_contact=True)

        if not weight_file is None:
            if is_train_model:
                self.model.load_weights(weight_file, by_name=True, skip_mismatch=True)
            else:
                self.model.load_weights(weight_file)  # , by_name=True, skip_mismatch=True)
        if not is_train_model:
            self.sub_model.trainable = False
            for layer in self.model.layers:
                layer.trainable = False
                if "efficient" in layer.name:
                    for l in layer.layers:
                        l.trainable = False
            self.names_of_model_inputs = [inp.name for inp in self.sub_model.input]
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
        optim = tfa.optimizers.MovingAverage(optim)
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
            if CROP_SHAPE is not None:
                if not GRAY:
                    transforms_train = [
                        HorizontalFlip(p=0.5),
                        # Rotation(p=0.5, max_angle=10),

                        #Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                        Crop(p=1, min_height=CROP_SHAPE[0], min_width=CROP_SHAPE[1]),
                        Resize(
                            height=self.input_shape[0],
                            width=self.input_shape[1],
                            target_height=self.output_shape[0],
                            target_width=self.output_shape[1]),
                        BrightnessContrast(p=1.0),
                        HueShift(p=0.8, min_offset=-0.25, max_offset=0.25),

                        # Oneof(
                        #      transforms = [PertialBrightnessContrast(p=0.2,
                        #                                              max_holes=3, max_height=80, max_width=80,
                        #                                              min_holes=1, min_height=30, min_width=30,
                        #                                              min_offset=0, max_offset=30,
                        #                                              min_multiply=1.0, max_multiply=1.5),
                        #                    Shadow(p=0.4,
                        #                           max_holes=3, max_height=120, max_width=120,
                        #                           min_holes=1, min_height=50, min_width=50,
                        #                           min_strength=0.2, max_strength=0.8, shadow_color=0.0)
                        #                    ],
                        #      probs=[0.2,0.2]
                        #      ),
                        PertialBrightnessContrast(p=0.2,
                                                  max_holes=3, max_height=80, max_width=80,
                                                  min_holes=1, min_height=30, min_width=30,
                                                  min_offset=0, max_offset=30,
                                                  min_multiply=1.0, max_multiply=1.5),
                        Shadow(p=0.2,
                               max_holes=3, max_height=120, max_width=120,
                               min_holes=1, min_height=50, min_width=50,
                               min_strength=0.2, max_strength=0.8, shadow_color=0.0),


                        # Blur(p=0.1),
                        #GaussianNoise(p=0.1, min_var=10, max_var=40),
                        # ToGlay(p=0.1),
                    ]
                else:
                    print("use Aug for GRAY")
                    transforms_train = [
                        HorizontalFlip(p=0.5),
                        # Rotation(p=0.5, max_angle=10),

                        #Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                        Crop(p=1, min_height=CROP_SHAPE[0], min_width=CROP_SHAPE[1]),
                        Resize(
                            height=self.input_shape[0],
                            width=self.input_shape[1],
                            target_height=self.output_shape[0],
                            target_width=self.output_shape[1]),
                        BrightnessContrast(p=1.0),

                        PertialBrightnessContrast(p=0.2,
                                                  max_holes=3, max_height=80, max_width=80,
                                                  min_holes=1, min_height=30, min_width=30,
                                                  min_offset=0, max_offset=30,
                                                  min_multiply=1.0, max_multiply=1.5),
                        Shadow(p=0.2,
                               max_holes=3, max_height=120, max_width=120,
                               min_holes=1, min_height=50, min_width=50,
                               min_strength=0.2, max_strength=0.8, shadow_color=0.0),


                    ]
            else:
                if not GRAY:
                    transforms_train = [
                        HorizontalFlip(p=0.5),
                        # Rotation(p=0.5, max_angle=10),

                        Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                        BrightnessContrast(p=1.0),
                        HueShift(p=0.8, min_offset=-0.25, max_offset=0.25),
                        PertialBrightnessContrast(p=0.2,
                                                  max_holes=3, max_height=80, max_width=80,
                                                  min_holes=1, min_height=30, min_width=30,
                                                  min_offset=0, max_offset=30,
                                                  min_multiply=1.0, max_multiply=1.5),
                        Shadow(p=0.2,
                               max_holes=3, max_height=120, max_width=120,
                               min_holes=1, min_height=50, min_width=50,
                               min_strength=0.2, max_strength=0.8, shadow_color=0.0),


                        # Blur(p=0.1),
                        #GaussianNoise(p=0.1, min_var=10, max_var=40),
                        # ToGlay(p=0.1),
                    ]
                else:
                    transforms_train = [
                        HorizontalFlip(p=0.5),
                        # Rotation(p=0.5, max_angle=10),

                        Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                        BrightnessContrast(p=1.0),
                        PertialBrightnessContrast(p=0.2,
                                                  max_holes=3, max_height=80, max_width=80,
                                                  min_holes=1, min_height=30, min_width=30,
                                                  min_offset=0, max_offset=30,
                                                  min_multiply=1.0, max_multiply=1.5),
                        Shadow(p=0.2,
                               max_holes=3, max_height=120, max_width=120,
                               min_holes=1, min_height=50, min_width=50,
                               min_strength=0.2, max_strength=0.8, shadow_color=0.0),
                    ]

            if CROP_SHAPE is not None:
                transforms_val = [
                    # Center_Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                    Center_Crop(p=1, min_height=CROP_SHAPE[0], min_width=CROP_SHAPE[1]),
                    Resize(
                        height=self.input_shape[0],
                        width=self.input_shape[1],
                        target_height=self.output_shape[0],
                        target_width=self.output_shape[1]),
                ]
            else:
                transforms_val = [Center_Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                                  ]

        train_transforms = Compose(transforms_train)
        val_transforms = Compose(transforms_val)
        # """
        # test run
        """
        tfd = get_tf_dataset(train_dataset[2::],
                             self.input_shape,
                             self.output_shape,
                       batch_size=1,
                       transforms=train_transforms,
                       is_train=True,
                       max_pair_num=120,
                       num_cls=self.num_cls,
                       )

        #tfd = get_tf_dataset_se(train_dataset[0][2::],
        #                        train_dataset[1][2::],
        #                     self.input_shape,
        #                     self.output_shape,
        #               batch_size=1,
        #               transforms=train_transforms,
        #               is_train=True,
        #               max_pair_num=120,)

        for inp, targ in tfd.take(10):
            #print(inp["num_player"])
            #print(targ["num_player"])
            #input_rgb = Input(input_shape, name="input_rgb")#256,256,3
            #input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
            #input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
            #input_player_positions = Input(shape=[num_boxes,2], name="input_player_positions")
            for k, val in inp.items():
                print(k, val.shape)
            #print(inp["input_boxes"])
            plt.imshow(inp["input_rgb"][0,...,:3].numpy())
            plt.show()
            #plt.imshow(inp["input_rgb"][0,...,3:].numpy())
            #plt.show()
            #print(tf.reduce_max(inp["input_rgb"][0,...,3:]))
            continue

            #plt.imshow(inp["input_rgb"][1].numpy())
            #plt.show()

            #print(inp["input_player_positions_s"][0])
            #print(inp["input_player_positions_e"][0])
            #print(inp["input_player_in_other_video_s"][0])
            #print(inp["input_player_in_other_video_e"][0])



            preds = self.model(inp)


            #p, maps = map_test(self.map_model, inp["input_rgb"], inp["input_boxes"])

            plt.imshow(inp["input_rgb"][0,...,:3])
            plt.show()
            for i in range(5):
                print(i)
                print(inp["input_pairs"][0,i])
                a,b = inp["input_pairs"][0,i].numpy()
                print(inp["input_boxes"][0,a-1])
                if b!=0:
                    print(inp["input_boxes"][0,b-1])
                plt.imshow(preds[0][0,i,:,:,0])
                plt.show()

            #plt.imshow(maps[0,...,:1])
            #plt.show()
            #plt.imshow(maps[0,...,1:2])
            #plt.show()
            #print(tf.reduce_min(maps[0,...,:2], axis=[0,1]))
            plt.scatter(x=p[0,:,0], y=p[0,:,1])
            plt.xlim(-0.7,0.7)
            plt.ylim(-0.7,0.7)
            plt.grid()
            plt.show()

            plt.imshow(inp["input_rgb"][0,...,:3])
            plt.show()
            depth = self.model(inp)[-1][0]
            plt.imshow(depth)
            plt.show()
            print(depth.numpy().min(), depth.numpy().max())
            continue

            outs = self.model(inp)

            #print("run model")
            #print(self.model(inp).shape)
            print(tf.math.reduce_std(inp["input_rgb"][...,3:], axis=[0,1,2]))
            print(tf.reduce_max(inp["input_rgb"][...,3:], axis=[0,1,2]))
            print(tf.reduce_min(inp["input_rgb"][...,3:], axis=[0,1,2]))
            mean_size = tf.reduce_mean(inp["input_boxes"][0,...,2] - inp["input_boxes"][0,...,0])
            plt.imshow(inp["input_rgb"][0,...,:3])
            #plt.title(mean_size.numpy())
            plt.show()
            print(tf.reduce_sum(tf.cast(targ["output_contact_label"][0]>-1, tf.float32)), "HAVELABELL")
            plt.imshow(inp["input_rgb"][0,...,3:6])
            plt.show()
            plt.imshow(inp["input_rgb"][0,...,6:9])
            plt.show()

            #print(tf.reduce_max(inp["input_rgb"][0,...,3:4]))
            #print(tf.reduce_max(inp["input_rgb"][0,...,4:5]))
            #print(tf.reduce_max(inp["input_rgb"][0,...,5:6]))
        raise Exception
        #"""
        print()

        monitor_dataset = get_tf_dataset(val_dataset[0:100],
                                         self.input_shape,
                                         self.output_shape,
                                         batch_size=1,
                                         transforms=val_transforms,
                                         is_train=False,
                                         )

        monitor = TrainMonitor(monitor_dataset, self.sub_model)

        print("step per epoch", num_data[0] // batch_size, num_data[1] // batch_size)

        self.hist = self.model.fit(get_tf_dataset(train_dataset,
                                                  self.input_shape,
                                                  self.output_shape,
                                                  batch_size=batch_size,
                                                  transforms=train_transforms,
                                                  is_train=True,
                                                  max_pair_num=50,
                                                  # use_cut_mix=False,
                                                  #num_cls = self.num_cls,
                                                  ),
                                   steps_per_epoch=num_data[0] // batch_size,
                                   epochs=n_epoch,
                                   validation_data=get_tf_dataset(val_dataset,
                                                                  self.input_shape,
                                                                  self.output_shape,
                                                                  batch_size=batch_size,
                                                                  transforms=val_transforms,
                                                                  max_pair_num=50,
                                                                  is_train=False,
                                                                  #num_cls = self.num_cls,
                                                                  ),
                                   validation_steps=num_data[1] // batch_size,
                                   #callbacks=[lr_schedule, logger, cp_callback, monitor] if not DEBUGMODE else [lr_schedule, logger, cp_callback],
                                   callbacks=[lr_schedule, logger, cp_callback],
                                   )

        print("Saving weights and results...")
        self.model.save_weights(save_dir + "final_weights.h5")
        csv_hist = save_dir + "hist.csv"
        pd.DataFrame(self.hist.history).to_csv(csv_hist, index=False)
        print("Done")

    def combine_map_model(self, weight_file):
        map_model = build_model_map_previous_competition(weight_file)
        self.tf_map_model = map_inference_func_wrapper(map_model, include_resize=True)
        self.use_map_model = True

    def predict(self, inputs_dict,
                #input_rgb, input_boxes, input_player_positions, input_pairs,
                ):
        """
        inputs = [input_rgb, input_boxes, (input_player_positions), input_pairs]
        """
        if self.use_map_model and "input_player_positions" not in inputs_dict.keys():
            player_pos, map_xy = self.tf_map_model(inputs_dict["input_rgb"], inputs_dict["input_boxes"])
            inputs_dict["input_player_positions"] = player_pos
        preds = self.tf_model([inputs_dict[k] for k in self.names_of_model_inputs])
        return preds, inputs_dict


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

    end_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Endzone")))
    side_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Sideline")))
    game_names = [int(os.path.basename(p).split("_", 1)[0]) for p in end_path]
    print("ALL", len(end_path), len(side_path))

    fold_info = pd.read_csv(FOLD_PATH)

    fold_train_game = fold_info.loc[np.any([fold_info["fold"] == i for i in TRAIN_FOLD], axis=0), "game"].values
    fold_val_game = fold_info.loc[np.any([fold_info["fold"] == i for i in VAL_FOLD], axis=0), "game"].values
    mask_fold_train = [name in fold_train_game for name in game_names]
    mask_fold_val = [name in fold_val_game for name in game_names]

    train_path = list(np.array(end_path)[mask_fold_train]) + list(np.array(side_path)[mask_fold_train])
    val_path = list(np.array(end_path)[mask_fold_val]) + list(np.array(side_path)[mask_fold_val])

    print("train, val", len(train_path), len(val_path))

    np.random.shuffle(train_path)
    np.random.shuffle(val_path)

    if DEBUGMODE:
        train_path = train_path[:4]
        val_path = val_path[:4]
    if CENTER_OF_STEP:
        train_files = load_dataset(train_path)
        val_files = load_dataset(val_path)[::5]

        step_rate = 3
        step_rate_val = 1
    else:
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
                    # "num_cls": num_cls,
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


def run_training_all29(epochs=20,
                       batch_size=4,
                       input_shape=(448, 768, 3),
                       output_shape=(224, 384),
                       load_path="",
                       save_path="",
                       train_all=False):

    K.clear_session()
    set_seeds(111)

    end_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Endzone")))
    game_play_names = [os.path.basename(p).rsplit("_", 1)[0] for p in end_path]
    game_names = [int(os.path.basename(p).split("_", 1)[0]) for p in end_path]
    print("ALL", len(end_path))

    fold_info = pd.read_csv(FOLD_PATH)
    fold_train_game = fold_info.loc[np.any([fold_info["fold"] == i for i in TRAIN_FOLD], axis=0), "game"].values
    fold_val_game = fold_info.loc[np.any([fold_info["fold"] == i for i in VAL_FOLD], axis=0), "game"].values
    mask_fold_train = [name in fold_train_game for name in game_names]
    mask_fold_val = [name in fold_val_game for name in game_names]

    train_game_play = list(np.array(game_play_names)[mask_fold_train])
    val_game_play = list(np.array(game_play_names)[mask_fold_val])

    # 一時的に無視する。
    low_res_videos = ["58536_002108", "58553_001995", "58553_003801"]
    train_game_play = [gp for gp in train_game_play if gp not in low_res_videos]
    val_game_play = [gp for gp in val_game_play if gp not in low_res_videos]

    print("train, val", len(train_game_play), len(val_game_play))

    #shuffle_indices = np.arange(len(end_path))
    np.random.shuffle(train_game_play)
    np.random.shuffle(val_game_play)

    if DEBUGMODE:
        train_game_play = train_game_play[:4]
        val_game_play = val_game_play[:4]

    df_all29_train = pd.read_csv("input_preprocess/All29_assignment/df_all29_train.csv")
    train_files = load_train_data_all29(df_all29_train, train_game_play, img_dir="input_preprocess/All29_assignment")
    val_files = load_train_data_all29(df_all29_train, val_game_play, img_dir="input_preprocess/All29_assignment")

    val_files = val_files[::3]

    step_rate = 3
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
                    # "num_cls": num_cls,
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
                           draw_pred=False,
                           crop_shape=None,
                           output_path=None):

    from_map_model = True

    K.clear_session()
    set_seeds(111)

    end_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Endzone")))
    side_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Sideline")))
    game_names = [int(os.path.basename(p).split("_", 1)[0]) for p in end_path]
    print("ALL", len(end_path), len(side_path))

    fold_info = pd.read_csv(FOLD_PATH)

    """
    fold_01_game = fold_info.loc[np.logical_or(fold_info["fold"]==0, fold_info["fold"]==1), "game"].values
    fold_23_game = fold_info.loc[np.logical_or(fold_info["fold"]==2, fold_info["fold"]==3), "game"].values
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
    """
    # fold_train_game = fold_info.loc[np.any([fold_info["fold"]==i for i in TRAIN_FOLD], axis=0), "game"].values
    fold_val_game = fold_info.loc[np.any([fold_info["fold"] == i for i in INF_FOLD], axis=0), "game"].values
    # mask_fold_train = [name in fold_train_game for name in game_names]
    mask_fold_val = [name in fold_val_game for name in game_names]

    # train_path = list(np.array(end_path)[mask_fold_train]) + list(np.array(side_path)[mask_fold_train])
    val_path_end = list(np.array(end_path)[mask_fold_val])
    val_path_side = list(np.array(side_path)[mask_fold_val])

    val_files_end = load_dataset(val_path_end[:])[:]
    val_files_side = load_dataset(val_path_side[:])[:]
    # print(len(val_files))
    # np.random.shuffle(val_files)

    max_pair = max([max([v['num_labels'] for v in val_files_end]), max([v['num_labels'] for v in val_files_side])])
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
        batch_size = 8
        if crop_shape is not None:
            transforms = [
                Center_Crop(p=1, min_height=crop_shape[0], min_width=crop_shape[1]),
                Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0] // 2, target_width=output_shape[1] // 2),
                #Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0], target_width=output_shape[1]),

            ]

        else:
            transforms = [
                Center_Crop(p=1, min_height=input_shape[0], min_width=input_shape[1]),
                #Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0], target_width=output_shape[1]),
                #Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0], target_width=output_shape[1]),

            ]
        transforms = Compose(transforms)
        tf_dataset_side = get_tf_dataset_inference(val_files_side,
                                                   transforms,
                                                   batch_size=batch_size,
                                                   max_box_num=23,
                                                   max_pair_num=max_pair,  # enough large
                                                   )
        tf_dataset_end = get_tf_dataset_inference(val_files_end,
                                                  transforms,
                                                  batch_size=batch_size,
                                                  max_box_num=23,
                                                  max_pair_num=max_pair,  # enough large
                                                  )
        tf_dataset = {"Endzone": tf_dataset_end, "Sideline": tf_dataset_side}
        # tf_dataset = get_tf_dataset(val_files,
        #                     input_shape,
        #                     output_shape,
        #               batch_size=1,
        #               transforms=transforms,
        #               is_train=False,
        #               max_pair_num=max_pair)

        model_params = {"input_shape": input_shape,
                        "output_shape": output_shape,
                        "weight_file": load_path,
                        "is_train_model": False,
                        }

    nfl = NFLContact(**model_params)

    if from_map_model:
        nfl.combine_map_model(os.path.join(WEIGHT_DIR, "map_model_final_weights_r.h5"))

    """
    inp, targ, info = inference_preprocess(val_files[-1],
                      transforms, # maybe crop
                      max_box_num=25, #
                      max_pair_num=100, # enough large
                      padding=True)
    print(targ["output_contact_label"].numpy(), info["num_labels"])
    """
    for view in ["Endzone", "Sideline"]:
        print("START PREDICT ", view)

        start_time = time.time()
        counter = 0
        predicted_labels = []
        gt_labels = []
        ground_labels = []
        players_1 = []
        players_2 = []
        frame_numbers = []
        game_plays = []

        save_title = "fold"
        for fold_num in INF_FOLD:
            save_title += str(fold_num)
        save_title = save_title + "_cnn_pred_side.csv" if view == "Sideline" else save_title + "_cnn_pred_end.csv"

        for inp, targ, info in tf_dataset[view]:  # .take(1000):

            preds, inp = nfl.predict(inp)
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

            # for pairs, p, gt, num in zip(inp["input_pairs"].numpy(), pred_label.numpy(),
            #                             targ["output_contact_label"].numpy(),
            #                             info["num_labels"]):
            for i, [p, gt, num, pairs, f] in enumerate(zip(pred_label.numpy(), targ["output_contact_label"].numpy(),
                                                           info["num_labels"].numpy(), info["contact_pairlabels"], info["file"])):
                predicted_labels += list(p[:num])
                gt_labels += list(gt[:num])
                players_1 += list(pairs[:num, 0].numpy())
                players_2 += list(pairs[:num, 1].numpy())

                # predicted_labels += list(p[:num])
                # gt_labels += list(gt[:num])
                ground_labels += list((pairs[:num, 1] == 0))
                file_decoded = f.numpy().decode('ascii')
                frame_no = int(os.path.splitext(os.path.basename(file_decoded))[0])
                frame_numbers += [frame_no] * num
                game_play_view = os.path.basename(os.path.dirname(file_decoded))
                game_play, _ = game_play_view.rsplit("_", 1)
                game_plays += [game_play] * num

            counter += batch_size
            time_elapsed = time.time() - start_time
            fps_inference = counter / time_elapsed
            print(f"\r{round(fps_inference, 1)} fps, at {counter} / {len(val_files_side)}data", end="")

        df_pred = pd.DataFrame(np.array(predicted_labels).reshape(-1, 1), columns=[f"cnn_pred_{view}"])
        df_pred["nfl_player_id_1"] = players_1
        df_pred["nfl_player_id_2"] = players_2
        df_pred["game_play"] = game_plays
        df_pred["frame"] = frame_numbers
        df_pred["gt_tmp"] = gt_labels
        print(df_pred.head())
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            df_pred.to_csv(os.path.join(output_path, save_title), index=False)

        print(inp["input_pairs"].numpy().shape, pred_label.numpy().shape, targ["output_contact_label"].numpy().shape)
        print(np.sum(gt_labels))
        print(np.sum(predicted_labels))
        print(len(predicted_labels))

        """
        a = pd.read_csv("output/TMP/fold23_cnn_pred_side.csv")
        b = pd.read_csv("output/TMP/fold23_cnn_pred_side_d.csv")

        gt_labels = a["gt_tmp"].values
        predicted_labels = (a["cnn_pred_Sideline"].values + b["cnn_pred_Sideline"].values) / 2
        ground_labels = (a["nfl_player_id_2"] == 0).values
        """

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
        gt_labels_w_easy_sample = list(gt_labels) + [0] * (len(gt_labels) * 8)
        predicted_labels_w_easy_sample = list(predicted_labels) + [1e-7] * (len(predicted_labels) * 8)

        gt_labels = tf.cast(gt_labels, tf.float32)
        predicted_labels = tf.cast(predicted_labels, tf.float32)
        gt_labels_w_easy_sample = tf.cast(gt_labels_w_easy_sample, tf.float32)
        predicted_labels_w_easy_sample = tf.cast(predicted_labels_w_easy_sample, tf.float32)
        for th in np.linspace(0.1, 0.9, 9):
            print(th, matthews_correlation_fixed(gt_labels, predicted_labels, threshold=th))
            print(th, matthews_correlation_fixed(gt_labels_w_easy_sample, predicted_labels_w_easy_sample, threshold=th))


def run_validation_predict_all29(load_path,
                                 input_shape=(704, 1280, 3),
                                 output_shape=(352, 640),
                                 draw_pred=False,
                                 crop_shape=None,
                                 output_path=None):

    from_map_model = True

    K.clear_session()
    set_seeds(111)

    end_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Endzone")))
    game_play_names = [os.path.basename(p).rsplit("_", 1)[0] for p in end_path]
    game_names = [int(os.path.basename(p).split("_", 1)[0]) for p in end_path]
    print("ALL", len(end_path))

    fold_info = pd.read_csv(FOLD_PATH)
    # fold_train_game = fold_info.loc[np.any([fold_info["fold"]==i for i in TRAIN_FOLD], axis=0), "game"].values
    fold_val_game = fold_info.loc[np.any([fold_info["fold"] == i for i in VAL_FOLD], axis=0), "game"].values
    # mask_fold_train = [name in fold_train_game for name in game_names]
    mask_fold_val = [name in fold_val_game for name in game_names]

    # train_game_play = list(np.array(game_play_names)[mask_fold_train])
    val_game_play = list(np.array(game_play_names)[mask_fold_val])

    # 一時的に無視する。
    low_res_videos = ["58536_002108", "58553_001995", "58553_003801"]
    # train_game_play = [gp for gp in train_game_play if gp not in low_res_videos]
    val_game_play = [gp for gp in val_game_play if gp not in low_res_videos]

    df_all29_train = pd.read_csv("input_preprocess/All29_assignment/df_all29_train.csv")
    val_files = load_train_data_all29(df_all29_train, val_game_play, img_dir="input_preprocess/All29_assignment")

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
        batch_size = 8
        if crop_shape is not None:
            transforms = [
                Center_Crop(p=1, min_height=crop_shape[0], min_width=crop_shape[1]),
                Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0] // 2, target_width=output_shape[1] // 2),
                #Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0], target_width=output_shape[1]),

            ]

        else:
            transforms = [
                Center_Crop(p=1, min_height=input_shape[0], min_width=input_shape[1]),
                #Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0], target_width=output_shape[1]),
                #Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0], target_width=output_shape[1]),

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

    if from_map_model:
        nfl.combine_map_model(os.path.join(WEIGHT_DIR, "map_model_final_weights_r.h5"))

    view = "All29"
    print("START PREDICT ", view)

    start_time = time.time()
    counter = 0
    predicted_labels = []
    gt_labels = []
    ground_labels = []
    players_1 = []
    players_2 = []
    frame_numbers = []
    game_plays = []

    save_title = "fold"
    for fold_num in INF_FOLD:
        save_title += str(fold_num)
    save_title = save_title + "_cnn_pred_all29.csv"

    for inp, targ, info in tf_dataset:  # .take(1000):

        preds, inp = nfl.predict(inp)
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

        # for pairs, p, gt, num in zip(inp["input_pairs"].numpy(), pred_label.numpy(),
        #                             targ["output_contact_label"].numpy(),
        #                             info["num_labels"]):
        for i, [p, gt, num, pairs, f] in enumerate(zip(pred_label.numpy(), targ["output_contact_label"].numpy(),
                                                       info["num_labels"].numpy(), info["contact_pairlabels"], info["file"])):
            predicted_labels += list(p[:num])
            gt_labels += list(gt[:num])
            players_1 += list(pairs[:num, 0].numpy())
            players_2 += list(pairs[:num, 1].numpy())

            # predicted_labels += list(p[:num])
            # gt_labels += list(gt[:num])
            ground_labels += list((pairs[:num, 1] == 0))
            file_decoded = f.numpy().decode('ascii')
            frame_no = int(os.path.splitext(os.path.basename(file_decoded))[0])
            frame_numbers += [frame_no] * num
            game_play_view = os.path.basename(os.path.dirname(file_decoded))
            game_play, _ = game_play_view.rsplit("_", 1)
            game_plays += [game_play] * num

        counter += batch_size
        time_elapsed = time.time() - start_time
        fps_inference = counter / time_elapsed
        print(f"\r{round(fps_inference, 1)} fps, at {counter} / {len(val_files)}data", end="")

    df_pred = pd.DataFrame(np.array(predicted_labels).reshape(-1, 1), columns=[f"cnn_pred_{view}"])
    df_pred["nfl_player_id_1"] = players_1
    df_pred["nfl_player_id_2"] = players_2
    df_pred["game_play"] = game_plays
    df_pred["frame"] = frame_numbers
    df_pred["gt_tmp"] = gt_labels
    print(df_pred.head())
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        df_pred.to_csv(os.path.join(output_path, save_title), index=False)

    print(inp["input_pairs"].numpy().shape, pred_label.numpy().shape, targ["output_contact_label"].numpy().shape)
    print(np.sum(gt_labels))
    print(np.sum(predicted_labels))
    print(len(predicted_labels))

    """
    a = pd.read_csv("output/TMP/fold23_cnn_pred_side.csv")
    b = pd.read_csv("output/TMP/fold23_cnn_pred_side_d.csv")

    gt_labels = a["gt_tmp"].values
    predicted_labels = (a["cnn_pred_Sideline"].values + b["cnn_pred_Sideline"].values) / 2
    ground_labels = (a["nfl_player_id_2"] == 0).values
    """

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
    gt_labels_w_easy_sample = list(gt_labels) + [0] * (len(gt_labels) * 8)
    predicted_labels_w_easy_sample = list(predicted_labels) + [1e-7] * (len(predicted_labels) * 8)

    gt_labels = tf.cast(gt_labels, tf.float32)
    predicted_labels = tf.cast(predicted_labels, tf.float32)
    gt_labels_w_easy_sample = tf.cast(gt_labels_w_easy_sample, tf.float32)
    predicted_labels_w_easy_sample = tf.cast(predicted_labels_w_easy_sample, tf.float32)
    for th in np.linspace(0.1, 0.9, 9):
        print(th, matthews_correlation_fixed(gt_labels, predicted_labels, threshold=th))
        print(th, matthews_correlation_fixed(gt_labels_w_easy_sample, predicted_labels_w_easy_sample, threshold=th))


def prepare_mapping_data():
    input_shape = (512, 896, 3)
    output_shape = (128, 224)
    map_model = build_model_map_previous_competition(os.path.join(WEIGHT_DIR, "map_model_final_weights_r.h5"))

    tf_map_model = map_inference_func_wrapper(map_model)

    K.clear_session()
    set_seeds(111)

    end_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Endzone")))
    side_path = sorted(glob.glob(os.path.join(DATA_PATH, "*Sideline")))
    game_names = [int(os.path.basename(p).split("_", 1)[0]) for p in end_path]
    print("ALL", len(end_path), len(side_path))
    all_path = end_path[:] + side_path[:]
    all_data = load_dataset(all_path)
    print(all_path)
    batch_size = 16
    transforms = [Resize(height=input_shape[0], width=input_shape[1], target_height=output_shape[0], target_width=output_shape[1]),
                  ]
    transforms = Compose(transforms)
    tf_dataset = get_tf_dataset_inference(all_data,
                                          transforms,
                                          batch_size=batch_size,
                                          max_box_num=23,
                                          max_pair_num=20,  # not use
                                          padding=True)

    filenames_label = [data["file"].replace(".jpg", "_label.json") for data in all_data]
    filenames_save = [data["file"].replace(".jpg", "_pos.npy") for data in all_data]
    start_time = time.time()
    counter = 0
    predicted_positions = []
    gt_labels = []
    ground_labels = []
    draw_pred = True
    print(len(filenames_save), filenames_save[0], filenames_save[-1])
    for i, [inp, targ, info] in enumerate(tf_dataset):  # .take(1000):
        #mean_size = tf.reduce_mean(inp["input_boxes"][0,...,2] - inp["input_boxes"][0,...,0])
        # plt.imshow(inp["input_rgb"][0,...,:3])
        # plt.title(mean_size.numpy())
        # plt.show()

        preds = tf_map_model(inp["input_rgb"], inp["input_boxes"])
        player_pos, map_xy = preds

        if draw_pred and i % 10 == 0:
            plt.imshow(inp["input_rgb"][0, ..., :3])
            plt.show()
            plt.scatter(x=player_pos[0, :, 0], y=player_pos[0, :, 1])
            plt.xlim(-0.7, 0.7)
            plt.ylim(-0.7, 0.7)
            plt.grid()
            plt.show()

        for positions, num in zip(player_pos.numpy(), info["num_player"]):
            predicted_positions += [positions[:num]]
            #gt_labels += list(gt[:num])
            #ground_labels += list((pairs[:num,1]==0))

        counter += batch_size
        time_elapsed = time.time() - start_time
        fps_inference = counter / time_elapsed
        print(f"\r{round(fps_inference, 1)} fps, at {counter} / {len(all_data)}data", end="")
    print(len(filenames_save), len(predicted_positions))

    for save_npy, preds in zip(filenames_save, predicted_positions):
        np.save(save_npy, preds.astype(np.float32))


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_rate', type=float, default=1.0)
    args = parser.parse_args()
    DEBUG = args.debug
    batch_rate = args.batch_rate
    """

    # TODO
    # 3D構成する
    # side end　別モデル化

    DEBUGMODE = False  # test by small data 505044

    DEBUG = False
    num_epoch = 2 if DEBUG else 20

    setting_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SETTINGS.json")
    DIRS = json.load(open(setting_file))
    BASE_DIR = DIRS["RAW_DATA_DIR"]
    TRAIN_DIR = DIRS["TRAIN_DATA_DIR"]
    WEIGHT_DIR = DIRS["WEIGHT_DIR"]  # model/weights/
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    CENTER_OF_STEP = False
    GRAY = False
    if CENTER_OF_STEP:
        #
        DATA_PATH = os.path.join(TRAIN_DIR, "train_img_10fps/")
        if GRAY:
            DATA_PATH = os.path.join(TRAIN_DIR, "train_img_gray/")
    else:
        DATA_PATH = os.path.join(TRAIN_DIR, "train_img/")

    FOLD_PATH = os.path.join(TRAIN_DIR, "game_fold.csv")

    run_train = True
    run_validation = True
    AUTO_RESIZE = False

    # prepare_mapping_data()# つかう１
    #raise Exception()

    if run_train:
        for run_no in range(4):

            """
            if run_no==0:
                TRAIN_FOLD = [0,1]
                VAL_FOLD = [2]
            elif run_no==1:
                TRAIN_FOLD = [0,2]
                VAL_FOLD = [1]
            elif run_no==2:
                TRAIN_FOLD = [1,2]
                VAL_FOLD = [0]

            INF_FOLD = VAL_FOLD + [3]
            """

            if run_no == 0:
                TRAIN_FOLD = [0, 1, 2]
                VAL_FOLD = [3]
            elif run_no == 1:
                TRAIN_FOLD = [0, 1, 3]
                VAL_FOLD = [2]
            elif run_no == 2:
                TRAIN_FOLD = [0, 2, 3]
                VAL_FOLD = [1]
            elif run_no == 3:
                TRAIN_FOLD = [1, 2, 3]
                VAL_FOLD = [0]

            INF_FOLD = VAL_FOLD

            # VAL23 = True
            name = "fold"
            for fold_num in TRAIN_FOLD:
                name += str(fold_num)
            print("train fold: ", name)
            if STACKFRAME:
                suffix = "_multiframe"
            else:
                suffix = ""

            print(name, "train")

            FIXED_SIZE_DETECTION = False
            CROP_SHAPE = None  # (384, 640, 3)

            save_path = os.path.join(WEIGHT_DIR, f"ex000_contdet_run070_{name}train_72crop6cbr_sc_mappretrain{suffix}/")
            run_training_main(epochs=num_epoch // 2 if STACKFRAME else num_epoch,
                              batch_size=int(6),  # int(12),
                              input_shape=(512, 896, 3),  # (384, 640, 3),
                              output_shape=(256, 448),
                              load_path=os.path.join(WEIGHT_DIR, "map_model_final_weights.h5"),
                              # load_path=os.path.join(WEIGHT_DIR, "det_model_final_weights.h5"),

                              save_path=save_path)

    if run_validation:
        CENTER_OF_STEP = False
        DATA_PATH = os.path.join(TRAIN_DIR, "train_img_10fps/")

        for run_no in range(4):

            """
            if run_no==0:
                TRAIN_FOLD = [0,1]
                VAL_FOLD = [2]
            elif run_no==1:
                TRAIN_FOLD = [0,2]
                VAL_FOLD = [1]
            elif run_no==2:
                TRAIN_FOLD = [1,2]
                VAL_FOLD = [0]
            INF_FOLD = VAL_FOLD + [3]
            """

            if run_no == 0:
                TRAIN_FOLD = [0, 1, 2]
                VAL_FOLD = [3]
            elif run_no == 1:
                TRAIN_FOLD = [0, 1, 3]
                VAL_FOLD = [2]
            elif run_no == 2:
                TRAIN_FOLD = [0, 2, 3]
                VAL_FOLD = [1]
            elif run_no == 3:
                TRAIN_FOLD = [1, 2, 3]
                VAL_FOLD = [0]

            INF_FOLD = VAL_FOLD

            # VAL23 = True
            name = "fold"
            for fold_num in TRAIN_FOLD:
                name += str(fold_num)
            print("train fold: ", name)
            if STACKFRAME:
                suffix = "_multiframe"
            else:
                suffix = ""

            print(name, "train")

            run_validation_predict(os.path.join(WEIGHT_DIR, f"ex000_contdet_run070_{name}train_72crop6cbr_sc_mappretrain/final_weights.h5"),
                                   #os.path.join(WEIGHT_DIR, f"ex000_contdet_run061_{name}train_72crop6cbr_detpretrain/final_weights.h5"),
                                   #input_shape=(960, 1728, 3),
                                   #output_shape=(480, 864),
                                   input_shape=(704, 1280, 3),
                                   output_shape=(352, 640),
                                   draw_pred=False,
                                   output_path="output/fold3_holdout_cnnpred_detpretrained/")
