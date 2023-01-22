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
from kmat.model.model import (build_model, build_model_explicit_distance,
                         build_model_explicit_distance_feature_dense,
                         build_model_explicit_distance_multi_view,
                         build_model_explicit_distance_multi_view_shift,
                         build_model_explicit_distance_shift,
                         build_model_multi,
                         build_model_multiframe_explicit_distance,
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
                                        get_tf_dataset_se,
                                        inference_preprocess, load_dataset,
                                        load_dataset_se)
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
                 is_train_model=False):

        print("\rLoading Models...", end="")

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.is_train_model = is_train_model
        self.weight_file = weight_file
        self.load_model(weight_file, is_train_model)
        self.use_map_model = False
        print("Loading Models......Finish")

    def load_model(self, weight_file=None, is_train_model=False):
        """build model and load weights"""
        #self.map_model = build_model_map_previous_competition(os.path.join(WEIGHT_DIR, "map_model_final_weights_r.h5"))
        """
        self.model, self.sub_model, self.losses, self.loss_weights, self.metrics = build_model_explicit_distance(self.input_shape,
                                                                          minimum_stride=self.input_shape[0]//self.output_shape[0],
                                                                          is_train=self.is_train_model,
                                                                          backbone="effv2s",
                                                                          from_scratch=False,
                                                                          size="SM",
                                                                          #map_model=self.map_model,
                                                                          )
        """

        """
        # 5FRAME
        self.model, self.sub_model, self.losses, self.loss_weights, self.metrics = build_model_multiframe_explicit_distance(self.input_shape,
                     backbone="effv2s",
                     minimum_stride=self.input_shape[0]//self.output_shape[0],
                     max_stride = 64,
                     is_train=self.is_train_model,
                     num_boxes = None,
                     size="SM",
                     feature_ext_weight=os.path.join(WEIGHT_DIR, f"ex000_contdet_run43_{name}train_72crop6cbr_concat_dist/final_weights.h5"),
                     )
        """
        # 出力前の形検討
        builder = build_model_explicit_distance_shift
        #builder = build_model_explicit_distance_multi_view_shift
        # builder = build_model_explicit_distance_multi_view
        # builder = build_model_explicit_distance_feature_dense
        self.model, self.sub_model, self.losses, self.loss_weights, self.metrics = builder(self.input_shape,
                                                                                           backbone="effv2s",
                                                                                           minimum_stride=self.input_shape[0] // self.output_shape[0],
                                                                                           max_stride=64,
                                                                                           is_train=self.is_train_model,
                                                                                           num_boxes=None,
                                                                                           size="SM",
                                                                                           #feature_ext_weight=os.path.join(WEIGHT_DIR, f"ex000_contdet_run43_{name}train_72crop6cbr_concat_dist/final_weights.h5"),
                                                                                           )

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
                       max_pair_num=120,)

        tfd = get_tf_dataset_se(train_dataset[0][2::],
                                train_dataset[1][2::],
                             self.input_shape,
                             self.output_shape,
                       batch_size=1,
                       transforms=train_transforms,
                       is_train=True,
                       max_pair_num=120,)

        for inp, targ in tfd.take(10):
            #print(inp["num_player"])
            #print(targ["num_player"])
            #input_rgb = Input(input_shape, name="input_rgb")#256,256,3
            #input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
            #input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
            #input_player_positions = Input(shape=[num_boxes,2], name="input_player_positions")
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            for k, val in inp.items():
                print(k, val.shape)
            #print(inp["input_boxes"])
            #plt.imshow(inp["input_rgb"][0].numpy())
            #plt.show()

            #plt.imshow(inp["input_rgb"][1].numpy())
            #plt.show()

            #print(inp["input_player_positions_s"][0])
            #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            #print(inp["input_player_positions_e"][0])
            #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            #print(inp["input_player_in_other_video_s"][0])
            #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
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
        """
        monitor_dataset = get_tf_dataset_se(val_dataset[0][0:100],
                                            val_dataset[1][0:100],
                                         self.input_shape,
                                         self.output_shape,
                                           batch_size=1,
                                           transforms=val_transforms,
                                           is_train=False,
                                           )


        """
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
                                                                  ),
                                   validation_steps=num_data[1] // batch_size,
                                   #callbacks=[lr_schedule, logger, cp_callback, monitor] if not DEBUGMODE else [lr_schedule, logger, cp_callback],
                                   callbacks=[lr_schedule, logger, cp_callback],
                                   )
        """
        self.hist = self.model.fit(get_tf_dataset_se(train_dataset[0], train_dataset[1],
                                               self.input_shape,
                                               self.output_shape,
                                               batch_size=batch_size,
                                               transforms=train_transforms,
                                               is_train=True,
                                               max_pair_num=50,
                                               #use_cut_mix=False,
                                               ),
                    steps_per_epoch=num_data[0]//batch_size,
                    epochs=n_epoch,
                    validation_data=get_tf_dataset_se(val_dataset[0], val_dataset[1],
                                               self.input_shape,
                                               self.output_shape,
                                               batch_size=batch_size,
                                               transforms=val_transforms,
                                               max_pair_num=50,
                                               is_train=False,
                                               ),
                    validation_steps=num_data[1]//batch_size,
                    #callbacks=[lr_schedule, logger, cp_callback, monitor] if not DEBUGMODE else [lr_schedule, logger, cp_callback],
                    callbacks=[lr_schedule,
                               logger,
                               cp_callback],
                    )

        """

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
    fold_train_game = fold_info.loc[np.any([fold_info["fold"] == i for i in TRAIN_FOLD], axis=0), "game"].values
    fold_val_game = fold_info.loc[np.any([fold_info["fold"] == i for i in VAL_FOLD], axis=0), "game"].values
    mask_fold_train = [name in fold_train_game for name in game_names]
    mask_fold_val = [name in fold_val_game for name in game_names]

    train_path = list(np.array(end_path)[mask_fold_train]) + list(np.array(side_path)[mask_fold_train])
    val_path = list(np.array(end_path)[mask_fold_val]) + list(np.array(side_path)[mask_fold_val])

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
    if DEBUGMODE:
        train_path = train_path[:4]
        val_path = val_path[:4]
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


def run_training_se_main(epochs=20,
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

    train_path = list(np.array(end_path)[mask_fold_train])  # + list(np.array(side_path)[mask_fold_train])
    val_path = list(np.array(end_path)[mask_fold_val])  # + list(np.array(side_path)[mask_fold_val])

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
    if DEBUGMODE:
        train_path = train_path[:4]
        val_path = val_path[:4]

    train_files_s, train_files_e = load_dataset_se(train_path)
    val_files_s, val_files_e = load_dataset_se(val_path)
    val_files_s = val_files_s[::10]
    val_files_e = val_files_e[::10]

    # print("------------------")
    # print(len(val_files_s))
    # print(val_files_s[0])
    # print(val_files_s[-1])
    # print("------------------")

    step_rate = 6
    step_rate_val = 1

    t_indices = np.arange(len(train_files_s))
    #t2_indices = np.arange(len(train_files_e))
    v_indices = np.arange(len(val_files_s))
    #v2_indices = np.arange(len(val_files_e))

    np.random.shuffle(t_indices)
    # np.random.shuffle(t2_indices)
    np.random.shuffle(v_indices)
    # np.random.shuffle(v2_indices)
    train_files_s = [train_files_s[i] for i in t_indices]
    train_files_e = [train_files_e[i] for i in t_indices]
    val_files_s = [val_files_s[i] for i in v_indices]
    val_files_e = [val_files_e[i] for i in v_indices]
    """

    np.random.shuffle(train_files_e)
    np.random.shuffle(train_files_s)
    np.random.shuffle(val_files_s)
    np.random.shuffle(val_files_e)
    """

    """
    val_files = load_dataset(val_path)
    val_files = val_files[::10]
    val_files_s = val_files[::2]
    val_files_e = val_files[1::2]
    if len(val_files_s) > len(val_files_e):
        val_files_s = val_files_s[1:]
    """

    """
    val_files = load_dataset([p.replace("Endzone", "Sideline") for p in val_path])
    val_files = val_files[::10]
    val_files_s = val_files_s[::10]
    val_files_e = val_files_e[::10]

    for vn, vs in zip(val_files, val_files_e):

        for key in vn.keys():
            if np.any(vn[key]!=vs[key]):
                print(key)
                print(vn[key])
                print(vs[key])
                raise Exception()

    #val_files_s = val_files

    #train_files_e = train_files_s
    #val_files_e = val_files_s
    np.random.shuffle(train_files_e)
    np.random.shuffle(train_files_s)
    np.random.shuffle(val_files_s)
    np.random.shuffle(val_files_e)
    """

    """
    # 一時的
    step_rate = 6
    step_rate_val = 1
    train_files = load_dataset(train_path)# 1564
    val_files = load_dataset(val_path)[::10]
    train_files_s = train_files[::2]
    train_files_e = train_files[1::2]
    val_files_s = val_files[::2]
    val_files_e = val_files[1::2]
    if len(val_files_s) > len(val_files_e):
        val_files_s = val_files_s[1:]
    if len(train_files_s) > len(train_files_e):
        train_files_s = train_files_s[1:]

    np.random.shuffle(train_files_e)
    np.random.shuffle(train_files_s)
    np.random.shuffle(val_files_s)
    np.random.shuffle(val_files_e)
    """
    num_data = [len(train_files_s) // step_rate, len(val_files_s) // step_rate_val]
    print(num_data)

    print(input_shape)
    model_params = {"input_shape": input_shape,
                    "output_shape": output_shape,
                    "weight_file": load_path,
                    "is_train_model": True,
                    }

    train_params = {"train_dataset": [train_files_s, train_files_e],
                    "val_dataset": [val_files_s, val_files_e],
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
                           crop_shape=None):

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
    fold_train_game = fold_info.loc[np.any([fold_info["fold"] == i for i in TRAIN_FOLD], axis=0), "game"].values
    fold_val_game = fold_info.loc[np.any([fold_info["fold"] == i for i in VAL_FOLD], axis=0), "game"].values
    mask_fold_train = [name in fold_train_game for name in game_names]
    mask_fold_val = [name in fold_val_game for name in game_names]

    train_path = list(np.array(end_path)[mask_fold_train]) + list(np.array(side_path)[mask_fold_train])
    val_path = list(np.array(end_path)[mask_fold_val]) + list(np.array(side_path)[mask_fold_val])

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
        preds, inp = nfl.predict(inp)
        pred_mask, pred_label = preds
        # pred_label = pred_label_new#+pred_label_new)/2

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
    print(inp["input_pairs"].numpy().shape, pred_label.numpy().shape, targ["output_contact_label"].numpy().shape)
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

クロスエリアでのアベレージプーリングを見てみる。
concatモデルも引っ付けるタイミングで存在する箇所以外をゼロ特徴にした方がいい。
"""

"""
full------
0.1 tf.Tensor(0.57761735, shape=(), dtype=float32)
0.1 tf.Tensor(0.61934906, shape=(), dtype=float32)
0.2 tf.Tensor(0.6039987, shape=(), dtype=float32)
0.2 tf.Tensor(0.64007574, shape=(), dtype=float32)
0.30000000000000004 tf.Tensor(0.6094741, shape=(), dtype=float32)
0.30000000000000004 tf.Tensor(0.64211756, shape=(), dtype=float32)
0.4 tf.Tensor(0.60308003, shape=(), dtype=float32)
0.4 tf.Tensor(0.6329983, shape=(), dtype=float32)
0.5 tf.Tensor(0.5858064, shape=(), dtype=float32)
0.5 tf.Tensor(0.6135026, shape=(), dtype=float32)
0.6 tf.Tensor(0.5562146, shape=(), dtype=float32)
0.6 tf.Tensor(0.58200425, shape=(), dtype=float32)
0.7000000000000001 tf.Tensor(0.5163115, shape=(), dtype=float32)
0.7000000000000001 tf.Tensor(0.54008555, shape=(), dtype=float32)
0.8 tf.Tensor(0.4576216, shape=(), dtype=float32)
0.8 tf.Tensor(0.47903767, shape=(), dtype=float32)
0.9 tf.Tensor(0.36975214, shape=(), dtype=float32)
0.9 tf.Tensor(0.387412, shape=(), dtype=float32)




0.1 tf.Tensor(0.61401916, shape=(), dtype=float32)
0.1 tf.Tensor(0.653427, shape=(), dtype=float32)
0.2 tf.Tensor(0.6386479, shape=(), dtype=float32)
0.2 tf.Tensor(0.67340994, shape=(), dtype=float32)
0.30000000000000004 tf.Tensor(0.6460579, shape=(), dtype=float32)
0.30000000000000004 tf.Tensor(0.6779122, shape=(), dtype=float32)
0.4 tf.Tensor(0.6386217, shape=(), dtype=float32)
0.4 tf.Tensor(0.6683014, shape=(), dtype=float32)
0.5 tf.Tensor(0.6191587, shape=(), dtype=float32)
0.5 tf.Tensor(0.6469996, shape=(), dtype=float32)
0.6 tf.Tensor(0.59643066, shape=(), dtype=float32)
0.6 tf.Tensor(0.6227849, shape=(), dtype=float32)
0.7000000000000001 tf.Tensor(0.57228583, shape=(), dtype=float32)
0.7000000000000001 tf.Tensor(0.5974951, shape=(), dtype=float32)
0.8 tf.Tensor(0.5344469, shape=(), dtype=float32)
0.8 tf.Tensor(0.55841565, shape=(), dtype=float32)
0.9 tf.Tensor(0.45934093, shape=(), dtype=float32)
0.9 tf.Tensor(0.48022786, shape=(), dtype=float32)
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

    # TODO!!!! 解像度
    # 広めクロップ！
    # 時系列もう一回。3frame予測？今はフレームが2stepとかなので注意する。
    # 多視点使ってみる！！とりあえずはプレイヤの特徴量のみ重ねて予測（ペア情報使わない）
    # 片側のみが特徴量を保有するケースもあると考える。無い側をどうする？平均値もしくはゼロ特徴。gap層出力
    # 他ペアとのsimilarityとる。
    # 3D構成する
    # とりあえずはデータシャッフルされてないか確認。スコアチェッキング。バッチサイズあやしい？？
    # なんかマックスペアおかしくない？？
    # side end　別モデル化
    # 同じse を同じバッチにいれると悪くなる？？なぜ？？
    # filterを二段階にしてバッチ6にしたらなんかかわった？？？？ほんまにわけわかめ・・・。
    # s==eでも悪くなるのでは？？ならない？
    # 03 thresh scoreの追加
    # maskはいてから昆布ありじゃね？

    DEBUGMODE = False
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
    # 画像系特徴ロールとるの試す。あなうめ。

    #TRAIN_FOLD = [0,1]
    #VAL_FOLD = [2,3]

    #TRAIN_FOLD = [0,1,2]
    #VAL_FOLD = [3]
    #TRAIN_FOLD = [0,1,3]
    #VAL_FOLD = [2]
    #TRAIN_FOLD = [0,2,3]
    #VAL_FOLD = [1]

    TRAIN_FOLD = [1, 2, 3]
    VAL_FOLD = [0]

    # VAL23 = True
    name = "fold"
    for fold_num in TRAIN_FOLD:
        name += str(fold_num)
    print("train fold: ", name)
    if STACKFRAME:
        suffix = "_multiframe"
    else:
        suffix = ""

    run_train = True
    AUTO_RESIZE = False

    # prepare_mapping_data() つかう１
    #raise Exception()

    if run_train:
        FIXED_SIZE_DETECTION = False
        CROP_SHAPE = None
        save_path = os.path.join(WEIGHT_DIR, f"ex000_contdet_run054_{name}train_72crop6cbr{suffix}/")
        run_training_main(epochs=num_epoch // 2 if STACKFRAME else num_epoch,
                          batch_size=int(6),  # int(12),
                          input_shape=(512, 896, 3),  # (384, 640, 3),
                          output_shape=(256, 448),
                          load_path=os.path.join(WEIGHT_DIR, "map_model_final_weights.h5"),
                          save_path=save_path)

        """
        # SIZE = "SM"
        #save_path = os.path.join(WEIGHT_DIR, f"ex000_contdet_runXX_{name}train_hresolution/")

        # save_path = os.path.join(WEIGHT_DIR, f"ex004_contdet_run00_{name}train_g_only/")

        CROP_SHAPE=(512, 896, 3)
        input_shape = (int(512*1.5), int(896*1.5), 3)
        output_shape = (int(256*1.5), int(448*1.5))
        run_training_main(epochs=num_epoch,
                          batch_size=int(6-2),#int(12),
                         ##input_shape=(512+64, 896+128, 3),#(384, 640, 3),
                         ##output_shape=(256+32, 448+64),
                         input_shape=input_shape,#(512, 896, 3),#(384, 640, 3),
                         output_shape=output_shape,
                         #(192, 320),
                         #load_path=None,#os.path.join(WEIGHT_DIR, f"ex000_contdet_run46_{name}train_72crop6cbr_concat_dist_Q/final_weights.h5"),
                         load_path=os.path.join(WEIGHT_DIR, "map_model_final_weights.h5"),
                         save_path=save_path)
        """

    else:
        # 43 ?? better
        run_validation_predict(os.path.join(WEIGHT_DIR, f"ex000_contdet_run054_{name}train_72crop6cbr/final_weights.h5"),
                               #input_shape=(960, 1728, 3),
                               #output_shape=(480, 864),
                               input_shape=(704, 1280, 3),
                               output_shape=(352, 640),
                               draw_pred=False)

        # run_validation_predict(os.path.join(WEIGHT_DIR, f"ex000_contdet_run43_{name}train_72crop6cbr_concat_dist/final_weights.h5"),
        #                           #input_shape=(960, 1728, 3),
        #                           #output_shape=(480, 864),
        #                           input_shape=(704, 1280, 3),
        #                           output_shape=(352, 640),
        #                           draw_pred=False)

        """
        load_path = os.path.join(WEIGHT_DIR, f"ex000_contdet_run49_{name}train_hresolution/final_weights.h5")
        run_validation_predict(load_path,
                                   #input_shape=(960, 1728, 3),
                                   #output_shape=(480, 864),
                                   input_shape=(int(1088), int(1280*1.5), 3), # 150%
                                   output_shape=(int(1088/2), int(640*1.5)),
                                   draw_pred=False,
                                   crop_shape=(720, 1280, 3))

        """

        """
        run_validation_predict(os.path.join(WEIGHT_DIR, f"ex000_contdet_run036_fold23train_72crop6cbr_concat_distance/final_weights.h5"),
                                   #input_shape=(960, 1728, 3),
                                   #output_shape=(480, 864),
                                   input_shape=(704, 1280, 3),
                                   output_shape=(352, 640),
                                   draw_pred=False)
        """

    """
    # High Resolution 23VAL
    ground------
    0.1 tf.Tensor(0.64292264, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6840334, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6958002, shape=(), dtype=float32)
    0.4 tf.Tensor(0.69152826, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6777197, shape=(), dtype=float32)
    0.6 tf.Tensor(0.66013676, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.63246065, shape=(), dtype=float32)
    0.8 tf.Tensor(0.59508264, shape=(), dtype=float32)
    0.9 tf.Tensor(0.5142878, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.5920514, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6205164, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6283723, shape=(), dtype=float32)
    0.4 tf.Tensor(0.62567186, shape=(), dtype=float32)
    0.5 tf.Tensor(0.61169064, shape=(), dtype=float32)
    0.6 tf.Tensor(0.5925554, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5679702, shape=(), dtype=float32)
    0.8 tf.Tensor(0.52262706, shape=(), dtype=float32)
    0.9 tf.Tensor(0.40493986, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.6047017, shape=(), dtype=float32)
    0.1 tf.Tensor(0.6438259, shape=(), dtype=float32)
    0.2 tf.Tensor(0.63388854, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6679562, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6420134, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6733119, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6390037, shape=(), dtype=float32)
    0.4 tf.Tensor(0.66837406, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6250506, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6529788, shape=(), dtype=float32)
    0.6 tf.Tensor(0.60609335, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6323092, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.58084035, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.6055803, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5364982, shape=(), dtype=float32)
    0.8 tf.Tensor(0.55948526, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4243018, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4429098, shape=(), dtype=float32)


    # NORMAL REOLUTION run 43 val 23
    ground------
    0.1 tf.Tensor(0.6403133, shape=(), dtype=float32)
    0.2 tf.Tensor(0.68539995, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6918818, shape=(), dtype=float32)
    0.4 tf.Tensor(0.68645835, shape=(), dtype=float32)
    0.5 tf.Tensor(0.66992474, shape=(), dtype=float32)
    0.6 tf.Tensor(0.65189815, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.6255581, shape=(), dtype=float32)
    0.8 tf.Tensor(0.58791685, shape=(), dtype=float32)
    0.9 tf.Tensor(0.5199979, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.59113014, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6187401, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6282852, shape=(), dtype=float32)
    0.4 tf.Tensor(0.62485623, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6084877, shape=(), dtype=float32)
    0.6 tf.Tensor(0.5843499, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5551044, shape=(), dtype=float32)
    0.8 tf.Tensor(0.51735514, shape=(), dtype=float32)
    0.9 tf.Tensor(0.42186904, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.603477, shape=(), dtype=float32)
    0.1 tf.Tensor(0.6427402, shape=(), dtype=float32)
    0.2 tf.Tensor(0.63242006, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6665779, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.64121693, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6724156, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6374356, shape=(), dtype=float32)
    0.4 tf.Tensor(0.666676, shape=(), dtype=float32)
    0.5 tf.Tensor(0.62109315, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6488571, shape=(), dtype=float32)
    0.6 tf.Tensor(0.59784067, shape=(), dtype=float32)
    0.6 tf.Tensor(0.62368304, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.56883985, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.59319466, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5308552, shape=(), dtype=float32)
    0.8 tf.Tensor(0.55364746, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4392991, shape=(), dtype=float32)
    0.9 tf.Tensor(0.45856687, shape=(), dtype=float32)


    # NORMAL REOLUTION run 43 train01 val 3
    ground------
    0.1 tf.Tensor(0.6226183, shape=(), dtype=float32)
    0.2 tf.Tensor(0.677719, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6831449, shape=(), dtype=float32)
    0.4 tf.Tensor(0.67601156, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6679071, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6589108, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.6387238, shape=(), dtype=float32)
    0.8 tf.Tensor(0.6069618, shape=(), dtype=float32)
    0.9 tf.Tensor(0.53761667, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.58947897, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6190696, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.62516487, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6228592, shape=(), dtype=float32)
    0.5 tf.Tensor(0.60845685, shape=(), dtype=float32)
    0.6 tf.Tensor(0.57909346, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5536243, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5124991, shape=(), dtype=float32)
    0.9 tf.Tensor(0.41358355, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.599808, shape=(), dtype=float32)
    0.1 tf.Tensor(0.63867205, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6320095, shape=(), dtype=float32)
    0.2 tf.Tensor(0.665443, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.63770336, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.66845673, shape=(), dtype=float32)
    0.4 tf.Tensor(0.63458145, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6633646, shape=(), dtype=float32)
    0.5 tf.Tensor(0.62103254, shape=(), dtype=float32)
    0.5 tf.Tensor(0.648058, shape=(), dtype=float32)
    0.6 tf.Tensor(0.59473133, shape=(), dtype=float32)
    0.6 tf.Tensor(0.62013525, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.56986165, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5935588, shape=(), dtype=float32)
    0.8 tf.Tensor(0.53004515, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5523882, shape=(), dtype=float32)
    0.9 tf.Tensor(0.43577173, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4550742, shape=(), dtype=float32)

    # NORMAL REOLUTION run 43 train012 val 3
    ground------
    0.1 tf.Tensor(0.633806, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6916334, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.70354015, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6970654, shape=(), dtype=float32)
    0.5 tf.Tensor(0.67887074, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6632769, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.6327206, shape=(), dtype=float32)
    0.8 tf.Tensor(0.59850323, shape=(), dtype=float32)
    0.9 tf.Tensor(0.5165779, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.5855344, shape=(), dtype=float32)
    0.2 tf.Tensor(0.62256306, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.62893575, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6257399, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6134309, shape=(), dtype=float32)
    0.6 tf.Tensor(0.58702946, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5589781, shape=(), dtype=float32)
    0.8 tf.Tensor(0.519357, shape=(), dtype=float32)
    0.9 tf.Tensor(0.42809284, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.5981841, shape=(), dtype=float32)
    0.1 tf.Tensor(0.63749117, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6369939, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6702133, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6438508, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.67432123, shape=(), dtype=float32)
    0.4 tf.Tensor(0.64014286, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6684947, shape=(), dtype=float32)
    0.5 tf.Tensor(0.62688524, shape=(), dtype=float32)
    0.5 tf.Tensor(0.653154, shape=(), dtype=float32)
    0.6 tf.Tensor(0.60201126, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6266245, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5732488, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.59617305, shape=(), dtype=float32)
    0.8 tf.Tensor(0.53421795, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5556482, shape=(), dtype=float32)
    0.9 tf.Tensor(0.44411635, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4629784, shape=(), dtype=float32)







    ground------
    0.1 tf.Tensor(0.6444714, shape=(), dtype=float32)
    0.2 tf.Tensor(0.68351775, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6844765, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6773337, shape=(), dtype=float32)
    0.5 tf.Tensor(0.66389054, shape=(), dtype=float32)
    0.6 tf.Tensor(0.648249, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.6245819, shape=(), dtype=float32)
    0.8 tf.Tensor(0.58647287, shape=(), dtype=float32)
    0.9 tf.Tensor(0.52960557, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.5919598, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6172264, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.62994444, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6211808, shape=(), dtype=float32)
    0.5 tf.Tensor(0.60413915, shape=(), dtype=float32)
    0.6 tf.Tensor(0.58221745, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.55847275, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5252869, shape=(), dtype=float32)
    0.9 tf.Tensor(0.45095658, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.6041022, shape=(), dtype=float32)
    0.1 tf.Tensor(0.6430726, shape=(), dtype=float32)
    0.2 tf.Tensor(0.62993217, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6642645, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6411574, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.67234, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6329025, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6622267, shape=(), dtype=float32)
    0.5 tf.Tensor(0.61652654, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6440125, shape=(), dtype=float32)
    0.6 tf.Tensor(0.59549206, shape=(), dtype=float32)
    0.6 tf.Tensor(0.62152797, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.57158476, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.59641635, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5373683, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5607498, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4653842, shape=(), dtype=float32)
    0.9 tf.Tensor(0.48575664, shape=(), dtype=float32)
















    motomoto
    ground------
    0.1 tf.Tensor(0.68451273, shape=(), dtype=float32)
    0.2 tf.Tensor(0.7152696, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.7062826, shape=(), dtype=float32)
    0.4 tf.Tensor(0.68799293, shape=(), dtype=float32)
    0.5 tf.Tensor(0.66935074, shape=(), dtype=float32)
    0.6 tf.Tensor(0.64583004, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.6180397, shape=(), dtype=float32)
    0.8 tf.Tensor(0.58012843, shape=(), dtype=float32)
    0.9 tf.Tensor(0.49922946, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.5849773, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6239815, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6397178, shape=(), dtype=float32)
    0.4 tf.Tensor(0.63296145, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6181393, shape=(), dtype=float32)
    0.6 tf.Tensor(0.5969931, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.55965924, shape=(), dtype=float32)
    0.8 tf.Tensor(0.51814103, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4443329, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.603855, shape=(), dtype=float32)
    0.1 tf.Tensor(0.64246035, shape=(), dtype=float32)
    0.2 tf.Tensor(0.64053494, shape=(), dtype=float32)
    0.2 tf.Tensor(0.67326194, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.65285194, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.68210846, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6445047, shape=(), dtype=float32)
    0.4 tf.Tensor(0.67207783, shape=(), dtype=float32)
    0.5 tf.Tensor(0.62917435, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6551952, shape=(), dtype=float32)
    0.6 tf.Tensor(0.60757184, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6318053, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.57153887, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5945801, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5303093, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5517933, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4548924, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4737839, shape=(), dtype=float32)



    ground------
    0.1 tf.Tensor(0.6782205, shape=(), dtype=float32)
    0.2 tf.Tensor(0.71008176, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.7122158, shape=(), dtype=float32)
    0.4 tf.Tensor(0.70745784, shape=(), dtype=float32)
    0.5 tf.Tensor(0.68743235, shape=(), dtype=float32)
    0.6 tf.Tensor(0.66702014, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.63385683, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5942265, shape=(), dtype=float32)
    0.9 tf.Tensor(0.5131027, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.58648306, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6261117, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6413001, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6352176, shape=(), dtype=float32)
    0.5 tf.Tensor(0.62283605, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6049857, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5664075, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5164292, shape=(), dtype=float32)
    0.9 tf.Tensor(0.44444385, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.60485935, shape=(), dtype=float32)
    0.1 tf.Tensor(0.6434133, shape=(), dtype=float32)
    0.2 tf.Tensor(0.64237475, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6749249, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6554391, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6847493, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6495848, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6771013, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6360712, shape=(), dtype=float32)
    0.5 tf.Tensor(0.66213113, shape=(), dtype=float32)
    0.6 tf.Tensor(0.61766005, shape=(), dtype=float32)
    0.6 tf.Tensor(0.64194465, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5797003, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.6026323, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5310999, shape=(), dtype=float32)
    0.8 tf.Tensor(0.55253756, shape=(), dtype=float32)
    0.9 tf.Tensor(0.45712075, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4760883, shape=(), dtype=float32)




    fold 3(012 train)
    ground------
    0.1 tf.Tensor(0.67194355, shape=(), dtype=float32)
    0.2 tf.Tensor(0.71032697, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.711421, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6983908, shape=(), dtype=float32)
    0.5 tf.Tensor(0.68996406, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6760897, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.65540814, shape=(), dtype=float32)
    0.8 tf.Tensor(0.6066513, shape=(), dtype=float32)
    0.9 tf.Tensor(0.5192881, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.59108466, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6282925, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.638378, shape=(), dtype=float32)
    0.4 tf.Tensor(0.639054, shape=(), dtype=float32)
    0.5 tf.Tensor(0.62954015, shape=(), dtype=float32)
    0.6 tf.Tensor(0.60753363, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5769933, shape=(), dtype=float32)
    0.8 tf.Tensor(0.53853387, shape=(), dtype=float32)
    0.9 tf.Tensor(0.46654496, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.60810167, shape=(), dtype=float32)
    0.1 tf.Tensor(0.64606684, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6443816, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6766222, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.65289253, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6824232, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6515181, shape=(), dtype=float32)
    0.4 tf.Tensor(0.67904204, shape=(), dtype=float32)
    0.5 tf.Tensor(0.64213943, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6678492, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6212715, shape=(), dtype=float32)
    0.6 tf.Tensor(0.64562356, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5921121, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.61532027, shape=(), dtype=float32)
    0.8 tf.Tensor(0.55171967, shape=(), dtype=float32)
    0.8 tf.Tensor(0.57360256, shape=(), dtype=float32)
    0.9 tf.Tensor(0.47685114, shape=(), dtype=float32)
    0.9 tf.Tensor(0.49656346, shape=(), dtype=float32)


    fold 2(013 train)
    ground------
    0.1 tf.Tensor(0.70037913, shape=(), dtype=float32)
    0.2 tf.Tensor(0.7357472, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.74768806, shape=(), dtype=float32)
    0.4 tf.Tensor(0.73818856, shape=(), dtype=float32)
    0.5 tf.Tensor(0.7219805, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6996061, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.67373246, shape=(), dtype=float32)
    0.8 tf.Tensor(0.62466437, shape=(), dtype=float32)
    0.9 tf.Tensor(0.52331746, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.60250086, shape=(), dtype=float32)
    0.2 tf.Tensor(0.63033146, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6441487, shape=(), dtype=float32)
    0.4 tf.Tensor(0.639042, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6272146, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6019037, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5729247, shape=(), dtype=float32)
    0.8 tf.Tensor(0.54317874, shape=(), dtype=float32)
    0.9 tf.Tensor(0.47213465, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.62126637, shape=(), dtype=float32)
    0.1 tf.Tensor(0.6589622, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6495455, shape=(), dtype=float32)
    0.2 tf.Tensor(0.68241847, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6630254, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.69266397, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6573409, shape=(), dtype=float32)
    0.4 tf.Tensor(0.68536484, shape=(), dtype=float32)
    0.5 tf.Tensor(0.64484143, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6713383, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6199674, shape=(), dtype=float32)
    0.6 tf.Tensor(0.64512056, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5913396, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.6151385, shape=(), dtype=float32)
    0.8 tf.Tensor(0.55817926, shape=(), dtype=float32)
    0.8 tf.Tensor(0.58040994, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4818768, shape=(), dtype=float32)
    0.9 tf.Tensor(0.5016697, shape=(), dtype=float32)


    fold1(023 train)
    ground------
    0.1 tf.Tensor(0.6511399, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6833933, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.6944672, shape=(), dtype=float32)
    0.4 tf.Tensor(0.69018215, shape=(), dtype=float32)
    0.5 tf.Tensor(0.68260115, shape=(), dtype=float32)
    0.6 tf.Tensor(0.67416835, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.6484703, shape=(), dtype=float32)
    0.8 tf.Tensor(0.60425353, shape=(), dtype=float32)
    0.9 tf.Tensor(0.5184109, shape=(), dtype=float32)
    not ground------
    0.1 tf.Tensor(0.58440256, shape=(), dtype=float32)
    0.2 tf.Tensor(0.6152205, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.62533516, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6248151, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6104806, shape=(), dtype=float32)
    0.6 tf.Tensor(0.587335, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.5590665, shape=(), dtype=float32)
    0.8 tf.Tensor(0.52953887, shape=(), dtype=float32)
    0.9 tf.Tensor(0.46140635, shape=(), dtype=float32)
    full------
    0.1 tf.Tensor(0.60097003, shape=(), dtype=float32)
    0.1 tf.Tensor(0.6428305, shape=(), dtype=float32)
    0.2 tf.Tensor(0.63061345, shape=(), dtype=float32)
    0.2 tf.Tensor(0.66719234, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.640137, shape=(), dtype=float32)
    0.30000000000000004 tf.Tensor(0.67366034, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6386064, shape=(), dtype=float32)
    0.4 tf.Tensor(0.6697579, shape=(), dtype=float32)
    0.5 tf.Tensor(0.6244187, shape=(), dtype=float32)
    0.5 tf.Tensor(0.65369916, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6023221, shape=(), dtype=float32)
    0.6 tf.Tensor(0.6298789, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.57368106, shape=(), dtype=float32)
    0.7000000000000001 tf.Tensor(0.59965426, shape=(), dtype=float32)
    0.8 tf.Tensor(0.5416374, shape=(), dtype=float32)
    0.8 tf.Tensor(0.56573427, shape=(), dtype=float32)
    0.9 tf.Tensor(0.4705029, shape=(), dtype=float32)
    0.9 tf.Tensor(0.49199376, shape=(), dtype=float32)



    """
