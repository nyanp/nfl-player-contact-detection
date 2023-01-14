# -*- coding: utf-8 -*-
"""
@author: k_mat
Training NFL helmet detector
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

from kmat.train_utils.tf_Augmentations_detection import Compose, Oneof, HorizontalFlip, VerticalFlip, Crop, Center_Crop, Resize, BrightnessContrast, CoarseDropout, HueShift, ToGlay, Blur, PertialBrightnessContrast, Shadow, GaussianNoise, Rotation
from kmat.train_utils.tf_Augmentations_detection import Center_Crop_by_box_shape, Crop_by_box_shape
from kmat.train_utils.scheduler import lrs_wrapper, lrs_wrapper_cos
from kmat.train_utils.dataloader import load_dataset, get_dataset_detection, load_dataset_helmet_imgs
from kmat.model.model_detection import build_detection_model
#from kmat.train_utils.log_utils import get_kwargs_of_current_func, save_log_params



class LogHolder():
    """
    This class combines mlflow log and original log,
    by experiment name and run name.
    This will help writing and reading log and parameters easily.
    """
    def __init__(self, experiment_name, run_name, base_path="model/weights/"):
        if type(experiment_name)!=str or type(experiment_name)!=str:
            raise Exception("use name of type str.")
        self.base_path = base_path
        self.experiment_name = experiment_name
        self.run_name= run_name
        self.exp_run_name = experiment_name + "_" + run_name
        self.save_path = self.base_path + self.exp_run_name + "/"
        self.final_weight_dir = self.save_path + "final_weights.h5"
        self.train_params_dir = self.save_path + "params.json"
        self.train_log_dir = self.save_path + "hist.csv"
        ##self.pred_imgs_path = self.save_path + "predict_imgs/"
        ##if not os.path.exists(self.pred_imgs_path): os.makedirs(self.pred_imgs_path)


    def load_log_params(self):
        params_exists = os.path.exists(self.train_params_dir)
        weight_exists = os.path.exists(self.final_weight_dir)
        #self.mlflow_log_exists = os.path.exists(self.final_weight_dir)
        if params_exists:# and weight_exists:
            if not weight_exists:
                warnings.warn("Weight file '{}' do not exists. There is only params.json.".format(self.final_weight_dir))
            self.log_params = json.load(open(self.train_params_dir))
            self.log_params["Model_Params"]["weight_file"] = self.final_weight_dir
            if "run_id" in self.log_params.keys():
                self.run_id = self.log_params["run_id"]
            else:
                self.run_id = None
            return True
        #elif params_exists or weight_exists:
        #    raise Exception("params.json or final_weights.h5 does not exists in {}".format(self.exp_run_name))
        else:
            self.log_params = None
            self.run_id = None
            return False

    def load_model_params(self):
        is_loaded = self.load_log_params()
        if not is_loaded:
            raise Exception("params.json does not exists in {}".format(self.exp_run_name))
        return self.log_params["Model_Params"]

    def start_mlflow_logging(self, start_tf_autolog=True):
        mlflow.end_run()#restart if it already has been running
        is_loaded = self.load_log_params()
        if self.run_id is None:
            print("Newly start experiment:{} run:{}".format(self.experiment_name, self.run_name))
        else:
            print("Restart logging experiment:{}, run:{}, id:{}".format(self.experiment_name, self.run_name, self.run_id))
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name, run_id=self.run_id)
        self.run_id = mlflow.active_run().info.run_id
        if is_loaded:
            mlflow.log_artifact(self.train_params_dir)
        if start_tf_autolog:
            mlflow.tensorflow.autolog(log_models=False, silent=True)

    def update_mlflow_logs(self):
        """
        update log of mlflow as follows:
        - set batch value
        - conbine with original log
        """
        is_loaded = self.load_log_params()
        if is_loaded:
            mlflow.log_param("batchsize", self.log_params["Train_Params"]["batch_size"])
            mlflow.log_artifact(self.train_params_dir)            #mlflow.log_dict(self.log_params, "params.json")
            ##mlflow.log_artifacts(self.pred_imgs_path, artifact_path="pred_imgs")

        else:
            print("{} cannnot be loaded.".format(self.train_params_dir))

    def end_mlflow_logging(self):
        mlflow.end_run()

    def delete(self):
        """
        delete mlflow logs and weight files
        NOT IMPLIMENTED
        """
        raise Exception("mijissou")
        is_loaded = self.load_log_params()
        if not is_loaded:
            raise Exception("params.json does not exists in {}".format(self.exp_run_name))
        if self.run_id is None:
            raise Exception("run id is not written in params.json in {}".format(self.exp_run_name))
        yes_or_no = input("Are you sure to delete {}? if yes, input y in console".format(self.exp_run_name))
        if yes_or_no == "y":
            # to be implemented
            print("{} is deleted.".format(self.exp_run_name))
            mlflow.delete_run()
        else:
            print("{} is NOT deleted.".format(self.exp_run_name))

    def clean_logs_with_no_training(self):

        pass


class NFL_Predictor():
    def __init__(self, #num_classes=30, solo_score_thresh=0.3,
                 input_shape=(288,288,4),
                 output_shape=(144,144),
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
        self.model, self.custom_losses, self.custom_loss_weights = build_detection_model(self.input_shape,
                                                                          minimum_stride=self.input_shape[0]//self.output_shape[0],
                                                                          is_train=self.is_train_model,
                                                                          backbone="effv2{}".format(V2_MODEL_TYPE),
                                                                          from_scratch=FROM_SCRATCH)
        if not weight_file is None:
            self.model.load_weights(weight_file)


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
                                      period = 10,
                                      verbose = 1)

        optim = Adam(lr=learning_rate, clipnorm=0.001)
        self.model.compile(loss = self.custom_losses,
                           loss_weights = self.custom_loss_weights,
                           optimizer = optim,
                           )

        if FIXED_SIZE_DETECTION:
            transforms_train = [
                          HorizontalFlip(p=0.5),
                          Crop_by_box_shape(p=1, target_box_length=TARGET_SIZE, target_img_height=self.input_shape[0], target_img_width=self.input_shape[1], img_height=720, img_width=1280),
                          Resize(height=self.input_shape[0], width=self.input_shape[1], target_height=self.output_shape[0], target_width=self.output_shape[1]),
                          BrightnessContrast(p=1.0),
                          HueShift(p=0.8, min_offset=-0.25, max_offset=0.25),
                          Oneof(
                                transforms = [PertialBrightnessContrast(p=0.2,
                                                                        max_holes=3, max_height=80, max_width=80,
                                                                        min_holes=1, min_height=30, min_width=30,
                                                                        min_offset=0, max_offset=30,
                                                                        min_multiply=1.0, max_multiply=1.5),
                                              Shadow(p=0.4,
                                                     max_holes=3, max_height=120, max_width=120,
                                                     min_holes=1, min_height=50, min_width=50,
                                                     min_strength=0.2, max_strength=0.8, shadow_color=0.0)
                                              ],
                                probs=[0.2,0.2]
                                ),
                          Blur(p=0.1),
                          GaussianNoise(p=0.1, min_var=10, max_var=40),
                          ToGlay(p=0.1),
                          ]
            transforms_val = [
                          Center_Crop_by_box_shape(p=1, target_box_length=25, target_img_height=self.input_shape[0], target_img_width=self.input_shape[1], img_height=720, img_width=1280),
                          Resize(height=self.input_shape[0], width=self.input_shape[1], target_height=self.output_shape[0], target_width=self.output_shape[1]),
                          ]
        else:
            transforms_train = [
                          HorizontalFlip(p=0.5),# not active for jersey classifier
                          Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                          BrightnessContrast(p=1.0),
                          HueShift(p=0.8, min_offset=-0.25, max_offset=0.25),
                          Oneof(
                                transforms = [PertialBrightnessContrast(p=0.2,
                                                                        max_holes=3, max_height=80, max_width=80,
                                                                        min_holes=1, min_height=30, min_width=30,
                                                                        min_offset=0, max_offset=30,
                                                                        min_multiply=1.0, max_multiply=1.5),
                                              Shadow(p=0.4,
                                                     max_holes=3, max_height=120, max_width=120,
                                                     min_holes=1, min_height=50, min_width=50,
                                                     min_strength=0.2, max_strength=0.8, shadow_color=0.0)
                                              ],
                                probs=[0.2,0.2]
                                ),

                          Blur(p=0.1),
                          GaussianNoise(p=0.1, min_var=10, max_var=40),
                          ToGlay(p=0.1),
                          ]


            transforms_val = [
                          Center_Crop(p=1, min_height=self.input_shape[0], min_width=self.input_shape[1]),
                          ]

        train_transforms = Compose(transforms_train)
        val_transforms = Compose(transforms_val)

        print("step per epoch", num_data[0]//(STEP_PER_EPOCH_RATE * batch_size), num_data[1]//batch_size)
        self.hist = self.model.fit(get_dataset_detection(train_dataset,
                                               self.input_shape,
                                               self.output_shape,
                                               batch_size=batch_size,
                                               transforms=train_transforms,
                                               is_train=True,
                                               use_cut_mix=False),
                    steps_per_epoch=num_data[0]//(STEP_PER_EPOCH_RATE*batch_size),
                    epochs=n_epoch,
                    validation_data=get_dataset_detection(val_dataset,
                                               self.input_shape,
                                               self.output_shape,
                                               batch_size=batch_size,
                                               transforms=val_transforms,
                                               is_train=False),
                    validation_steps=num_data[1]//batch_size,
                    callbacks=[lr_schedule, logger, cp_callback],
                    )
        print("Saving weights and results...")
        self.model.save_weights(save_dir + "final_weights.h5")
        csv_hist = save_dir + "hist.csv"
        pd.DataFrame(self.hist.history).to_csv(csv_hist, index=False)
        print("Done")


def set_seeds(num=111):
    tf.random.set_seed(num)
    np.random.seed(num)
    random.seed(num)
    os.environ["PYTHONHASHSEED"] = str(num)


def run_training_pre(epochs=20, batch_size=4,
                     input_shape=(448, 768, 3), output_shape=(224, 384), save_path=""):

    K.clear_session()
    set_seeds(111)

    paths_endzone = sorted(glob.glob(DATA_PATH + "*Endzone"))
    paths_sideline = sorted(glob.glob(DATA_PATH + "*Sideline"))
    np.random.shuffle(paths_endzone)
    np.random.shuffle(paths_sideline)

    interval = 1
    extra_files = load_dataset_helmet_imgs(img_path = DATA_PATH_EXT,
                                           annotation_path = ANNOTATINO_PATH_EXT)

    #endzone_files_train, endzone_files_val = load_dataset(paths_endzone[:], frame_interval=interval, rate=0.7, detection_dataset=True)
    #sideline_files_train, sideline_files_val = load_dataset(paths_sideline[:], frame_interval=interval, rate=0.7, detection_dataset=True)

    #paths_endzone = [path.replace("_interp","") for path in paths_endzone]
    #paths_sideline = [path.replace("_interp","") for path in paths_sideline]
    _, endzone_files_val = load_dataset(paths_endzone[:], frame_interval=interval, rate=0.7, detection_dataset=True)
    _, sideline_files_val = load_dataset(paths_sideline[:], frame_interval=interval, rate=0.7, detection_dataset=True)


    train_files = extra_files#endzone_files_train + sideline_files_train#load_data(path=paths[0]) + load_data(path=paths[1]) + load_data(path=paths[2])
    val_files = endzone_files_val + sideline_files_val#load_data(path=paths[4]) + load_data(path=paths[5])

    np.random.shuffle(train_files)
    np.random.shuffle(val_files)


    num_data = [len(train_files), len(val_files)]
    print(num_data)

    model_params = {"input_shape": input_shape,
                    "output_shape": output_shape,
                    "weight_file": None,
                    "is_train_model": True,
                    }

    train_params = {"train_dataset": train_files,
                    "val_dataset": val_files,
                    "num_data": num_data,
                    "save_dir": save_path,
                    "learning_rate": 0.00025*batch_size/8,
                    "n_epoch": epochs,
                    "batch_size": batch_size,#8
                    }



    with tf.device('/device:GPU:0'):
        nfl = NFL_Predictor(**model_params)
        nfl.train(**train_params)


def run_training_main(epochs=20, batch_size=4,
                      input_shape=(448, 768, 3), output_shape=(224, 384),
                      load_path="", save_path="", train_all=False):

    K.clear_session()
    set_seeds(111)

    paths_endzone = sorted(glob.glob(DATA_PATH + "*Endzone"))
    paths_sideline = sorted(glob.glob(DATA_PATH + "*Sideline"))
    np.random.shuffle(paths_endzone)
    np.random.shuffle(paths_sideline)

    interval = 1

    #extra_files = load_dataset_helmet_imgs()
    endzone_files_train, endzone_files_val = load_dataset(paths_endzone[:], frame_interval=interval, rate=0.7, detection_dataset=True)
    sideline_files_train, sideline_files_val = load_dataset(paths_sideline[:], frame_interval=interval, rate=0.7, detection_dataset=True)

    if train_all:
        train_files = endzone_files_train + sideline_files_train + endzone_files_val + sideline_files_val# + extra_files#load_data(path=paths[0]) + load_data(path=paths[1]) + load_data(path=paths[2])
    else:
        train_files = endzone_files_train + sideline_files_train

    #paths_endzone = [path.replace("_interp","") for path in paths_endzone]
    #paths_sideline = [path.replace("_interp","") for path in paths_sideline]
    #_, endzone_files_val = load_dataset(paths_endzone[:], frame_interval=interval, rate=0.7, detection_dataset=True)
    #_, sideline_files_val = load_dataset(paths_sideline[:], frame_interval=interval, rate=0.7, detection_dataset=True)

    val_files = endzone_files_val + sideline_files_val#load_data(path=paths[4]) + load_data(path=paths[5])

    np.random.shuffle(train_files)
    np.random.shuffle(val_files)

    num_data = [len(train_files), len(val_files)]
    print(num_data)

    model_params = {"input_shape": input_shape,
                    "output_shape": output_shape,
                    "weight_file": os.path.join(load_path, "final_weights.h5"),
                    "is_train_model": True,
                    }

    train_params = {"train_dataset": train_files,
                    "val_dataset": val_files,
                    "num_data": num_data,
                    "save_dir": save_path,
                    "learning_rate": 0.00025*batch_size/8,
                    "n_epoch": epochs,
                    "batch_size": batch_size,
                    }

    with tf.device('/device:GPU:0'):
        nfl = NFL_Predictor(**model_params)
        nfl.train(**train_params)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_rate', type=float, default=1.0)
    args = parser.parse_args()
    DEBUG = args.debug
    batch_rate = args.batch_rate

    num_epoch =  2 if DEBUG else 20

    setting_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"SETTINGS.json")
    DIRS = json.load(open(setting_file))
    BASE_DIR = DIRS["RAW_DATA_DIR"]
    TRAIN_DIR = DIRS["TRAIN_DATA_DIR"]
    WEIGHT_DIR = DIRS["WEIGHT_DIR"]# model/weights/
    os.makedirs(WEIGHT_DIR, exist_ok=True)


    DATA_PATH_EXT = os.path.join(BASE_DIR, "images/")
    ANNOTATINO_PATH_EXT = os.path.join(BASE_DIR, "image_labels.csv")
    DATA_PATH = os.path.join(TRAIN_DIR, "train_img_interp/")
    FROM_SCRATCH = False
    TARGET_SIZE = 25

    # normal resolution model
    FIXED_SIZE_DETECTION = False
    V2_MODEL_TYPE = "s"
    # pretrain
    STEP_PER_EPOCH_RATE = 1
    save_path = os.path.join(WEIGHT_DIR, "det_base_pre/")
    run_training_pre(epochs=num_epoch,
                     batch_size=int(12*batch_rate),
                     input_shape=(384, 640, 3),
                     output_shape=(192, 320),
                     save_path=save_path)
    # finetune
    STEP_PER_EPOCH_RATE = 6
    load_path = save_path
    save_path = os.path.join(WEIGHT_DIR, "det_base/")
    run_training_main(epochs=num_epoch,
                      batch_size=int(12*batch_rate),
                     input_shape=(384, 640, 3),
                     output_shape=(192, 320),
                     load_path=load_path,
                     save_path=save_path)

    # high resolution model
    FIXED_SIZE_DETECTION = True
    list_backbone = ["s", "m", "l", "xl"]
    list_batch_size = [8, 6, 4, 3]
    for V2_MODEL_TYPE, batch_size in zip(list_backbone, list_batch_size):
        # pretrain
        STEP_PER_EPOCH_RATE = 1
        save_path = os.path.join(WEIGHT_DIR, "det_v2{}_pre/".format(V2_MODEL_TYPE))
        run_training_pre(epochs=num_epoch,
                         batch_size=int(batch_rate*batch_size),
                         input_shape=(448, 768, 3),
                         output_shape=(224, 384),
                         save_path=save_path)
        # finetune
        STEP_PER_EPOCH_RATE = 6
        load_path = save_path
        save_path = os.path.join(WEIGHT_DIR, "det_v2{}/".format(V2_MODEL_TYPE))
        run_training_main(epochs=num_epoch,
                          batch_size=int(batch_rate*batch_size),
                          input_shape=(448, 768, 3),
                          output_shape=(224, 384),
                          load_path=load_path,
                          save_path=save_path,
                          train_all=True)




