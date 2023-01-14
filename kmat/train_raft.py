# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:17:45 2021

@author: kmat
"""

import os
import random
import glob
import json
import gc

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, CSVLogger, ModelCheckpoint
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from model.RAFT import RAFT, loss_wrapper
from train_utils.scheduler import lrs_wrapper, lrs_wrapper_cos
from train_utils.dataloader import load_dataset, get_dataset_raft, get_dataset_raft_inference
from train_utils.tf_Augmentations_raft import Compose, BrightnessContrast, HueShift

setting_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"SETTINGS.json")
SETTING = json.load(open(setting_file))

def flow_to_rgb(flow):
    hsv = np.zeros((flow.shape[0],flow.shape[1],3))
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb

class OpticalFlowNet():
    def __init__(self,
                 input_shape=(200,480,3), 
                 weight_file=None, 
                 is_train_model=False,
                 from_scratch=False):
        
        print("\rLoading Models...", end="")
        
        self.input_shape = tuple(input_shape)
        self.is_train_model = is_train_model
        self.load_model(weight_file, is_train_model, from_scratch)
        print("Loading Models......Finish")
        
            
    def load_model(self, weight_file=None, is_train_model=False, from_scratch=False):
        """build model and load weights"""
        self.model = RAFT(iters_pred=6 if is_train_model else 4)
        test_run = self.model([tf.ones((1,64,64,3), tf.float32),tf.ones((1,64,64,3), tf.float32)], 
                              training=is_train_model, is_inference=False)
        
        if not weight_file is None:
            self.model.load_weights(weight_file, by_name=True)
            if not is_train_model:
                self.model.freeze_layers()
                self.model.trainable = False
                self.tf_model = tf.function(lambda x: self.model.quick_inference(x, training=False, both_flow=True))


    def train(self, train_dataset, val_dataset, save_dir, num_data, 
              learning_rate=0.002, n_epoch=150, batch_size=32, 
              ):
        if not self.is_train_model:
            raise ValueError("Model must be loaded as is_train_model=True")
        
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        
        lr_schedule = LearningRateScheduler(lrs_wrapper_cos(learning_rate, n_epoch, epoch_st=0))
        logger = CSVLogger(save_dir + 'log.csv')
        weight_file = "{epoch:02d}.hdf5"
        cp_callback = ModelCheckpoint(save_dir + weight_file, 
                                      monitor = 'val_loss', 
                                      save_weights_only = True,
                                      save_best_only = True,
                                      period = 4,
                                      verbose = 1)
        
        optim = Adam(lr=learning_rate, clipnorm=0.01)
        
        print("step per epoch", num_data[0]//(batch_size*FRAME_STEP), num_data[1]//batch_size)
        print("occulusion removal at", OCC_MASK_EPOCH, "epoch = ", (num_data[0]//(batch_size*FRAME_STEP))*OCC_MASK_EPOCH, "batch")
        loss_weights = {"rgb": 3.0, "census":0.0, "ssim": 3.0, "smooth": 1.5}
        self.model.compile(loss = loss_wrapper(loss_weights, 
                                               gammma=0.8, 
                                               batch_for_occ_mask=(num_data[0]//(batch_size*FRAME_STEP))*OCC_MASK_EPOCH),
                            clip_norm=0.01,
                            optimizer = optim,
                            )
        
        transforms_train = [
                      BrightnessContrast(p=0.8),
                      HueShift(p=0.5, min_offset=-0.15, max_offset=0.15),
                      ]
        transforms_val = [
                      ]

        train_transforms = Compose(transforms_train)
        val_transforms = Compose(transforms_val)

        self.hist = self.model.fit(get_dataset_raft(train_dataset, 
                                               batch_size=batch_size, 
                                               transforms=train_transforms,
                                               input_shape=self.input_shape,
                                               is_train=True,
                                               ), 
                    steps_per_epoch=num_data[0]//(batch_size*FRAME_STEP), 
                    epochs=n_epoch, 
                    validation_data=get_dataset_raft(val_dataset, 
                                               batch_size=batch_size, 
                                               transforms=train_transforms,
                                               input_shape=self.input_shape,
                                               is_train=False,
                                               ),
                    validation_steps=num_data[1]//batch_size,
                    #validation_freq=2,
                    callbacks=[lr_schedule, logger, cp_callback],
                    )
        print("Saving weights and results...")
        self.model.save_weights(save_dir + "final_weights.h5")
        csv_hist = save_dir + "hist.csv"
        pd.DataFrame(self.hist.history).to_csv(csv_hist, index=False)
        print("Done")
    
    def predict_from_file(self, inputs):
        
        inputs, model_inputs = self.preprocess_inputs(inputs)
        preds = self.model(model_inputs, training=False, is_inference=False)
        
        show_iters = [-1]
        for i in show_iters:
            rgb_1_re = preds[i]["rgb_1_re"][0].numpy()
            rgb_2_re = preds[i]["rgb_2_re"][0].numpy()
            """
            disparity_1 = 1./(preds[i]["flow_1"][0].numpy()+1e-7)
            disparity_1 = disparity_1 / disparity_1.max()
            disparity_2 = 1./(preds[i]["flow_2"][0].numpy()+1e-7)
            disparity_2 = disparity_2 / disparity_2.max()
            disparity_1low = preds[i]["flow_1_low_res"][0].numpy()
            disparity_2low = preds[i]["flow_2_low_res"][0].numpy()
            """
            # TODO 2D vector -> Color
            """
            flow_1 = preds[i]["flow_1"][0].numpy()
            flow_1 = flow_1 / (2 * np.max(np.abs(flow_1), axis=(1,2), keepdims=True)) + 0.5
            flow_1 = np.concatenate([flow_1, np.ones_like(flow_1[...,:1])], axis=-1)
            
            flow_2 = preds[i]["flow_2"][0].numpy()
            flow_2 = flow_2 / (2 * np.max(np.abs(flow_2), axis=(1,2), keepdims=True)) + 0.5
            flow_2 = np.concatenate([flow_2, np.ones_like(flow_2[...,:1])], axis=-1)
            """


            plt.imshow(flow_to_rgb(preds[i]["flow_1"][0].numpy()))
            plt.show()
            

            plt.imshow(inputs["frame_1"][0])
            plt.show()
            plt.imshow(rgb_1_re)
            plt.show()
            plt.imshow(rgb_2_re)
            plt.show()
            plt.imshow(inputs["frame_2"][0])
            plt.show()
            plt.imshow(flow_to_rgb(preds[i]["flow_2"][0].numpy()))
            plt.show()
            
            plt.imshow(np.minimum(np.abs(inputs["frame_1"][0].numpy()-rgb_1_re), 1.))
            plt.show()
            
            #plt.imshow(disparity_2)
            #plt.show()
            #plt.imshow(disparity_1low)
            #plt.show()
            #plt.imshow(disparity_2low)
            #plt.show()
    
    def predict(self, rgb_frames):
        inputs = self.preprocess_input_arrays(rgb_frames)
        outputs = self.tf_model(inputs)
        return outputs
            
    def predict_and_save_flow(self, dataset, 
                              save_low_res=True, 
                               original_folder="train_interval_images", 
                               save_folder="flow_01_320_568"):
        
        self.model.freeze_layers()
        self.model.trainable = False
        predict_bothflow = True
        batch_size = 4
        
        model_tf_func = tf.function(lambda x: self.model.quick_inference(x, training=False, both_flow=predict_bothflow),
                                    )
        #model_tf_func = tf.function(lambda x: self.model(x, training=False),
        #                            )                                             
        import time
        start_time = time.perf_counter()
        
        
        for i, (inp, files) in enumerate(get_dataset_raft_inference(dataset, 
                                                                    batch_size=batch_size, 
                                                                    input_shape=self.input_shape)):
        
            if i%10==0:
                elapsed = time.perf_counter() - start_time
                fps_inference = i*batch_size/elapsed
                print(f"\r{i*batch_size} / {len(dataset)} at {round(fps_inference)} fps", end="")
                gc.collect()
            save_files_1 = [f.decode().replace(original_folder, save_folder).replace(".jpg", "flow12.npy") for f in files["rgb_file_1"].numpy()]
            save_files_2 = [f.decode().replace(original_folder, save_folder).replace(".jpg", "flow21.npy") for f in files["rgb_file_2"].numpy()]
            os.makedirs(os.path.dirname(save_files_1[0]), exist_ok=True)
            os.makedirs(os.path.dirname(save_files_1[-1]), exist_ok=True)
            preds = model_tf_func([inp["frame_1"], inp["frame_2"]])
            
            for batch_idx, [save_1, save_2] in enumerate(zip(save_files_1, save_files_2)):
                if save_low_res:
                    flow_1 = preds[-1]["flow_1_low_res"][batch_idx].numpy()
                    if predict_bothflow:
                        flow_2 = preds[-1]["flow_2_low_res"][batch_idx].numpy()
                else:
                    flow_1 = preds[-1]["flow_1"][batch_idx].numpy()
                    if predict_bothflow:
                        flow_2 = -preds[-1]["flow_2"][batch_idx].numpy()
                np.save(save_1, flow_1)
                if predict_bothflow:
                    np.save(save_2, flow_2)#3.7sec for 50
                    
                
                
            if i%20==0:
                #rgb_1_re = preds[-1]["rgb_1_re"][batch_idx].numpy()
                #rgb_2_re = preds[-1]["rgb_2_re"][batch_idx].numpy()
                
                plt.imshow(flow_to_rgb(flow_1))
                plt.show()
                plt.imshow(inp["frame_1"][batch_idx])
                plt.show()
                #plt.imshow(rgb_1_re)
                #plt.show()
                if predict_bothflow:
                    plt.imshow(flow_to_rgb(flow_2))
                    plt.show()
                    plt.imshow(inp["frame_2"][batch_idx])
                    plt.show()
            
        
    def preprocess_inputs(self, inputs):
        #frame_1, frame_2, inf_DP
        ##height = 420#data["rgb_height"]
        ##width = 1000#data["rgb_width"]
        inputs["frame_1"] = tf.image.decode_jpeg(tf.io.read_file(inputs["rgb_file_1"]), channels=3)#[:height, :width, :]
        inputs["frame_2"] = tf.image.decode_jpeg(tf.io.read_file(inputs["rgb_file_2"]), channels=3)#[:height, :width, :]
        
        #inputs["frame_1"] = inputs["frame_1"][300:780,500:-500,:]
        #inputs["frame_2"] = inputs["frame_2"][300:780,500:-500,:]
        
        inputs["frame_1"] = tf.cast(inputs["frame_1"], tf.float32) / 255.
        inputs["frame_2"] = tf.cast(inputs["frame_2"], tf.float32) / 255.

        target_height = self.input_shape[0]
        target_width = self.input_shape[1]
    
        inputs["frame_1"]= tf.image.resize(inputs["frame_1"][tf.newaxis, ...], (target_height, target_width), method="bilinear")
        inputs["frame_2"]= tf.image.resize(inputs["frame_2"][tf.newaxis, ...], (target_height, target_width), method="bilinear")
        model_inputs = [inputs[key] for key in ["frame_1", "frame_2"]]
        
        return inputs, model_inputs
    
    @tf.function
    def preprocess_input_arrays(self, inputs):
        """
        inputs have the shape of [num_frames, height, width, 3] 
        batch size is num_frames-1
        -> frame_1 [num_frames-1, height, width, 3]
        -> frame_2 [num_frames-1, height, width, 3]

        """
        
        inputs = tf.cast(inputs, tf.float32) / 255.

        target_height = self.input_shape[0]
        target_width = self.input_shape[1]
    
        inputs = tf.image.resize(inputs, (target_height, target_width), method="bilinear")
        model_inputs = [inputs[:-1], inputs[1:]]
        
        return model_inputs
            
def set_seeds(num=111):
    tf.random.set_seed(num)
    np.random.seed(num)
    random.seed(num)
    os.environ["PYTHONHASHSEED"] = str(num)

        
def run_training_main(epochs=20, batch_size=8, base_lr=0.0001,
                      input_shape=(200,480, 3),
                      save_path="",
                      initial_weight=None):
    
    
    K.clear_session()
    set_seeds(111)

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
    train_files = load_dataset(end_path[:200] + side_path[:200], 
                               raft_model=True)
    val_files = load_dataset(end_path[-24:] + side_path[-24:], raft_model=True)[::FRAME_STEP_VAL]
    
    np.random.shuffle(train_files)
    np.random.shuffle(val_files)
    
    num_data = [len(train_files), len(val_files)]    
    
    print(num_data)

    model_params = {"input_shape": input_shape,
                    "weight_file": initial_weight,
                    "is_train_model": True,
                    "from_scratch": False,
                    }
    
    train_params = {"train_dataset": train_files,
                    "val_dataset": val_files,
                    "num_data": num_data,
                    "save_dir": save_path,
                    "learning_rate": base_lr, 
                    "n_epoch": epochs, 
                    "batch_size": batch_size,
                    }
    #with tf.device('/device:GPU:1'):
    net = OpticalFlowNet(**model_params)
    net.train(**train_params)  
        
def save_validation_predicts(weight_file, input_shape=(200,480, 3),):
    K.clear_session()
    set_seeds(111)

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
    train_files = load_dataset(end_path[:240] + side_path[:240], raft_model=True)
    val_files = load_dataset(end_path[-24:] + side_path[-24:], raft_model=True)
    
    #np.random.shuffle(train_files)
    #np.random.shuffle(val_files)
    
    num_data = [len(train_files), len(val_files)]    
    
    print(num_data)
    #raise Exception("not implemented")



    model_params = {"input_shape": input_shape,
                    "weight_file": weight_file,
                    "is_train_model": False,
                    "from_scratch": False,
                    }
    net = OpticalFlowNet(**model_params)
    net.predict_and_save_flow(train_files,
                              original_folder="train_img",
                              save_folder="train_flow_img_512x896")


if __name__=="__main__":
 
    FRAME_STEP = 40
    FRAME_STEP_VAL = 40
    
    setting_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"SETTINGS.json")
    DIRS = json.load(open(setting_file))
    BASE_DIR = DIRS["RAW_DATA_DIR"]
    TRAIN_DIR = DIRS["TRAIN_DATA_DIR"]
    WEIGHT_DIR = DIRS["WEIGHT_DIR"]# model/weights/
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    DATA_PATH = os.path.join(TRAIN_DIR, "train_img/")
    run_train = False
    if run_train:
        OCC_MASK_EPOCH = 20
        save_path = os.path.join(SETTING["WEIGHT_DIR"], "ex001_raft_run003_512x896/")
        run_training_main(epochs=30, batch_size=3,
                          ##input_shape=(360,544,3), 
                          input_shape=(512, 896, 3),
                          save_path=save_path,
                          initial_weight="model/weights/ex001_raft_run003_512x896/08.hdf5"
                          )
    else:
        save_validation_predicts(weight_file=os.path.join(SETTING["WEIGHT_DIR"], "ex001_raft_run002_512x896/final_weights.h5"), 
                                 input_shape=(512, 896,3),)
        

    
    