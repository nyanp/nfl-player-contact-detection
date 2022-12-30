# -*- coding: utf-8 -*-
"""
@author: kmat
Lerning rate scheduler
"""
import numpy as np

def lrs_wrapper_cos(learning_rate, epoch_1, epoch_st=5):    
    max_lr = learning_rate
    min_lr = learning_rate/100
    def lrs(epoch):
        if epoch<epoch_st:
            lr = (max_lr-min_lr)*epoch/epoch_st + min_lr
        elif epoch<epoch_1:
            angle = np.clip(np.pi * ((epoch-epoch_st)/(epoch_1-epoch_st)), 0, np.pi)
            lr = min_lr + 0.5 * (max_lr-min_lr) * (1 + np.cos(angle))                    
        else:
            lr=min_lr
        return lr
    return lrs

def lrs_wrapper_cos_annealing(learning_rate, epochs, epoch_wu=5):    
    max_lr = learning_rate
    min_lr = learning_rate/100
    key_epochs = np.array([0] + epochs)
    def lrs(epoch):
        epochs = epoch - key_epochs
        epochs = np.where(epochs>=0, epochs, np.inf)
        stage = np.argmin(epochs[:-1])
        current_stage_epoch = key_epochs[stage]
        next_stage_epoch = key_epochs[stage+1]
        epoch_dev = epoch-current_stage_epoch
        epoch_step = next_stage_epoch-current_stage_epoch
        if epoch_dev<epoch_wu:
            lr = (max_lr-min_lr)*epoch_dev/epoch_wu + min_lr
        elif epoch_dev<epoch_step:
            angle = np.clip(np.pi * ((epoch_dev-epoch_wu)/(epoch_step-epoch_wu)), 0, np.pi)
            lr = min_lr + 0.5 * (max_lr-min_lr) * (1 + np.cos(angle))                    
        else:
            lr=min_lr
        return lr
    return lrs

def lrs_wrapper(learning_rate, epoch_1, epoch_2, epoch_st=5):    
    def lrs(epoch):
        if epoch<epoch_st:
            lr = learning_rate*epoch/epoch_st+learning_rate/100
        elif epoch<epoch_1:
            lr=learning_rate
        elif epoch<epoch_2:
            lr=learning_rate/10
        else:
            lr=learning_rate/100
        return lr
    return lrs