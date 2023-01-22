# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:30:24 2022

@author: ras
"""


import os
import sys
import math
#import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv3D, Conv2D, Conv1D, Conv2DTranspose, BatchNormalization, Activation, GlobalAveragePooling2D, AveragePooling2D, Lambda, Input, Concatenate, Add, UpSampling2D, LeakyReLU, ZeroPadding2D,Multiply, DepthwiseConv2D, MaxPooling2D, LayerNormalization
from tensorflow.keras.models import Model

from .efficientnetv2 import effnetv2_model

sys.path.append('../')
WEIGHT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"weights/")
USE_TPU = False

if USE_TPU:
    batch_norm = tf.keras.layers.experimental.SyncBatchNormalization
else:
    batch_norm = BatchNormalization

def resblock(x, out_ch, kernel, stride, name, bias=True, use_se=True):
    inputs = x
    x = cbr(x, out_ch, kernel, stride, name+"_cbr0", bias)
    x = cbr(x, out_ch, kernel, 1, name+"_cbr1", bias)
    if use_se:
        x = se(x, out_ch, rate=4, name=name+"_se")
    #x_in = cbr(inputs, out_ch, 1, stride, name+"_cbr_shortcut", bias)
    x = Add()([x, inputs])
    return x

def cbr(x, out_layer, kernel, stride, name, bias=False, use_batchnorm=True):
    x = Conv2D(out_layer, kernel_size=kernel, strides=stride,use_bias=bias, padding="same", name=name+"_conv")(x)
    if use_batchnorm:
        x = batch_norm(name=name+"_bw")(x)
    else:
        raise Exception("tensorflow addons")
        #x = tfa.layers.GroupNormalization(name=name+"_bw")(x)
    x = Activation("relu",name=name+"_activation")(x)
    return x

def depth_cbr(x, kernel, stride, name, bias=True):#,use_bias=False
    x = DepthwiseConv2D(kernel_size=kernel, strides=stride,use_bias=bias,  padding="same", name=name+"_dwconv")(x)
    x = batch_norm(name=name+"_bw")(x)
    x = Activation("relu",name=name+"_activation")(x)
    return x

def cb(x, out_layer, kernel, stride, name, bias=True):
    x=Conv2D(out_layer, kernel_size=kernel, strides=stride,use_bias=bias,  padding="same", name=name+"_conv")(x)
    x = batch_norm(name=name+"_bw")(x)
    return x

def se(x_in, layer_n, rate, name):
    x = GlobalAveragePooling2D(name=name+"_squeeze")(x_in)
    x = Reshape((1,1,layer_n),name=name+"_reshape")(x)
    x = Conv2D(layer_n//rate, kernel_size=1,strides=1, name=name+"_reduce")(x)
    x= Activation("relu",name=name+"_relu")(x)
    x = Conv2D(layer_n, kernel_size=1,strides=1, name=name+"_expand")(x)
    x= Activation("sigmoid",name=name+"_sigmoid")(x)
    #x = Dense(layer_n, activation="sigmoid")(x)
    x_out=Multiply(name=name+"_excite")([x_in, x])
    return x_out

def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
    x_deep= Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
    x_deep = batch_norm()(x_deep)   
    x_deep = LeakyReLU(alpha=0.1)(x_deep)
    x = Concatenate()([x_shallow, x_deep])
    x = Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)
    x = batch_norm()(x)   
    x = LeakyReLU(alpha=0.1)(x)
    return x

def aggregation(skip_connections, output_layer_n, prefix=""):
    x_1= cbr(skip_connections["c1"], output_layer_n, 1, 1,prefix+"aggregation_1")
    x_1 = aggregation_block(x_1, skip_connections["c2"], output_layer_n, output_layer_n)
    x_2= cbr(skip_connections["c2"], output_layer_n, 1, 1,prefix+"aggregation_2")
    x_2 = aggregation_block(x_2, skip_connections["c3"], output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_3 = cbr(skip_connections["c3"], output_layer_n, 1, 1,prefix+"aggregation_3")
    x_3 = aggregation_block(x_3, skip_connections["c4"], output_layer_n, output_layer_n)
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_4 = cbr(skip_connections["c4"], output_layer_n, 1, 1,prefix+"aggregation_4")
    skip_connections_out=[x_1,x_2,x_3,x_4]
    return skip_connections_out

def effv2_encoder(inputs, is_train, from_scratch, model_name="s"):
    skip_connections={}
    pretrained_model = effnetv2_model.get_model('efficientnetv2-{}'.format(model_name), 
                                                model_config={"bn_type":"tpu_bn" if USE_TPU else None},
                                                include_top=False, 
                                                pretrained=False,
                                                training=is_train,
                                                input_shape=(None,None,3),
                                                input_tensor=inputs,
                                                with_endpoints=True)
    if not from_scratch:
        pretrained_model.load_weights(WEIGHT_DIR + 'effv2-{}-21k.h5'.format(model_name), by_name=True, skip_mismatch=True)    

    skip_connections["c1"] = pretrained_model.output[1]
    skip_connections["c2"] = pretrained_model.output[2]
    skip_connections["c3"] = pretrained_model.output[3]
    skip_connections["c4"] = pretrained_model.output[4]
    x = pretrained_model.output[5]

    return x, skip_connections


def decoder(inputs, skip_connections, use_batchnorm=True, 
            num_channels = 32, minimum_stride=2, max_stride=128,
            prefix=""):
    if not minimum_stride in [1,2,4,8]:
        raise Exception("minimum stride must be 1 or 2 or 4 or 8")
    if not max_stride in [32,64,128]:
        raise Exception("maximum stride must be 32 or 64 or 128")
    outs = []
    skip_connections = aggregation(skip_connections, num_channels, prefix=prefix)
    
    x = Dropout(0.2,noise_shape=(None, 1, 1, 1),name=prefix+'top_drop')(inputs)
    
    if max_stride>32:#more_deep        
        x_64 = cbr(x, 256, 3, 2,prefix+"top_64", use_batchnorm=use_batchnorm)
        if max_stride>64:
            x_128 = cbr(x_64, 256, 3, 2,prefix+"top_128", use_batchnorm=use_batchnorm)
            outs.append(x_128)
            x_64u = UpSampling2D(size=(2, 2))(x_128)
            x_64 = Concatenate()([x_64, x_64u])
        x_64 = cbr(x_64, 256, 3, 1,prefix+"top_64u", use_batchnorm=use_batchnorm)
        outs.append(x_64)
        x_32u = UpSampling2D(size=(2, 2))(x_64)
        x = Concatenate()([x, x_32u])    
    #x = Lambda(add_coords)(x)    
    x = cbr(x, num_channels*16, 3, 1,prefix+"decode_1", use_batchnorm=use_batchnorm)
    outs.append(x)
    x = UpSampling2D(size=(2, 2))(x)#8->16 tconvのがいいか

    x = Concatenate()([x, skip_connections[3]])
    x = cbr(x, num_channels*8, 3, 1,prefix+"decode_2", use_batchnorm=use_batchnorm)
    outs.append(x)
    x = UpSampling2D(size=(2, 2))(x)#16->32
    
    x = Concatenate()([x, skip_connections[2]])
    x = cbr(x, num_channels*4, 3, 1,prefix+"decode_3", use_batchnorm=use_batchnorm)
    outs.append(x)
   
    if minimum_stride<=4:
        x = UpSampling2D(size=(2, 2))(x)#32->64 
        x = Concatenate()([x, skip_connections[1]])
        x = cbr(x, num_channels*2, 3, 1,prefix+"decode_4", use_batchnorm=use_batchnorm)
        outs.append(x)
    if minimum_stride<=2:    
        x = UpSampling2D(size=(2, 2))(x)#64->128
        x = Concatenate()([x, skip_connections[0]])
        x = cbr(x, num_channels, 3, 1,prefix+"decode_5", use_batchnorm=use_batchnorm)
        outs.append(x)
    if minimum_stride==1:
        x = UpSampling2D(size=(2, 2))(x)#128->256
        outs.append(x)
    return outs

def add_high_freq_coords(inputs):
    
    
    batch_num, height, width = tf.unstack(tf.shape(inputs))[:3]
    
    h_grid = tf.expand_dims(tf.linspace(0., 5.0, height), 1) % 1.
    h_grid = 4 * tf.maximum(h_grid, 1. - h_grid) - 3. # -1 ro 1
    h_grid = tf.tile(h_grid, [1, width])
    w_grid = tf.expand_dims(tf.linspace(0., 5.0, width), 0) % 1.
    w_grid = 4 * tf.maximum(w_grid, 1. - w_grid) - 3.
    w_grid = tf.tile(w_grid, [height,1])
    hw_grid = tf.concat([tf.expand_dims(h_grid, -1),tf.expand_dims(w_grid, -1)], axis=-1)
    hw_grid = tf.expand_dims(hw_grid, 0)
    hw_grid = tf.tile(hw_grid, [batch_num, 1, 1, 1])
    
    return tf.concat([inputs, hw_grid], axis=-1)

def add_coords(inputs):
    batch_num, height, width = tf.unstack(tf.shape(inputs))[:3]
    
    h_grid = tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1)
    h_grid = tf.tile(h_grid, [1, width])
    w_grid = tf.expand_dims(tf.linspace(-1.0, 1.0, width), 0)
    w_grid = tf.tile(w_grid, [height,1])
    hw_grid = tf.concat([tf.expand_dims(h_grid, -1),tf.expand_dims(w_grid, -1)], axis=-1)
    hw_grid = tf.expand_dims(hw_grid, 0)
    hw_grid = tf.tile(hw_grid, [batch_num, 1, 1, 1])
    return tf.concat([inputs, hw_grid], axis=-1)

def add_bbox_img(inputs, only_overlap=True):
    
    imgs, boxes = inputs
    img_height, img_width = tf.unstack(tf.shape(imgs))[1:3]
    batch, num_box = tf.unstack(tf.shape(boxes))[:2]
    height_range = tf.range(img_height, dtype=tf.float32)/tf.cast(img_height, tf.float32)
    height_range = tf.tile(height_range[tf.newaxis,tf.newaxis,:],[batch, num_box, 1])
    width_range = tf.range(img_width, dtype=tf.float32)/tf.cast(img_width, tf.float32)
    width_range = tf.tile(width_range[tf.newaxis,tf.newaxis,:],[batch, num_box, 1])
    height_inside = tf.math.logical_and(height_range >= boxes[:,:,0:1], height_range < boxes[:,:,2:3])    
    width_inside = tf.math.logical_and(width_range >= boxes[:,:,1:2], width_range < boxes[:,:,3:4])
    # 重複あるとよくなさそう。どうする？
    boxes_mask = tf.einsum('bnh,bnw->bhw', 
                tf.cast(height_inside, tf.float32), 
                tf.cast(width_inside, tf.float32))
    if only_overlap:#if>=2, overlap
        boxes_mask = tf.cast((boxes_mask>1.0), tf.float32)
    
    return tf.concat([imgs, boxes_mask[...,tf.newaxis]], axis=-1)


def get_dev_overlap(inputs, multi_mask=True):
    # 重複対策。テンポラリ？
    all_box_mask = tf.maximum(inputs[...,-2:-1], 1.)
    crop_box_mask = inputs[...,-1:]
    other_box_mask  = all_box_mask - crop_box_mask
    other_box_mask = tf.where(other_box_mask<0.99, 0.0, 1.0)
    if multi_mask:
        return tf.concat([inputs[..., :-2], crop_box_mask, other_box_mask], axis=-1)
    else:
        return tf.concat([inputs[..., :-2], crop_box_mask], axis=-1) 

def crop_resize_layer(inputs, crop_size=[16,16], num_ch=1, unbatch=True):
    images, boxes = inputs
    batch, num_box, _ = tf.unstack(tf.shape(boxes))
    boxes = tf.reshape(boxes, [-1, 4])
    
    box_indices = tf.tile(tf.reshape(tf.range(batch),[-1,1]),[1,num_box])
    box_indices = tf.reshape(box_indices, [batch*num_box])
    crop_images = tf.image.crop_and_resize(images, boxes, box_indices, crop_size, method='bilinear')
    if unbatch:
        crop_images = tf.reshape(crop_images, [batch*num_box, crop_size[0], crop_size[1], num_ch])
    else:
        crop_images = tf.reshape(crop_images, [batch, num_box, crop_size[0], crop_size[1], num_ch])

    return crop_images

def larger_crop_resize_layer(inputs, crop_size=[24,16], num_ch=1, unbatch=True, add_crop_mask=False, wide_mode=False):
    images, boxes = inputs
    ##pad_size = 150//4
    img_height, img_width = tf.unstack(tf.shape(images))[1:3]
    img_height_f = tf.cast(img_height, tf.float32)
    img_width_f = tf.cast(img_width, tf.float32)
    batch, num_box = tf.unstack(tf.shape(boxes))[:2]
    #pad_img = np.pad(img, [(pad_size,pad_size),(pad_size,pad_size),(0,0)], 'edge')
    ##pad_img = tf.pad(img, [[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]], mode='CONSTANT', constant_values=0)
    top = boxes[:,:,0:1]
    left = boxes[:,:,1:2]
    bottom = boxes[:,:,2:3]
    right = boxes[:,:,3:4]
    width = right - left
    height = bottom - top
    cx = (left + right)/2
    cy = (top + bottom)/2
    
    # rateになってしまってるので注意
    # average_size = tf.reduce_mean(tf.math.sqrt(width*height), axis=1, keepdims=True)
    average_size = tf.reduce_mean(tf.math.sqrt((img_width_f*width) * (img_height_f*height)), axis=1, keepdims=True)
    average_size_h = average_size / img_height_f
    average_size_w = average_size / img_width_f


    # if out of range, crop_and_resize adds zero padding automatically.
    if wide_mode:
        lr_length = 4
        t_length = 3
        b_length = 6.0     
        #lr_length = 5.2
        #t_length = 3
        #b_length = 7.8 
    else:
        lr_length = 1.5#*2
        t_length = 1.5#*2
        b_length = 3.0#*2
    h_length = t_length + b_length
    w_length = lr_length * 2

    #if crop_size[0]!=(crop_size[1]*h_length/w_length):
    #    raise Exception("check aspect ratio is {}:{}".format(h_length, w_length))
    
    ts = (cy - average_size_h*t_length)
    ls = (cx - average_size_w*lr_length)
    bs = (cy + average_size_h*b_length)
    rs = (cx + average_size_w*lr_length)
    
    large_boxes = tf.concat([ts, ls, bs, rs], axis=-1)
    large_boxes = tf.reshape(large_boxes, [-1, 4])
    box_indices = tf.tile(tf.reshape(tf.range(batch),[-1,1]),[1,num_box])
    box_indices = tf.reshape(box_indices, [batch*num_box])
    
    crop_images = tf.image.crop_and_resize(images, large_boxes, box_indices, crop_size, method='bilinear')
    if unbatch:
        crop_images = tf.reshape(crop_images, [batch*num_box, crop_size[0], crop_size[1], num_ch])
    else:
        crop_images = tf.reshape(crop_images, [batch, num_box, crop_size[0], crop_size[1], num_ch])
        
    if add_crop_mask:
        # batch, box, 1
        top_aft_resize = ((top - cy)/(h_length*average_size_h)) + t_length/h_length 
        left_aft_resize = ((left - cx)/(w_length*average_size_w)) + lr_length/w_length
        bottom_aft_resize = ((bottom - cy)/(h_length*average_size_h)) + t_length/h_length
        right_aft_resize = ((right - cx)/(w_length*average_size_w)) + lr_length/w_length
        
        y_coord = tf.tile(tf.linspace(0.0, 1.0, crop_size[0])[tf.newaxis,tf.newaxis,:], [batch,num_box,1])
        x_coord = tf.tile(tf.linspace(0.0, 1.0, crop_size[1])[tf.newaxis,tf.newaxis,:], [batch,num_box,1])
        
        y_inside = tf.cast(tf.math.logical_and(top_aft_resize<y_coord, y_coord<bottom_aft_resize), tf.float32)
        x_inside = tf.cast(tf.math.logical_and(left_aft_resize<x_coord, x_coord<right_aft_resize), tf.float32)
        
        box_mask_imgs = tf.einsum('bny,bnx->bnyx', y_inside, x_inside)
        if unbatch:
            box_mask_imgs = tf.reshape(box_mask_imgs, [batch*num_box, crop_size[0], crop_size[1], 1])
        else:
            box_mask_imgs = tf.reshape(box_mask_imgs, [batch, num_box, crop_size[0], crop_size[1], 1])
        crop_images = tf.concat([crop_images, box_mask_imgs], axis=-1)
        
    return crop_images

def larger_crop_resize_layer_2(inputs, crop_size=[24,16], num_ch=1, unbatch=True, add_crop_mask=False, wide_mode=False):
    images, boxes = inputs
    ##pad_size = 150//4
    img_height, img_width = tf.unstack(tf.shape(images))[1:3]
    img_height_f = tf.cast(img_height, tf.float32)
    img_width_f = tf.cast(img_width, tf.float32)
    batch, num_box = tf.unstack(tf.shape(boxes))[:2]
    #pad_img = np.pad(img, [(pad_size,pad_size),(pad_size,pad_size),(0,0)], 'edge')
    ##pad_img = tf.pad(img, [[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]], mode='CONSTANT', constant_values=0)
    top = boxes[:,:,0:1]
    left = boxes[:,:,1:2]
    bottom = boxes[:,:,2:3]
    right = boxes[:,:,3:4]
    width = right - left
    height = bottom - top
    cx = (left + right)/2
    cy = (top + bottom)/2
    
    # rateになってしまってるので注意
    # average_size = tf.reduce_mean(tf.math.sqrt(width*height), axis=1, keepdims=True)
    average_size = tf.reduce_mean(tf.math.sqrt((img_width_f*width) * (img_height_f*height)), axis=1, keepdims=True)
    average_size_h = average_size / img_height_f
    average_size_w = average_size / img_width_f


    # if out of range, crop_and_resize adds zero padding automatically.
    if wide_mode:
        lr_length = 4
        t_length = 3
        b_length = 6.0     
    else:
        lr_length = 1.5#*2
        t_length = 1.5#*2
        b_length = 3.0#*2
    h_length = t_length + b_length
    w_length = lr_length * 2

    #if crop_size[0]!=(crop_size[1]*h_length/w_length):
    #    raise Exception("check aspect ratio is {}:{}".format(h_length, w_length))
    
    ts = (cy - average_size_h*t_length)
    ls = (cx - average_size_w*lr_length)
    bs = (cy + average_size_h*b_length)
    rs = (cx + average_size_w*lr_length)
    
    large_boxes = tf.concat([ts, ls, bs, rs], axis=-1)
    large_boxes = tf.reshape(large_boxes, [-1, 4])
    box_indices = tf.tile(tf.reshape(tf.range(batch),[-1,1]),[1,num_box])
    box_indices = tf.reshape(box_indices, [batch*num_box])
    
    crop_images = tf.image.crop_and_resize(images, large_boxes, box_indices, crop_size, method='bilinear')
    if unbatch:
        crop_images = tf.reshape(crop_images, [batch*num_box, crop_size[0], crop_size[1], num_ch])
    else:
        crop_images = tf.reshape(crop_images, [batch, num_box, crop_size[0], crop_size[1], num_ch])
        
    if add_crop_mask:
        # batch, box, 1
        top_aft_resize = ((top - cy)/(h_length*average_size_h)) + t_length/h_length 
        left_aft_resize = ((left - cx)/(w_length*average_size_w)) + lr_length/w_length
        bottom_aft_resize = ((bottom - cy)/(h_length*average_size_h)) + t_length/h_length
        right_aft_resize = ((right - cx)/(w_length*average_size_w)) + lr_length/w_length
        
        y_coord = tf.tile(tf.linspace(0.0, 1.0, crop_size[0])[tf.newaxis,tf.newaxis,:], [batch,num_box,1])
        x_coord = tf.tile(tf.linspace(0.0, 1.0, crop_size[1])[tf.newaxis,tf.newaxis,:], [batch,num_box,1])
        
        y_inside = tf.cast(tf.math.logical_and(top_aft_resize<y_coord, y_coord<bottom_aft_resize), tf.float32)
        x_inside = tf.cast(tf.math.logical_and(left_aft_resize<x_coord, x_coord<right_aft_resize), tf.float32)
        
        ### 注意！あえて縦横残す
        # box_mask_imgs = tf.einsum('bny,bnx->bnyx', y_inside, x_inside)
        box_mask_imgs = y_inside[:,:,:,tf.newaxis] +  x_inside[:,:,tf.newaxis,:] # tf.einsum('bny,bnx->bnyx', y_inside, x_inside)
        if unbatch:
            box_mask_imgs = tf.reshape(box_mask_imgs, [batch*num_box, crop_size[0], crop_size[1], 1])
        else:
            box_mask_imgs = tf.reshape(box_mask_imgs, [batch, num_box, crop_size[0], crop_size[1], 1])
        crop_images = tf.concat([crop_images, box_mask_imgs], axis=-1)
    large_boxes = tf.reshape(large_boxes, [batch, num_box, 4])
    return (crop_images, large_boxes)

def inv_larger_crop_resize_layer(inputs, num_ch=1, wide_mode=False):
    images, boxes, ref_img = inputs
    ##pad_size = 150//4
    img_height, img_width = tf.unstack(tf.shape(ref_img))[1:3]
    img_height_f = tf.cast(img_height, tf.float32)
    img_width_f = tf.cast(img_width, tf.float32)
    
    batch, num_box = tf.unstack(tf.shape(boxes))[:2]
    #pad_img = np.pad(img, [(pad_size,pad_size),(pad_size,pad_size),(0,0)], 'edge')
    ##pad_img = tf.pad(img, [[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]], mode='CONSTANT', constant_values=0)
    top = boxes[:,:,0:1]
    left = boxes[:,:,1:2]
    bottom = boxes[:,:,2:3]
    right = boxes[:,:,3:4]
    width = right - left
    height = bottom - top
    cx = (left + right)/2
    cy = (top + bottom)/2
    
    # rateになってしまってるので注意
    # average_size = tf.reduce_mean(tf.math.sqrt(width*height), axis=1, keepdims=True)
    average_size = tf.reduce_mean(tf.math.sqrt((img_width_f*width) * (img_height_f*height)), axis=1, keepdims=True)
    average_size_h = average_size / img_height_f
    average_size_w = average_size / img_width_f

    # if out of range, crop_and_resize adds zero padding automatically.
    if wide_mode:
        lr_length = 4
        t_length = 3
        b_length = 6.0
        
        #lr_length = 5.2
        #t_length = 3
        #b_length = 7.8 
    else:
        lr_length = 1.5#*2
        t_length = 1.5#*2
        b_length = 3.0#*2
    h_length = t_length + b_length
    w_length = lr_length * 2

    #if crop_size[0]!=(crop_size[1]*h_length/w_length):
    #    raise Exception("check aspect ratio is {}:{}".format(h_length, w_length))
    
    ts = (cy - average_size_h*t_length)
    ls = (cx - average_size_w*lr_length)
    bs = (cy + average_size_h*b_length)
    rs = (cx + average_size_w*lr_length)
    
    large_boxes = tf.concat([ts, ls, bs, rs], axis=-1)
    large_boxes = tf.reshape(large_boxes, [-1, 4])
    box_indices = tf.tile(tf.reshape(tf.range(batch),[-1,1]),[1,num_box])
    box_indices = tf.reshape(box_indices, [batch*num_box])
    
    h_scale = 1. / (large_boxes[:,2:3] - large_boxes[:,0:1])
    w_scale = 1. / (large_boxes[:,3:4] - large_boxes[:,1:2])
    inv_tops = -large_boxes[:,:1] * h_scale
    inv_lefts = -large_boxes[:,1:2] * w_scale
    inv_bottoms = 1. + (1. - large_boxes[:,2:3]) * h_scale
    inv_rights = 1. + (1. - large_boxes[:,3:4]) * w_scale
    inv_boxes = tf.concat([inv_tops, inv_lefts, inv_bottoms, inv_rights], axis=1)
    
    reconst_size = [img_height, img_width]
    reconst_images = tf.image.crop_and_resize(images, inv_boxes, box_indices, reconst_size, method='bilinear')
    reconst_images = tf.reshape(reconst_images, [batch, num_box, img_height, img_width, num_ch])
        
    return reconst_images


def contact_btw_selected_pairs(inputs, reduced=False):
    """
    TODO
    ★同じピクセルでA-Bはコンタクト、B-Cはコンタクトのときに、A-Cはコンタクトでない場合もある。この場合を表現できるかな？
    ★コンタクトが全１でも成立しちゃうな。まぁええけど正則化いれてマスクを最小限出すようにしてもいいかも？。
    
    contact_map:
        [batch, height, width, 1]
        it include all contact event
    instance_masks:
        [batch, num_box+1, height, width, 1]
        initial box index([:,0,:,:,:]) is ground mask. others [:,1:,:,:,:] are players' mask 
    pairs:
        [batch, num_pairs, 2]
        values are index of player(bbox). 0 <= val < num_player+1.
    reduced:
        bool. 
        if True:
            return [batch, num_pairs]
        else:
            return [batch, num_pairs, height, width, 1]
        the latter can be helpful to find the contact point. 
        (useful for second stage & benefit for user)
    
    contact_btw_pairs:
        [batch, num_pairs] or [batch, num_pairs, height, width, 1]
    """
    contact_map, instance_masks, pairs = inputs
    player_mask_1 = tf.gather(instance_masks, pairs[:,:,0], axis=1, batch_dims=1)
    player_mask_2 = tf.gather(instance_masks, pairs[:,:,1], axis=1, batch_dims=1)
    # originalはひとつだけ
    # contact_btw_players = contact_map[:,tf.newaxis,:,:,:] * player_mask_1 * player_mask_2
    is_ground = tf.cast(pairs[:,:,1]==0, tf.float32)[:,:,tf.newaxis,tf.newaxis,tf.newaxis] # ground
    contact_btw_players = contact_map[:,tf.newaxis,:,:,:] * player_mask_1 * player_mask_2
    contact_w_ground = player_mask_1 * player_mask_2
    contact_btw_players = contact_btw_players * (1.-is_ground) + contact_w_ground * is_ground
    if reduced:
        return tf.math.reduce_max(contact_btw_players, axis=[2,3,4])
    else:
        return contact_btw_players

def contact_btw_selected_pairs_depth(inputs, reduced=False):
    """
    TODO
    ★同じピクセルでA-Bはコンタクト、B-Cはコンタクトのときに、A-Cはコンタクトでない場合もある。この場合を表現できるかな？
    ★コンタクトが全１でも成立しちゃうな。まぁええけど正則化いれてマスクを最小限出すようにしてもいいかも？。
    
    contact_map:
        [batch, height, width, 1]
        it include all contact event
    instance_masks:
        [batch, num_box+1, height, width, 1]
        initial box index([:,0,:,:,:]) is ground mask. others [:,1:,:,:,:] are players' mask 
    pairs:
        [batch, num_pairs, 2]
        values are index of player(bbox). 0 <= val < num_player+1.
    reduced:
        bool. 
        if True:
            return [batch, num_pairs]
        else:
            return [batch, num_pairs, height, width, 1]
        the latter can be helpful to find the contact point. 
        (useful for second stage & benefit for user)
    
    contact_btw_pairs:
        [batch, num_pairs] or [batch, num_pairs, height, width, 1]
    """
    depth_minmax, pairs = inputs
    depth_minmax_1 = tf.gather(depth_minmax, pairs[:,:,0], axis=1, batch_dims=1)
    depth_minmax_2 = tf.gather(depth_minmax, pairs[:,:,1], axis=1, batch_dims=1)
    
    depth_minmax_1_invalid = tf.reduce_all(depth_minmax_1==0, axis=-1, keepdims=True)
    depth_minmax_2_invalid = tf.reduce_all(depth_minmax_2==0, axis=-1, keepdims=True)
    depth_minmax_invalid = tf.cast(tf.logical_or(depth_minmax_1_invalid, depth_minmax_2_invalid), tf.float32)
    # originalはひとつだけ
    # contact_btw_players = contact_map[:,tf.newaxis,:,:,:] * player_mask_1 * player_mask_2
    is_ground = tf.cast(pairs[:,:,1]==0, tf.float32)[:,:,tf.newaxis,tf.newaxis,tf.newaxis] # ground
    #positive_value if penetrated
    penetration_btw_players = tf.minimum(depth_minmax_1[...,1:2] - depth_minmax_2[...,:1], 
                                         depth_minmax_2[...,1:2] - depth_minmax_1[...,:1]) - depth_minmax_invalid * 1e7
    penetration_btw_players = penetration_btw_players * (1.-is_ground)
    if reduced:
        return tf.math.reduce_max(penetration_btw_players, axis=[2,3,4])
    else:
        return penetration_btw_players

def extract_peak_features(inputs, num_ch=32):
    """
    batch, num_pair, h, w, 1
    batch, num_player+1, h, w, features
    batch, num_pair, 2
    """
    pairs_contact, player_features, pairs = inputs
    batch, num_pair, h, w, _ = tf.unstack(tf.shape(pairs_contact))
    batch, num_player, h, w, num_features = tf.unstack(tf.shape(player_features))
    player_features_1 = tf.gather(player_features, pairs[:,:,0], axis=1, batch_dims=1)
    player_features_2 = tf.gather(player_features, pairs[:,:,1], axis=1, batch_dims=1)
    
    argmax = tf.argmax(tf.reshape(pairs_contact, [batch, num_pair, h*w, 1]), axis=2)
    player_features_1 = tf.reshape(player_features_1, [batch, num_pair, h*w, num_features])
    player_features_2 = tf.reshape(player_features_2, [batch, num_pair, h*w, num_features])
    player_features_1 = tf.gather(player_features_1, argmax, batch_dims=2)
    player_features_2 = tf.gather(player_features_2, argmax, batch_dims=2)
    
    features = tf.concat([player_features_1, player_features_2], axis=-1)
    #features = tf.reshape(features, [batch, num_pair, 2*num_ch])
    #features = tf.reshape(player_features_1-player_features_2, [batch, num_pair, num_ch])
    return features
                       
def extract_xyd_features(inputs):
    
    xy_pos, pairs = inputs
    b, p, _ = tf.unstack(tf.shape(pairs))
    xy_pos_1 = tf.gather(xy_pos, pairs[:,:,0], axis=1, batch_dims=1)
    xy_pos_2 = tf.gather(xy_pos, pairs[:,:,1], axis=1, batch_dims=1)
    xy_diff = xy_pos_1 - xy_pos_2
    dist = tf.math.sqrt(tf.reduce_sum((xy_diff)**2, axis=-1, keepdims=True))
    outputs = tf.concat([xy_diff, dist], axis=-1)
    outputs = tf.reshape(outputs, [b,p,3])
    return outputs

def extract_pair_features(inputs, num_ch=3):
    
    player_feat, pairs = inputs
    b, p, _ = tf.unstack(tf.shape(pairs))
    player_feat_1 = tf.gather(player_feat, pairs[:,:,0], axis=1, batch_dims=1)
    player_feat_2 = tf.gather(player_feat, pairs[:,:,1], axis=1, batch_dims=1)
    outputs = tf.concat([player_feat_1, player_feat_2], axis=-1)
    outputs = tf.reshape(outputs, [b,p,num_ch*2])
    return outputs

def extract_pair_features_single(inputs, num_ch=3, only_1=False):
    
    features, pairs = inputs
    b, p, _ = tf.unstack(tf.shape(pairs))
    features_1 = tf.gather(features, pairs[:,:,0], axis=1, batch_dims=1)
    features_2 = tf.gather(features, pairs[:,:,1], axis=1, batch_dims=1)
    if only_1:
        outputs = features_1
        outputs = tf.reshape(outputs, [b,p,num_ch])
    else:
        outputs = tf.concat([features_1, features_2], axis=-1)
        outputs = tf.reshape(outputs, [b,p,num_ch*2])
    return outputs


class PenetrationLayer(tf.keras.layers.Layer):
    def __init__(self, max_thickness):
        super(PenetrationLayer, self).__init__()
        self.max_thickness = max_thickness

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                  shape=())
        self.bias = self.add_weight("bias",
                                  shape=())

    def call(self, inputs):
        #bias = tf.clip_by_value(self.bias, -0.5, 0.5)
        kernel = 10**tf.clip_by_value(self.kernel, 0.0, 0.5)
        return tf.sigmoid(0.1 + kernel * tf.clip_by_value(inputs, -self.max_thickness*5, self.max_thickness) / self.max_thickness)

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, activation="relu", *args, **kargs):
    super(MyDenseLayer, self).__init__(*args, **kargs)
    self.num_outputs = num_outputs
    self.activation_name = activation

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  initializer="ones",#tf.keras.initializers.GlorotNormal(),
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    self.bias = self.add_weight("bias",
                                #initializer = tf.keras.initializers.Zeros(),
                                #  initializer=tf.keras.initializers.GlorotNormal(),
                                  shape=[self.num_outputs])
    self.activation = tf.sigmoid if self.activation_name=="sigmoid" else tf.nn.relu

  def call(self, inputs):
    return self.activation(tf.matmul(inputs, self.kernel)+self.bias)

def contact_btw_selected_pairs_logit(inputs, reduced=False):
    """
    TODO
    ★同じピクセルでA-Bはコンタクト、B-Cはコンタクトのときに、A-Cはコンタクトでない場合もある。この場合を表現できるかな？
    ★コンタクトが全１でも成立しちゃうな。まぁええけど正則化いれてマスクを最小限出すようにしてもいいかも？。
    
    contact_map:
        [batch, height, width, 1]
        it include all contact event
    instance_masks:
        [batch, num_box+1, height, width, num_feature]
        initial box index([:,0,:,:,:]) is ground mask. others [:,1:,:,:,:] are players' mask 
    pairs:
        [batch, num_pairs, 2]
        values are index of player(bbox). 0 <= val < num_player+1.
    reduced:
        bool. 
        if True:
            return [batch, num_pairs]
        else:
            return [batch, num_pairs, height, width, 1]
        the latter can be helpful to find the contact point. 
        (useful for second stage & benefit for user)
    
    contact_btw_pairs:
        [batch, num_pairs] or [batch, num_pairs, height, width, 1]
    """
    contact_map, instance_masks, pairs = inputs
    player_mask_1 = tf.gather(instance_masks, pairs[:,:,0], axis=1, batch_dims=1)
    player_mask_2 = tf.gather(instance_masks, pairs[:,:,1], axis=1, batch_dims=1)
    # originalはひとつだけ
    # contact_btw_players = contact_map[:,tf.newaxis,:,:,:] * player_mask_1 * player_mask_2
    is_ground = tf.cast(pairs[:,:,1]==0, tf.float32)[:,:,tf.newaxis,tf.newaxis,tf.newaxis] # ground
    cross_section = tf.cast(tf.reduce_all(tf.concat([player_mask_1, player_mask_2], axis=-1)!=0., axis=-1, keepdims=True), tf.float32) 
    contact_btw_players = contact_map[:,tf.newaxis,:,:,:] * tf.math.sigmoid(tf.reduce_sum(player_mask_1 * player_mask_2, axis=-1, keepdims=True)) * cross_section
    contact_w_ground = tf.math.sigmoid(tf.reduce_sum(player_mask_1 * player_mask_2, axis=-1, keepdims=True)) * cross_section
    contact_btw_players = contact_btw_players * (1.-is_ground) + contact_w_ground * is_ground
    if reduced:
        return tf.math.reduce_max(contact_btw_players, axis=[2,3,4])
    else:
        return contact_btw_players

def contact_btw_selected_pairs_each(inputs, reduced=False):
    """
    TODO
    ★同じピクセルでA-Bはコンタクト、B-Cはコンタクトのときに、A-Cはコンタクトでない場合もある。この場合を表現できるかな？
    ★コンタクトが全１でも成立しちゃうな。まぁええけど正則化いれてマスクを最小限出すようにしてもいいかも？。
    
    contact_map:
        [batch, height, width, 1]
        it include all contact event
    instance_masks:
        [batch, num_box+1, height, width, 1]
        initial box index([:,0,:,:,:]) is ground mask. others [:,1:,:,:,:] are players' mask 
    pairs:
        [batch, num_pairs, 2]
        values are index of player(bbox). 0 <= val < num_player+1.
    reduced:
        bool. 
        if True:
            return [batch, num_pairs]
        else:
            return [batch, num_pairs, height, width, 1]
        the latter can be helpful to find the contact point. 
        (useful for second stage & benefit for user)
    
    contact_btw_pairs:
        [batch, num_pairs] or [batch, num_pairs, height, width, 1]
    """
    all_masks_p, all_masks_g, pairs = inputs
    player_mask_0 = all_masks_p[:,0:1]
    player_mask_1 = tf.gather(all_masks_p, pairs[:,:,0], axis=1, batch_dims=1)
    player_mask_2 = tf.gather(all_masks_p, pairs[:,:,1], axis=1, batch_dims=1)
    
    ground_mask_0 = all_masks_g[:,0:1]
    ground_mask_1 = tf.gather(all_masks_g, pairs[:,:,0], axis=1, batch_dims=1)
    
    player_contact_conf = player_mask_0 * player_mask_1 * player_mask_2
    ground_contact_conf = ground_mask_0 * ground_mask_1
    
    # originalはひとつだけ
    # contact_btw_players = contact_map[:,tf.newaxis,:,:,:] * player_mask_1 * player_mask_2
    is_ground = tf.cast(pairs[:,:,1]==0, tf.float32)[:,:,tf.newaxis,tf.newaxis,tf.newaxis] # ground
    #contact_btw_players = contact_map[:,tf.newaxis,:,:,:] * player_mask_1 * player_mask_2
    #contact_w_ground = player_mask_1 * player_mask_2
    
    contact_conf = player_contact_conf * (1.-is_ground) + ground_contact_conf * is_ground
    if reduced:
        return tf.math.reduce_max(contact_conf, axis=[2,3,4])
    else:
        return contact_conf
    
def contact_btw_selected_pairs_uraomote(inputs, reduced=False):
    """
    TODO
    ★同じピクセルでA-Bはコンタクト、B-Cはコンタクトのときに、A-Cはコンタクトでない場合もある。この場合を表現できるかな？
    ★コンタクトが全１でも成立しちゃうな。まぁええけど正則化いれてマスクを最小限出すようにしてもいいかも？。
    
    contact_map:
        [batch, height, width, 1]
        it include all contact event
    instance_masks:
        [batch, num_box+1, height, width, 1]
        initial box index([:,0,:,:,:]) is ground mask. others [:,1:,:,:,:] are players' mask 
    pairs:
        [batch, num_pairs, 2]
        values are index of player(bbox). 0 <= val < num_player+1.
    reduced:
        bool. 
        if True:
            return [batch, num_pairs]
        else:
            return [batch, num_pairs, height, width, 1]
        the latter can be helpful to find the contact point. 
        (useful for second stage & benefit for user)
    
    contact_btw_pairs:
        [batch, num_pairs] or [batch, num_pairs, height, width, 1]
    """
    contact_map, instance_masks, pairs = inputs
    player_mask_1 = tf.gather(instance_masks, pairs[:,:,0], axis=1, batch_dims=1)
    player_mask_2 = tf.gather(instance_masks, pairs[:,:,1], axis=1, batch_dims=1)
    contact_btw_players = contact_map[:,tf.newaxis,:,:,:] * player_mask_1[...,::-1] * player_mask_2
    contact_btw_players = tf.math.reduce_max(contact_btw_players, axis=-1, keepdims=True)
    if reduced:
        return tf.math.reduce_max(contact_btw_players, axis=[2,3,4])
    else:
        return contact_btw_players    
    
    
def contact_btw_selected_pairs_v2(inputs, reduced=False):
        """
        TODO
        ★同じピクセルでA-Bはコンタクト、B-Cはコンタクトのときに、A-Cはコンタクトでない場合もある。この場合を表現できるかな？
        ★コンタクトが全１でも成立しちゃうな。まぁええけど正則化いれてマスクを最小限出すようにしてもいいかも？。
        
        contact_map:
            [batch, height, width, 1]
            it include all contact event
        instance_masks:
            [batch, num_box*2, height, width, 1]
            initial box index([:,0,:,:,:]) is ground mask. others [:,1:,:,:,:] are players' mask 
        pairs:
            [batch, num_pairs, 2]
            values are index of player(bbox). 0 <= val < num_player+1.
        reduced:
            bool. 
            if True:
                return [batch, num_pairs]
            else:
                return [batch, num_pairs, height, width, 1]
            the latter can be helpful to find the contact point. 
            (useful for second stage & benefit for user)
        
        contact_btw_pairs:
            [batch, num_pairs] or [batch, num_pairs, height, width, 1]
        """
        contact_map, instance_masks, pairs = inputs
        
        num_players = tf.shape(instance_masks)[1] // 2
        
        is_ground_contact = tf.cast(pairs[:,:,1]==0, tf.int32)
        is_not_ground_contact = 1 - is_ground_contact
        
        p_0 = (pairs[:,:,0] - 1) + num_players
        p_1 = (pairs[:,:,1] - 1 * is_not_ground_contact) + num_players * is_not_ground_contact + (pairs[:,:,0] - 1) * is_ground_contact
        
        player_mask_1 = tf.gather(instance_masks, p_0, axis=1, batch_dims=1)
        player_mask_2 = tf.gather(instance_masks, p_1, axis=1, batch_dims=1)
        
        contact_btw_players = contact_map[:,tf.newaxis,:,:,:] * player_mask_1 * player_mask_2
        if reduced:
            return tf.math.reduce_max(contact_btw_players, axis=[2,3,4])
        else:
            return contact_btw_players


def contact_btw_selected_pairs_nomask(inputs, reduced=False):
    """
    TODO
    ★同じピクセルでA-Bはコンタクト、B-Cはコンタクトのときに、A-Cはコンタクトでない場合もある。この場合を表現できるかな？
    ★コンタクトが全１でも成立しちゃうな。まぁええけど正則化いれてマスクを最小限出すようにしてもいいかも？。
    
    contact_map:
        [batch, height, width, 1]
        it include all contact event
    instance_masks:
        [batch, num_box+1, height, width, 1]
        initial box index([:,0,:,:,:]) is ground mask. others [:,1:,:,:,:] are players' mask 
    pairs:
        [batch, num_pairs, 2]
        values are index of player(bbox). 0 <= val < num_player+1.
    reduced:
        bool. 
        if True:
            return [batch, num_pairs]
        else:
            return [batch, num_pairs, height, width, 1]
        the latter can be helpful to find the contact point. 
        (useful for second stage & benefit for user)
    
    contact_btw_pairs:
        [batch, num_pairs] or [batch, num_pairs, height, width, 1]
    """
    instance_masks, pairs = inputs
    player_mask_1 = tf.gather(instance_masks, pairs[:,:,0], axis=1, batch_dims=1)
    player_mask_2 = tf.gather(instance_masks, pairs[:,:,1], axis=1, batch_dims=1)
    contact_btw_players = tf.concat([player_mask_1, player_mask_2], axis=-1)
    return contact_btw_players

def contact_btw_selected_pairs_feature_only_exist(inputs, reduced=False):
    """
    TODO
    ★同じピクセルでA-Bはコンタクト、B-Cはコンタクトのときに、A-Cはコンタクトでない場合もある。この場合を表現できるかな？
    ★コンタクトが全１でも成立しちゃうな。まぁええけど正則化いれてマスクを最小限出すようにしてもいいかも？。
    
    contact_map:
        [batch, height, width, 1]
        it include all contact event
    instance_masks:
        [batch, num_box+1, height, width, 1]
        initial box index([:,0,:,:,:]) is ground mask. others [:,1:,:,:,:] are players' mask 
    pairs:
        [batch, num_pairs, 2]
        values are index of player(bbox). 0 <= val < num_player+1.
    reduced:
        bool. 
        if True:
            return [batch, num_pairs]
        else:
            return [batch, num_pairs, height, width, 1]
        the latter can be helpful to find the contact point. 
        (useful for second stage & benefit for user)
    
    contact_btw_pairs:
        [batch, num_pairs] or [batch, num_pairs, height, width, 1]
    """
    instance_masks, pairs = inputs
    player_1 = tf.gather(instance_masks, pairs[:,:,0], axis=1, batch_dims=1)
    player_1_feature = player_1[...,:-1]
    player_1_mask = player_1[...,-1:]
    player_2 = tf.gather(instance_masks, pairs[:,:,1], axis=1, batch_dims=1)
    player_2_feature = player_2[...,:-1]
    player_2_mask = player_2[...,-1:]
    player_12_mask = player_1_mask * player_2_mask
    feature_btw_players = tf.concat([player_1_feature, player_2_feature], axis=-1)
    return feature_btw_players, player_12_mask

def contact_btw_selected_pairs_nomask_v2(inputs, reduced=False):
    """
    TODO
    ★同じピクセルでA-Bはコンタクト、B-Cはコンタクトのときに、A-Cはコンタクトでない場合もある。この場合を表現できるかな？
    ★コンタクトが全１でも成立しちゃうな。まぁええけど正則化いれてマスクを最小限出すようにしてもいいかも？。
    
    contact_map:
        [batch, height, width, 1]
        it include all contact event
    instance_masks:
        [batch, num_box+1, height, width, 1]
        initial box index([:,0,:,:,:]) is ground mask. others [:,1:,:,:,:] are players' mask 
    pairs:
        [batch, num_pairs, 2]
        values are index of player(bbox). 0 <= val < num_player+1.
    reduced:
        bool. 
        if True:
            return [batch, num_pairs]
        else:
            return [batch, num_pairs, height, width, 1]
        the latter can be helpful to find the contact point. 
        (useful for second stage & benefit for user)
    
    contact_btw_pairs:
        [batch, num_pairs] or [batch, num_pairs, height, width, 1]
    """
    instance_masks, pairs = inputs
    num_players = tf.shape(instance_masks)[1] // 2
    
    is_ground_contact = tf.cast(pairs[:,:,1]==0, tf.int32)
    is_not_ground_contact = 1 - is_ground_contact
    
    p_0 = (pairs[:,:,0] - 1) + num_players
    p_1 = (pairs[:,:,1] - 1 * is_not_ground_contact) + num_players * is_not_ground_contact + (pairs[:,:,0] - 1) * is_ground_contact
    
    player_mask_1 = tf.gather(instance_masks, p_0, axis=1, batch_dims=1)
    player_mask_2 = tf.gather(instance_masks, p_1, axis=1, batch_dims=1)
    contact_btw_players = tf.concat([player_mask_1, player_mask_2], axis=-1)
    return contact_btw_players

def l2_regularization(y_true, y_pred):
    loss = tf.reduce_mean(y_pred**2)
    return loss

def bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    # minus value is invalid label. (just to control the number of labels constant)
    mask = tf.cast(y_true>=-1e-7, tf.float32)
    y_true = y_true * mask
    y_pred = y_pred * mask
    
    epsilon = K.epsilon()    
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    loss = - y_true * tf.math.log(y_pred) - (1.0-y_true) * tf.math.log(1.0-y_pred)
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask)+1e-7)

def matthews_correlation_fixed(y_true, y_pred, threshold=0.3):
    y_pred = tf.cast(y_pred>threshold, y_pred.dtype)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.-y_pred))
    fp = tf.reduce_sum((1.-y_true) * y_pred)
    tn = tf.reduce_sum((1.-y_true) * (1.-y_pred))
    score = (tp*tn - fp*fn) / tf.math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-7)
    return score

def matthews_correlation_03(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    # minus value is invalid label. (just to control the number of labels constant)
    mask = y_true>=-1e-7
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    
    return matthews_correlation_fixed(y_true, y_pred, threshold=0.3)

def matthews_correlation_best(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    # minus value is invalid label. (just to control the number of labels constant)
    mask = y_true>=-1e-7
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    
    best_score = 0.
    for threshold in [0.2,0.5,0.8]:#tf.linspace(0.2,0.8,5):
        score = matthews_correlation_fixed(y_true, y_pred, threshold)
        best_score = tf.maximum(best_score, score)
    return best_score
    
    
def average_box_size(inputs):
    sizes = tf.math.sqrt((inputs[:,:,2] - inputs[:,:,0]) * (inputs[:,:,3] - inputs[:,:,1]))
    sizes_mask = tf.cast(sizes > 1e-7, tf.float32)
    average_size = tf.reduce_sum(sizes, axis=1, keepdims=True) / (tf.reduce_sum(sizes_mask, axis=1, keepdims=True)+1e-7)
    return average_size

def map_test(map_model, input_rgb, input_boxes):
    averagesize = average_box_size(input_boxes)
    out_p, out_map = map_model([input_rgb, input_boxes, averagesize])
    return out_p, out_map

def map_inference_func_wrapper(map_model, include_resize=False, input_shape = (512, 896, 3)):
    
    @tf.function
    def map_test(input_rgb, input_boxes):
        if include_resize:
            input_rgb = tf.image.resize(input_rgb, (input_shape[0], input_shape[1]), method="bilinear")
        averagesize = average_box_size(input_boxes)
        out_p, out_map = map_model([input_rgb, input_boxes, averagesize])
        return out_p, out_map
    
    return map_test
    
    
def build_model(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             from_scratch=False,
             return_feature_ext=False,
             size="SM",
             map_model=None):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    input_rgb = Input(input_shape, name="input_rgb")#256,256,3
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    enc_in = input_rgb
    if map_model is not None:
        averagesize = Lambda(average_box_size)(input_boxes)
        
        out_map = map_model([input_rgb, input_boxes, averagesize])[1]
        out_map = UpSampling2D(4)(out_map)
        enc_in = Lambda(lambda x: tf.concat(x, axis=-1))([input_rgb, out_map])
    
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])

    use_coord_conv = False

    if use_coord_conv:
        print("use coords")
        
        x = Lambda(add_coords, name="add_coords")(x)
        x = Lambda(add_high_freq_coords, name="add_high_freq_coords")(x)
    
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)
    decoder_out = outs[-1]
    x = outs[-1]
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    ground_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="ground_mask",)(x)
    
    if size=="SS":
        roi_size = 72
        num_cbr = 3
    elif size=="SM":
        roi_size = 72
        num_cbr = 6
    elif size=="MM":
        roi_size = 108
        num_cbr = 6
        
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [roi_size,roi_size], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)
    mode="direct_mask"
    if mode=="direct_mask":
        #x_0 = feature_w_mask
        #x_1 = AveragePooling2D(2)(x_0)
        #x_2 = AveragePooling2D(2)(x_1)
        for layer_idx in range(num_cbr):#7x3
            feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_0 = cbr(x_0, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_1 = cbr(x_1, 32, kernel=7, stride=1, name=f"player_cbrs{layer_idx}")
            #x_2 = cbr(x_2, 32, kernel=7, stride=1, name=f"player_cbrss{layer_idx}")
        
        #x_1 = UpSampling2D(2)(x_1)
        #x_2 = UpSampling2D(4)(x_2)
        #feature_w_mask = Lambda(lambda x: tf.concat(x, axis=-1))([x_0,x_1,x_2])
        
        
        #feature_w_mask = cbr(feature_w_mask, 48, kernel=7, stride=1, name="player_cbr0")
        #for layer_idx in range(3):
        #    feature_w_mask = resblock(feature_w_mask, 48, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        #"""
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 1, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        """#12/20一時的に変更。裏表オクルージョンモデル。
        player_mask = Conv2D(2, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 2, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        # is_contactを弱めに学習してもいいのかもしれない？？
        # もしくは共通のグランドマスクを使う。
        #"""
        
        
    elif mode=="ch_attention":
        x = feature_w_mask
        #for layer_idx in range(3):
        #    x = cbr(x, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
        x = cbr(x, 48, kernel=7, stride=1, name="player_cbr0")
        for layer_idx in range(3):#5,7
            x = resblock(x, 32, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        
        x = GlobalAveragePooling2D()(x)
        attention_weight = Dense(num_feature_ch, activation="sigmoid", name="ch_attention")(x)
        attention_feature = Lambda(lambda x: x[0][...,:num_feature_ch] * tf.reshape(x[1], [-1,1,1,num_feature_ch]), name="mul_attention")([feature_w_mask, attention_weight])
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(attention_feature)
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                                              arguments={"num_ch": 1, 
                                                         "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        #Reshape((1,1,num_feature_ch), name="ch_attention_reshape")(attention_weight)
        #x_out = Multiply()([features, attention_weight])
        
        
    #"""
    # concat masks, [batch, num_player+1(ground), h, w, 1]
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    
    pairs_contact = Lambda(contact_btw_selected_pairs, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    """#12/20一時的に変更。uraomoteモデル
    #all_masks = Lambda(lambda x: tf.concat([x[...,0:1], x[...,1:2]], axis=1))(player_mask) # concat ground and player_mask at 2nd axis
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    pairs_contact = Lambda(contact_btw_selected_pairs_uraomote, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    #pairs_contact = Lambda(contact_btw_selected_pairs_v2, name="contact_btw_selected_pairs",
    #               #arguments={"num_ch": 1, 
    #               #           "wide_mode": True},
    #               )([contact_map, all_masks, input_pairs]) 
    
    #"""
    
    
    
    # ペア予測がないと、ABC三選手が画像上重なる場合に、A-B, B-Cのみが干渉するケースに対応できないかも…。
    # マルチなロスにする方が自然かな…。この特徴が強すぎると詰むかも？
    add_pairwise_mask = False
    if add_pairwise_mask:
        num_pair_feature = 4
        feature_g = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_g")(features)
        feature_p = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_p")(feature_w_mask)
        feature_p = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize_features",
                                              arguments={"num_ch": num_pair_feature, 
                                                         "wide_mode": True})([feature_p, input_boxes, ground_mask]) 
        features_gp = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([feature_g, feature_p])
        pairs_feature = Lambda(contact_btw_selected_pairs_nomask, name="feature_btw_selected_pairs",
                       )([features_gp, input_pairs]) 
        pairwise_prediction = Conv2D(1, activation="sigmoid", kernel_size=1, strides=1, padding="same", 
                              name="pairwise_prediction")(pairs_feature)
        pairs_contact = Lambda(lambda x: x[0] * x[1], name="multiply_final_preds")([pairs_contact, pairwise_prediction])
    
    
    
    
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    
    inputs = [input_rgb, input_boxes, input_pairs]
    outputs = [pairs_contact_reduced, contact_map]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "contact_map": l2_regularization,
              #"zoom_dev_abs": weighted_dummy_loss
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "contact_map": 0.01,
                    #"zoom_dev_abs": 0.1*4
                    }
    metrics = {"output_contact_label": [matthews_correlation_best]}
    
    
    
    
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, pairs_contact_reduced])
    if not return_feature_ext:
        return model, sub_model, losses, loss_weights, metrics
    else:
        model_feature_ext = Model([input_rgb], [decoder_out])
        return model, sub_model, model_feature_ext

def fill_zero_with_average(inputs):
    b, n = tf.unstack(tf.shape(inputs))[:2]
    is_zero = tf.cast(inputs==0, tf.float32)
    average = tf.reduce_sum(inputs * is_zero) / (tf.reduce_sum(is_zero) + 1e-7)
    outputs = inputs * (1. - is_zero) + is_zero * average
    return tf.reshape(outputs, [b,n,1])

def player_coords_feature_branch(input_boxes, input_player_positions, out_ch=16):
    
    def concat_features(inputs):
        input_boxes, input_player_positions = inputs
        batch, num_player = tf.unstack(tf.shape(input_boxes))[:2]
        box_pos_x = (input_boxes[:,:,1:2] + input_boxes[:,:,3:4])/2
        box_pos_y = (input_boxes[:,:,0:1] + input_boxes[:,:,2:3])/2
        box_w = -input_boxes[:,:,1:2] + input_boxes[:,:,3:4]
        box_h = -input_boxes[:,:,0:1] + input_boxes[:,:,2:3]
        box_pos_x_std = tf.math.reduce_std(box_pos_x, axis=1, keepdims=True) + 1e-7
        box_pos_y_std = tf.math.reduce_std(box_pos_y, axis=1, keepdims=True) + 1e-7
        
        positions_adjust = input_player_positions - tf.reduce_mean(input_player_positions, axis=1, keepdims=True)
        map_pos_x = positions_adjust[:,:,:1] 
        map_pos_y = positions_adjust[:,:,1:2]
        map_pos_x_std = tf.math.reduce_std(map_pos_x, axis=1, keepdims=True) + 1e-7
        map_pos_y_std = tf.math.reduce_std(map_pos_y, axis=1, keepdims=True) + 1e-7
        
        box_pos_x_adjust = (box_pos_x - tf.reduce_mean(box_pos_x, axis=1, keepdims=True)) * map_pos_x_std / box_pos_x_std 
        box_pos_y_adjust = (box_pos_y - tf.reduce_mean(box_pos_y, axis=1, keepdims=True)) * map_pos_y_std / box_pos_y_std 
        
        map_box_pos_dx = map_pos_x - box_pos_x_adjust
        map_box_pos_dy = map_pos_y - box_pos_y_adjust
        
        map_pos_x_std_tile = tf.tile(map_pos_x_std, [1, num_player, 1])
        map_pos_y_std_tile = tf.tile(map_pos_y_std, [1, num_player, 1])
        box_pos_x_std_tile = tf.tile(box_pos_x_std, [1, num_player, 1])
        box_pos_y_std_tile = tf.tile(box_pos_y_std, [1, num_player, 1])
        
        features = [box_pos_x, box_pos_y, box_w, box_h,
                    box_pos_x_std_tile, box_pos_y_std_tile,
                    map_pos_x, map_pos_y, 
                    map_pos_x_std_tile, map_pos_y_std_tile,
                    box_pos_x_adjust, box_pos_y_adjust,
                    map_box_pos_dx, map_box_pos_dy,
                    ]
        features = tf.concat(features, axis=-1)
        return features
    
    # DENSEしてそのあとまたアグリゲーション取るほうがいい気もする TODO
    
    
    
    
    
    x = Lambda(concat_features)([input_boxes, input_player_positions])
    x = Dense(128, activation="relu", name="coods_branch_dense_0")(x)
    x = Dense(out_ch, activation="relu", name="coods_branch_dense_1")(x)
    return x
    
    

def build_model_explicit_distance(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             from_scratch=False,
             return_feature_ext=False,
             size="SS",
             map_model=None):
    """
    TODO 寸法大丈夫かな？？フルスケールでまわるか？
    対地面の場合、扱いがムズイかも。。
    ボクセル的な奥行きを出力して三次元奥行きで交わり具合を出すのもいい気がする。
    
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    input_rgb = Input(input_shape, name="input_rgb")#256,256,3
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    input_player_positions = Input(shape=[num_boxes,2], name="input_player_positions")
    enc_in = input_rgb
    """
    if map_model is not None:
        averagesize = Lambda(average_box_size)(input_boxes)
        
        player_pos, out_map = map_model([input_rgb, input_boxes, averagesize])
        player_depth = Lambda(lambda x: x[...,1:])(player_pos) # [batch, num_p, 1]
        depth = Lambda(lambda x: x[...,1:])(out_map)
        depth = UpSampling2D(4)(depth)
        enc_in = Lambda(lambda x: tf.concat(x, axis=-1))([input_rgb, depth])
    """
    
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])

    use_coord_conv = False

    if use_coord_conv:
        print("use coords")
        
        x = Lambda(add_coords, name="add_coords")(x)
        x = Lambda(add_high_freq_coords, name="add_high_freq_coords")(x)
    
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)
    decoder_out = outs[-1]
    x = outs[-1]
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    ground_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="ground_mask",)(x)
    
    if size=="SS":
        roi_size = 72
        num_cbr = 3
    elif size=="SM":
        roi_size = 72
        num_cbr = 6
    elif size=="MM":
        roi_size = 108
        num_cbr = 6
        
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [roi_size,roi_size], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)
    mode="direct_mask"
    if mode=="direct_mask":
        #x_0 = feature_w_mask
        #x_1 = AveragePooling2D(2)(x_0)
        #x_2 = AveragePooling2D(2)(x_1)
        for layer_idx in range(num_cbr):#7x3
            feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_0 = cbr(x_0, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_1 = cbr(x_1, 32, kernel=7, stride=1, name=f"player_cbrs{layer_idx}")
            #x_2 = cbr(x_2, 32, kernel=7, stride=1, name=f"player_cbrss{layer_idx}")
        
        #x_1 = UpSampling2D(2)(x_1)
        #x_2 = UpSampling2D(4)(x_2)
        #feature_w_mask = Lambda(lambda x: tf.concat(x, axis=-1))([x_0,x_1,x_2])
        
        
        #feature_w_mask = cbr(feature_w_mask, 48, kernel=7, stride=1, name="player_cbr0")
        #for layer_idx in range(3):
        #    feature_w_mask = resblock(feature_w_mask, 48, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        #"""
        # 近くても干渉していないときロスがでちゃう。どうする？
        """
        offset_scale = 0.025
        thickness_scale = 0.025#/100
        player_depth_offset = Conv2D(1, activation="tanh", kernel_size=7, strides=1, padding="same", 
                           name="player_depth_offset")(feature_w_mask)
        player_thickness = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_thickness")(feature_w_mask)
        player_depth_min = Lambda(lambda x: x[0]*offset_scale - x[1]*thickness_scale + tf.reshape(x[2], [-1])[:,tf.newaxis,tf.newaxis,tf.newaxis])([player_depth_offset, player_thickness, player_depth])
        player_depth_max = Lambda(lambda x: x[0]*offset_scale + x[1]*thickness_scale + tf.reshape(x[2], [-1])[:,tf.newaxis,tf.newaxis,tf.newaxis])([player_depth_offset, player_thickness, player_depth])
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        
        player_mask_w_d = Lambda(lambda x: tf.concat(x, axis=-1))([player_mask, player_depth_min, player_depth_max])
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask_w_d = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 1+2, 
                              "wide_mode": True})([player_mask_w_d, input_boxes, ground_mask]) 
        """
        
        # ボクセル試す前にシンプルに距離さばき。
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 1, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        """
        #### 0108 feature ext
        ### 0108 一時的
        ext_ch = 8
        feature_w_mask_stop = Lambda(lambda x: tf.stop_gradient(x))(feature_w_mask)
        
        feature_for_ext = cbr(feature_w_mask_stop, ext_ch, kernel=7, stride=1, name="player_cbr_ext")
        
        player_mask_w_f = Lambda(lambda x: tf.concat(x, axis=-1))([player_mask, feature_for_ext])

        player_mask_w_f = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 1+ext_ch, 
                              "wide_mode": True})([player_mask_w_f, input_boxes, ground_mask]) 
        player_mask = Lambda(lambda x: x[...,:1])(player_mask_w_f)
        player_features = Lambda(lambda x: x[...,1:])(player_mask_w_f)
        """
        
        
        """#12/20一時的に変更。裏表オクルージョンモデル。
        player_mask = Conv2D(2, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 2, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        # is_contactを弱めに学習してもいいのかもしれない？？
        # もしくは共通のグランドマスクを使う。
        #"""
        
        
    elif mode=="ch_attention":
        x = feature_w_mask
        #for layer_idx in range(3):
        #    x = cbr(x, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
        x = cbr(x, 48, kernel=7, stride=1, name="player_cbr0")
        for layer_idx in range(3):#5,7
            x = resblock(x, 32, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        
        x = GlobalAveragePooling2D()(x)
        attention_weight = Dense(num_feature_ch, activation="sigmoid", name="ch_attention")(x)
        attention_feature = Lambda(lambda x: x[0][...,:num_feature_ch] * tf.reshape(x[1], [-1,1,1,num_feature_ch]), name="mul_attention")([feature_w_mask, attention_weight])
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(attention_feature)
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                                              arguments={"num_ch": 1, 
                                                         "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        #Reshape((1,1,num_feature_ch), name="ch_attention_reshape")(attention_weight)
        #x_out = Multiply()([features, attention_weight])
        
        
    #"""
    # concat masks, [batch, num_player+1(ground), h, w, 1]
    #player_mask_w_d
    
    #player_mask = Lambda(lambda x: x[..., :1])(player_mask_w_d)
    #depth_minmax = Lambda(lambda x: x[..., 1:])(player_mask_w_d)
    #depth_minmax_w_dummy = Lambda(lambda x: tf.concat([tf.ones_like(x[:,0:1]), x],axis=1))(depth_minmax)
    pos_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(input_player_positions)
    
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    
    pairs_contact = Lambda(contact_btw_selected_pairs, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    ### 0108 一時的
    # player_features_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(player_features)
    # pairs_features = Lambda(extract_peak_features, arguments={"num_ch": ext_ch})([pairs_contact, player_features_dummy, input_pairs])
    

    """
    pairs_contact_penetration = Lambda(contact_btw_selected_pairs_depth, name="contact_btw_selected_pairs_d",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([depth_minmax_w_dummy, input_pairs]) 
    pairs_contact_penetration = PenetrationLayer(max_thickness=thickness_scale*2)(pairs_contact_penetration)
    """
    
    
    """#12/20一時的に変更。uraomoteモデル
    #all_masks = Lambda(lambda x: tf.concat([x[...,0:1], x[...,1:2]], axis=1))(player_mask) # concat ground and player_mask at 2nd axis
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    pairs_contact = Lambda(contact_btw_selected_pairs_uraomote, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    #pairs_contact = Lambda(contact_btw_selected_pairs_v2, name="contact_btw_selected_pairs",
    #               #arguments={"num_ch": 1, 
    #               #           "wide_mode": True},
    #               )([contact_map, all_masks, input_pairs]) 
    
    #"""
    
    
    
    # ペア予測がないと、ABC三選手が画像上重なる場合に、A-B, B-Cのみが干渉するケースに対応できないかも…。
    # マルチなロスにする方が自然かな…。この特徴が強すぎると詰むかも？
    add_pairwise_mask = False
    if add_pairwise_mask:
        num_pair_feature = 4
        feature_g = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_g")(features)
        feature_p = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_p")(feature_w_mask)
        feature_p = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize_features",
                                              arguments={"num_ch": num_pair_feature, 
                                                         "wide_mode": True})([feature_p, input_boxes, ground_mask]) 
        features_gp = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([feature_g, feature_p])
        pairs_feature = Lambda(contact_btw_selected_pairs_nomask, name="feature_btw_selected_pairs",
                       )([features_gp, input_pairs]) 
        pairwise_prediction = Conv2D(1, activation="sigmoid", kernel_size=1, strides=1, padding="same", 
                              name="pairwise_prediction")(pairs_feature)
        pairs_contact = Lambda(lambda x: x[0] * x[1], name="multiply_final_preds")([pairs_contact, pairwise_prediction])
    
    
    #pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    #pairs_contact_reduced_penetration = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label_penetration")(pairs_contact_penetration)
    
    # TODO ピークの場所に対応する特徴量を引っ張り出してconcatなど。
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    image_contact_reduced_stopgrad = Lambda(lambda x: tf.stop_gradient(x[...,tf.newaxis]), name="stop_grad")(pairs_contact_reduced)
    # image_contact_reduced_stopgrad_filled = Lambda(fill_zero_with_average, name="player_ref_fill")(pairs_contact_reduced)
    pairs_xyd = Lambda(extract_xyd_features)([pos_w_dummy, input_pairs])
    ground_mask = Lambda(lambda x: tf.cast(x[:,:,1:2]==0, tf.float32))(input_pairs)
    
    #pairs_xydconf = Lambda(lambda x: tf.concat([tf.math.log(tf.clip_by_value(x[0]/(1-x[0]),1e-7,1e7)), x[1]], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    pairs_xydconf = Lambda(lambda x: tf.concat([x[0]-0.5, x[1]*10], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    
    ### 0108 一時的
    ## pairs_features = Lambda(lambda x: tf.concat(x, axis=-1))([pairs_features, pairs_xydconf])
    pairs_features = pairs_xydconf
    
    # too shallow?
    #pairs_xydconf = Dense(32, activation="relu")(pairs_xydconf)
    # pairs_xydconf = Dropout(0.2)(pairs_xydconf)
    #image_contact_reduced_stopgrad_bn = BatchNormalization()(image_contact_reduced_stopgrad)
    
    #total_pred = Dense(1, activation="sigmoid", 
    #                   #kernel_initializer='ones',
    #                   #bias_initializer='zeros',
    #                   name="poutput_contact_label_penetration")(image_contact_reduced_stopgrad)# 一時的な名称
    
    # predict by image and positions
    pairs_features = Dense(128, activation="relu", name="dense0")(pairs_features)
    pairs_features = Dense(128, activation="relu", name="dense1")(pairs_features)
    pairs_features = Dropout(0.2)(pairs_features)
    player_pred = Dense(1, activation="sigmoid", name="output_contact_label_player")(pairs_features)
    #total_pred = Lambda(lambda x: x[0], name="output_contact_label_penetration")([total_pred, player_mask])# 一時的な名称
    
    
    # features for ground contact
    """
    pf_ch = 16
    player_features = player_coords_feature_branch(input_boxes, input_player_positions, out_ch=pf_ch)
    player_features_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(player_features)
    pairs_g_contact_features = Lambda(extract_pair_features_single,
                                       arguments={"num_ch":pf_ch, "only_1":True})([player_features_w_dummy, input_pairs])# for ground
    pairs_xydconf_g = Lambda(lambda x: tf.concat([x[0]-0.5, x[1]], axis=-1))([image_contact_reduced_stopgrad, pairs_g_contact_features])
    pairs_features_g = Dense(128, activation="relu", name="dense0_g")(pairs_xydconf_g)
    pairs_features_g = Dense(128, activation="relu", name="dense1_g")(pairs_features_g)
    pairs_features_g = Dropout(0.2)(pairs_features_g)
    ground_pred = Dense(1, activation="sigmoid", name="output_contact_label_ground")(pairs_features_g)
    output_contact_label_total = Lambda(lambda x: (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([ground_pred, player_pred, ground_mask])
    """
    
    
    
    player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    #output_contact_label_total = Lambda(lambda x: (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    
    
    
    #player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    output_contact_label_total = Lambda(lambda x: #x[0]*x[2] + x[1]*(1.-x[2])[...,0]
                                        (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    #total_pred = Lambda(lambda x:x[...,0], name="output_contact_label_penetration")(total_pred)# 一時的な名称
    
    
    inputs = [input_rgb, input_boxes, input_pairs, input_player_positions]
    outputs = [pairs_contact_reduced, 
               player_pred, 
               output_contact_label_total, 
               contact_map]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_player": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_total": bce_loss,
              "contact_map": l2_regularization,
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "output_contact_label_player": 1.,#{"z_error": 1e-4,
                    "output_contact_label_total":1,
                    "contact_map": 0.01,
                    }
    metrics = {"output_contact_label": [matthews_correlation_best],
                "output_contact_label_player": [matthews_correlation_best],
                "output_contact_label_total": [matthews_correlation_best],
                }
    
    
    
    
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, 
                               pairs_contact_reduced,
                               output_contact_label_total,
                               ])
    if not return_feature_ext:
        return model, sub_model, losses, loss_weights, metrics
    else:
        model_feature_ext = Model([input_rgb], [decoder_out])
        return model, sub_model, model_feature_ext



def build_model_explicit_distance_feature_dense(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             size="SS",
             feature_ext_weight=""):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """

    model, sub_model, model_feature_ext = build_model_explicit_distance(input_shape=input_shape,
                                         backbone=backbone, 
                                         minimum_stride=minimum_stride, 
                                         max_stride = max_stride,
                                         is_train=False,
                                         num_boxes = num_boxes,
                                         from_scratch=True,
                                         size=size,
                                         return_feature_ext=True)
    model.load_weights(feature_ext_weight)
    model.trainable = False
    model_feature_ext.trainable = False
    
    input_rgb = Input(input_shape, name="input_rgb")
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    input_player_positions = Input(shape=[num_boxes,2], name="input_player_positions")
    
    #input_warp_p = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_p")
    #input_warp_n = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_n")
    
    features = model_feature_ext(input_rgb, training=False)
    
    x = features
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    ground_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="ground_mask",)(x)
    
    if size=="SS":
        roi_size = 72
        num_cbr = 3
    elif size=="SM":
        roi_size = 72
        num_cbr = 6
    elif size=="MM":
        roi_size = 108
        num_cbr = 6
        
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [roi_size,roi_size], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)

    for layer_idx in range(num_cbr):#7x3
        feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
    
    player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                       name="player_mask")(feature_w_mask)
    
    ### 0108 一時的オフ
    #player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
    #           arguments={"num_ch": 1, 
    #                      "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
    
    
    #### 0108 feature ext
    ### 0108 一時的
    ext_ch = 8
    feature_w_mask_stop = Lambda(lambda x: tf.stop_gradient(x))(feature_w_mask)
    
    feature_for_ext = cbr(feature_w_mask_stop, ext_ch, kernel=7, stride=1, name="player_cbr_ext")
    
    player_mask_w_f = Lambda(lambda x: tf.concat(x, axis=-1))([player_mask, feature_for_ext])

    player_mask_w_f = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
               arguments={"num_ch": 1+ext_ch, 
                          "wide_mode": True})([player_mask_w_f, input_boxes, ground_mask]) 
    player_mask = Lambda(lambda x: x[...,:1])(player_mask_w_f)
    player_features = Lambda(lambda x: x[...,1:])(player_mask_w_f)
    

    pos_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(input_player_positions)
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    pairs_contact = Lambda(contact_btw_selected_pairs, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    ### 0108 一時的
    player_features_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(player_features)
    pairs_features = Lambda(extract_peak_features, arguments={"num_ch": ext_ch})([pairs_contact, player_features_dummy, input_pairs])
    print(pairs_features)
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    image_contact_reduced_stopgrad = Lambda(lambda x: tf.stop_gradient(x[...,tf.newaxis]), name="stop_grad")(pairs_contact_reduced)
    # image_contact_reduced_stopgrad_filled = Lambda(fill_zero_with_average, name="player_ref_fill")(pairs_contact_reduced)
    pairs_xyd = Lambda(extract_xyd_features)([pos_w_dummy, input_pairs])
    ground_mask = Lambda(lambda x: tf.cast(x[:,:,1:2]==0, tf.float32))(input_pairs)
    
    #pairs_xydconf = Lambda(lambda x: tf.concat([tf.math.log(tf.clip_by_value(x[0]/(1-x[0]),1e-7,1e7)), x[1]], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    pairs_xydconf = Lambda(lambda x: tf.concat([x[0]-0.5, x[1]*10], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    
    ### 0108 一時的
    pairs_features_compare = Lambda(lambda x: tf.concat([tf.reshape(x[0][:,:,0,:], [-1, tf.shape(x[0])[1], ext_ch*2]), x[1]], axis=-1))([pairs_features, pairs_xydconf])
    pairs_features = pairs_xydconf
    
    # predict by image and positions
    pairs_features = Dense(128, activation="relu", name="dense0")(pairs_features)
    pairs_features = Dense(128, activation="relu", name="dense1")(pairs_features)
    pairs_features = Dropout(0.2)(pairs_features)
    player_pred = Dense(1, activation="sigmoid", name="output_contact_label_player")(pairs_features)
    
    # predict by image and features
    pairs_features_compare = Dense(128, activation="relu", name="dense0c")(pairs_features_compare)
    pairs_features_compare = Dense(128, activation="relu", name="dense1c")(pairs_features_compare)
    pairs_features_compare = Dropout(0.2)(pairs_features_compare)
    player_pred_compare = Dense(1, activation="sigmoid", name="output_contact_label_playerc")(pairs_features_compare)
    
    
    player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    #output_contact_label_total = Lambda(lambda x: (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    
    
    
    #player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    output_contact_label_total = Lambda(lambda x: #x[0]*x[2] + x[1]*(1.-x[2])[...,0]
                                        (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    #total_pred = Lambda(lambda x:x[...,0], name="output_contact_label_penetration")(total_pred)# 一時的な名称
    
    
    inputs = [input_rgb, input_boxes, input_pairs, input_player_positions]
    outputs = [pairs_contact_reduced, 
               player_pred, 
               player_pred_compare,
               output_contact_label_total, 
               contact_map]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_player": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_playerc": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_total": bce_loss,
              "contact_map": l2_regularization,
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "output_contact_label_player": 1.,#{"z_error": 1e-4,
                    "output_contact_label_playerc": 1.,#{"z_error": 1e-4,
                    "output_contact_label_total":1,
                    "contact_map": 0.01,
                    }
    metrics = {"output_contact_label": [matthews_correlation_best],
                "output_contact_label_player": [matthews_correlation_best],
                "output_contact_label_playerc": [matthews_correlation_best],
                "output_contact_label_total": [matthews_correlation_best, matthews_correlation_03],
                }
    
    
    
    
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, 
                               pairs_contact_reduced,
                               output_contact_label_total,
                               ])
    return model, sub_model, losses, loss_weights, metrics
    


def build_model_explicit_distance_multi_view(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             size="SS",
             feature_ext_weight=""):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """

    model, sub_model, model_feature_ext = build_model_explicit_distance(input_shape=input_shape,
                                         backbone=backbone, 
                                         minimum_stride=minimum_stride, 
                                         max_stride = max_stride,
                                         is_train=False,
                                         num_boxes = num_boxes,
                                         from_scratch=True,
                                         size=size,
                                         return_feature_ext=True)
    model.load_weights(feature_ext_weight)
    model.trainable = False
    model_feature_ext.trainable = False
    
    input_rgb = Input(input_shape, name="input_rgb")
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    input_player_positions = Input(shape=[num_boxes,2], name="input_player_positions")
    input_player_in_other_video = Input(shape=[num_boxes], name="input_player_in_other_video", dtype=tf.int32)
    
    #input_warp_p = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_p")
    #input_warp_n = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_n")
    
    features = model_feature_ext(input_rgb, training=False)
    
    x = features
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    ground_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="ground_mask",)(x)
    
    if size=="SS":
        roi_size = 72
        num_cbr = 3
    elif size=="SM":
        roi_size = 72
        num_cbr = 6
    elif size=="MM":
        roi_size = 108
        num_cbr = 6
        
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [roi_size,roi_size], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)

    for layer_idx in range(num_cbr):#7x3
        feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
    
    player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                       name="player_mask")(feature_w_mask)
    
    ### 0108 一時的オフ
    #player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
    #           arguments={"num_ch": 1, 
    #                      "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
    
    
    #### 0108 feature ext
    ### 0108 一時的
    ext_ch = 8
    feature_w_mask_stop = Lambda(lambda x: tf.stop_gradient(x))(feature_w_mask)
    
    feature_for_ext = cbr(feature_w_mask_stop, ext_ch, kernel=7, stride=1, name="player_cbr_ext")
    player_gap_features = GlobalAveragePooling2D()(feature_for_ext) # batch, num_player, features
    player_gap_features = Lambda(lambda x: tf.reshape(x[0], [tf.shape(x[1])[0], tf.shape(x[1])[1], ext_ch]))([player_gap_features, input_boxes])
    
    player_mask_w_f = Lambda(lambda x: tf.concat(x, axis=-1))([player_mask, feature_for_ext])

    player_mask_w_f = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
               arguments={"num_ch": 1+ext_ch, 
                          "wide_mode": True})([player_mask_w_f, input_boxes, ground_mask]) 
    player_mask = Lambda(lambda x: x[...,:1])(player_mask_w_f)
    player_features = Lambda(lambda x: x[...,1:])(player_mask_w_f)
    

    pos_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(input_player_positions)
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    pairs_contact = Lambda(contact_btw_selected_pairs, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    ### 0108 一時的
    player_features_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(player_features)
    pairs_features = Lambda(extract_peak_features, arguments={"num_ch": ext_ch})([pairs_contact, player_features_dummy, input_pairs])
    
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    image_contact_reduced_stopgrad = Lambda(lambda x: tf.stop_gradient(x[...,tf.newaxis]), name="stop_grad")(pairs_contact_reduced)
    # image_contact_reduced_stopgrad_filled = Lambda(fill_zero_with_average, name="player_ref_fill")(pairs_contact_reduced)
    pairs_xyd = Lambda(extract_xyd_features)([pos_w_dummy, input_pairs])
    ground_mask = Lambda(lambda x: tf.cast(x[:,:,1:2]==0, tf.float32))(input_pairs)
    
    #pairs_xydconf = Lambda(lambda x: tf.concat([tf.math.log(tf.clip_by_value(x[0]/(1-x[0]),1e-7,1e7)), x[1]], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    pairs_xydconf = Lambda(lambda x: tf.concat([x[0]-0.5, x[1]*10], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    
    
    
    ### 作業中。！！！！
    # split batch into two side and end and use features of other side
    player_gap_features_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(player_gap_features)
    
    player_gap_features_w_dummy_s, player_gap_features_w_dummy_e = Lambda(lambda x: tf.split(x, 2, axis=0))(player_gap_features_w_dummy)
    other_video_player_s, other_video_player_e = Lambda(lambda x: tf.split(x, 2, axis=0))(input_player_in_other_video)
    
    # pairs_features_gap_e_for_s = Lambda(extract_xyd_features, arguments={"num_ch": ext_ch})([player_gap_features_w_dummy_e, other_video_player_s])
    # pairs_features_gap_s_for_e = Lambda(extract_xyd_features, arguments={"num_ch": ext_ch})([player_gap_features_w_dummy_s, other_video_player_e])
    player_gap_features_e_for_s = Lambda(lambda x: tf.gather(x[0], x[1], batch_dims=1))([player_gap_features_w_dummy_e, other_video_player_s])
    player_gap_features_s_for_e = Lambda(lambda x: tf.gather(x[0], x[1], batch_dims=1))([player_gap_features_w_dummy_s, other_video_player_e])

    player_gap_features_other_side = Lambda(lambda x: tf.concat(x, axis=0))([player_gap_features_e_for_s, player_gap_features_s_for_e])
    player_gap_features_other_side_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(player_gap_features_other_side)
    
    pairs_features_gap = Lambda(extract_pair_features, arguments={"num_ch": ext_ch})([player_gap_features_w_dummy, input_pairs])
    pairs_features_gap_other_side = Lambda(extract_pair_features, arguments={"num_ch": ext_ch})([player_gap_features_other_side_w_dummy, input_pairs])
    pairs_features_gap_multi = Lambda(lambda x: x[0] * x[1])([pairs_features_gap, pairs_features_gap_other_side])
    
    ### 0108 一時的
    pairs_features_compare = Lambda(lambda x: tf.concat([tf.reshape(x[0][:,:,0,:], [-1, tf.shape(x[0])[1], ext_ch*2]), x[1], x[2]], axis=-1))([pairs_features, pairs_xydconf, pairs_features_gap_multi])
    pairs_features = pairs_xydconf
    
    # predict by image and positions
    pairs_features = Dense(128, activation="relu", name="dense0")(pairs_features)
    pairs_features = Dense(128, activation="relu", name="dense1")(pairs_features)
    pairs_features = Dropout(0.2)(pairs_features)
    player_pred = Dense(1, activation="sigmoid", name="output_contact_label_player")(pairs_features)
    
    # predict by image and features
    pairs_features_compare = Dense(128, activation="relu", name="dense0c")(pairs_features_compare)
    pairs_features_compare = Dense(128, activation="relu", name="dense1c")(pairs_features_compare)
    pairs_features_compare = Dropout(0.2)(pairs_features_compare)
    player_pred_compare = Dense(1, activation="sigmoid", name="output_contact_label_playerc")(pairs_features_compare)
    
    
    player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    #output_contact_label_total = Lambda(lambda x: (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    
    
    
    #player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    output_contact_label_total = Lambda(lambda x: #x[0]*x[2] + x[1]*(1.-x[2])[...,0]
                                        (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    #total_pred = Lambda(lambda x:x[...,0], name="output_contact_label_penetration")(total_pred)# 一時的な名称
    
    
    inputs = [input_rgb, input_boxes, input_pairs, input_player_positions, input_player_in_other_video]
    outputs = [pairs_contact_reduced, 
               player_pred, 
               player_pred_compare,
               output_contact_label_total, 
               contact_map]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_player": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_playerc": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_total": bce_loss,
              "contact_map": l2_regularization,
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "output_contact_label_player": 1.,#{"z_error": 1e-4,
                    "output_contact_label_playerc": 1.,#{"z_error": 1e-4,
                    "output_contact_label_total":1,
                    "contact_map": 0.01,
                    }
    metrics = {"output_contact_label": [matthews_correlation_best],
                "output_contact_label_player": [matthews_correlation_best],
                "output_contact_label_playerc": [matthews_correlation_best],
                "output_contact_label_total": [matthews_correlation_best, matthews_correlation_03],
                }
    
    
    
    
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, 
                               pairs_contact_reduced,
                               output_contact_label_total,
                               ])
    return model, sub_model, losses, loss_weights, metrics

def contact_btw_selected_pairs_shift_return_next_inputs(inputs):
    def tf_shift_image_nearest(images, shifts):
        """
        images: batch, height, width, ch
        shifts: float32 [batch, 2 (y, x)]
        not x, y order 
        """
        def yx_coords(batch, height, width):    
            w_grid, h_grid = tf.meshgrid(tf.range(width), tf.range(height))
            hw_grid = tf.tile(tf.stack([h_grid, w_grid], axis=-1)[tf.newaxis,...], [batch,1,1,1])
            return hw_grid
        
        def tf_shift_image(images, hw_grid_shift):
            """
            images: batch, height, width, ch
            shifts: float32 [batch, 2 (y, x)]
            not x, y order 
            """
            batch, height, width = tf.unstack(tf.shape(images))[:3] 
            height = tf.cast(height, tf.float32)
            width = tf.cast(width, tf.float32)
            
            hw_grid_shift_mask_h = tf.logical_and(hw_grid_shift[...,0]>=0, hw_grid_shift[...,0]<height)
            hw_grid_shift_mask_w = tf.logical_and(hw_grid_shift[...,1]>=0, hw_grid_shift[...,1]<width)
            hw_grid_shift_mask = tf.logical_and(hw_grid_shift_mask_w, hw_grid_shift_mask_h)
            indices = tf.where(hw_grid_shift_mask)
            
            origin_hw_ind = tf.cast(hw_grid_shift, tf.int32)
            origin_b_ind = tf.broadcast_to(tf.range(batch)[:,tf.newaxis,tf.newaxis,tf.newaxis], tf.shape(origin_hw_ind[...,:1]))
            origin_bhw_ind = tf.concat([origin_b_ind, origin_hw_ind], axis=-1)
            original_indices = tf.boolean_mask(origin_bhw_ind, hw_grid_shift_mask)
            values = tf.gather_nd(images, original_indices)
            shifted = tf.scatter_nd(tf.cast(indices, tf.int32), values, shape=tf.shape(images))
            
            #values = tf.gather_nd(images, indices)
            #values = tf.boolean_mask(images, hw_grid_shift_mask)
            #shifted = tf.scatter_nd(tf.cast(indices, tf.int32), values, shape=tf.shape(images))
            return shifted

        batch, height, width = tf.unstack(tf.shape(images))[:3] 
        hw_grid = tf.cast(yx_coords(batch, height, width), tf.float32)
        hw_grid_shift = hw_grid - shifts[:,tf.newaxis,tf.newaxis,:]
        nearest_grid = tf.math.round(hw_grid_shift)
        shifted = tf_shift_image(images, nearest_grid)
        return shifted


    contact_map_ground, player_mask, player_contact_w_mask, pairs, box_tlbrs = inputs
    
    batch, num_player, height, width, ch = tf.unstack(tf.shape(player_contact_w_mask))
    batch, num_pair, _ = tf.unstack(tf.shape(pairs))
    
    g_player_mask = tf.gather(tf.concat([contact_map_ground, player_mask], axis=-1), pairs[:,:,0]-1, axis=1, batch_dims=1)
    contact_w_ground = g_player_mask[...,:1] * g_player_mask[...,1:2]
    inputs_for_ground = tf.concat([g_player_mask, contact_w_ground], axis=-1)
    
    player_cm_1 = tf.gather(player_contact_w_mask, tf.maximum(pairs[:,:,0]-1, 0), axis=1, batch_dims=1)
    player_cm_2 = tf.gather(player_contact_w_mask, tf.maximum(pairs[:,:,1]-1, 0), axis=1, batch_dims=1)
    
    player_tlbr_1 = tf.gather(box_tlbrs, tf.maximum(pairs[:,:,0]-1, 0), axis=1, batch_dims=1)
    player_tlbr_2 = tf.gather(box_tlbrs, tf.maximum(pairs[:,:,1]-1, 0), axis=1, batch_dims=1) # batch, num_pair, 4
    w_2 = player_tlbr_2[:,:,3] - player_tlbr_2[:,:,1]
    h_2 = player_tlbr_2[:,:,2] - player_tlbr_2[:,:,0]
    w_rate = tf.cast(width, tf.float32) / w_2
    h_rate = tf.cast(height, tf.float32) / h_2
    shift_x = w_rate * (player_tlbr_2[:,:,1] - player_tlbr_1[:,:,1])
    shift_y = h_rate * (player_tlbr_2[:,:,0] - player_tlbr_1[:,:,0])
    
    images = tf.reshape(player_cm_2[...,1:], [batch*num_pair, height, width, 1])  # use only mask
    shifts = tf.reshape(tf.stack([shift_y, shift_x], axis=-1), [batch*num_pair, 2])
    #player_m_2_shift = tf_shift_image_bilinear(images, shifts)
    
    player_m_2_shift = tf_shift_image_nearest(images, shifts)
    
    player_m_2_shift = tf.reshape(player_m_2_shift, [batch, num_pair, height, width, 1])
    contact_btw_players = player_cm_1[...,:1] * player_cm_1[...,1:2] * player_m_2_shift
    inputs_for_player = tf.concat([player_cm_1[...,:1], player_cm_1[...,1:2], player_m_2_shift], axis=-1)
    
    is_ground = tf.cast(pairs[:,:,1]==0, tf.float32)[:,:,tf.newaxis,tf.newaxis,tf.newaxis] # ground
    all_contacts = contact_btw_players * (1.-is_ground) + contact_w_ground * is_ground
    
    next_inputs = inputs_for_ground * is_ground + inputs_for_player * (1-is_ground)
    return all_contacts, next_inputs

def contact_btw_selected_pairs_shift(inputs):
    def tf_shift_image_nearest(images, shifts):
        """
        images: batch, height, width, ch
        shifts: float32 [batch, 2 (y, x)]
        not x, y order 
        """
        def yx_coords(batch, height, width):    
            w_grid, h_grid = tf.meshgrid(tf.range(width), tf.range(height))
            hw_grid = tf.tile(tf.stack([h_grid, w_grid], axis=-1)[tf.newaxis,...], [batch,1,1,1])
            return hw_grid
        
        def tf_shift_image(images, hw_grid_shift):
            """
            images: batch, height, width, ch
            shifts: float32 [batch, 2 (y, x)]
            not x, y order 
            """
            batch, height, width = tf.unstack(tf.shape(images))[:3] 
            height = tf.cast(height, tf.float32)
            width = tf.cast(width, tf.float32)
            
            hw_grid_shift_mask_h = tf.logical_and(hw_grid_shift[...,0]>=0, hw_grid_shift[...,0]<height)
            hw_grid_shift_mask_w = tf.logical_and(hw_grid_shift[...,1]>=0, hw_grid_shift[...,1]<width)
            hw_grid_shift_mask = tf.logical_and(hw_grid_shift_mask_w, hw_grid_shift_mask_h)
            indices = tf.where(hw_grid_shift_mask)
            
            origin_hw_ind = tf.cast(hw_grid_shift, tf.int32)
            origin_b_ind = tf.broadcast_to(tf.range(batch)[:,tf.newaxis,tf.newaxis,tf.newaxis], tf.shape(origin_hw_ind[...,:1]))
            origin_bhw_ind = tf.concat([origin_b_ind, origin_hw_ind], axis=-1)
            original_indices = tf.boolean_mask(origin_bhw_ind, hw_grid_shift_mask)
            values = tf.gather_nd(images, original_indices)
            shifted = tf.scatter_nd(tf.cast(indices, tf.int32), values, shape=tf.shape(images))
            
            #values = tf.gather_nd(images, indices)
            #values = tf.boolean_mask(images, hw_grid_shift_mask)
            #shifted = tf.scatter_nd(tf.cast(indices, tf.int32), values, shape=tf.shape(images))
            return shifted

        batch, height, width = tf.unstack(tf.shape(images))[:3] 
        hw_grid = tf.cast(yx_coords(batch, height, width), tf.float32)
        hw_grid_shift = hw_grid - shifts[:,tf.newaxis,tf.newaxis,:]
        nearest_grid = tf.math.round(hw_grid_shift)
        shifted = tf_shift_image(images, nearest_grid)
        return shifted


    player_contact_w_ground, player_contact_w_mask, pairs, box_tlbrs = inputs
    
    batch, num_player, height, width, ch = tf.unstack(tf.shape(player_contact_w_mask))
    batch, num_pair, _ = tf.unstack(tf.shape(pairs))
    
    contact_w_ground = tf.gather(player_contact_w_ground, pairs[:,:,0]-1, axis=1, batch_dims=1)
    
    player_cm_1 = tf.gather(player_contact_w_mask, tf.maximum(pairs[:,:,0]-1, 0), axis=1, batch_dims=1)
    player_cm_2 = tf.gather(player_contact_w_mask, tf.maximum(pairs[:,:,1]-1, 0), axis=1, batch_dims=1)
    
    player_tlbr_1 = tf.gather(box_tlbrs, tf.maximum(pairs[:,:,0]-1, 0), axis=1, batch_dims=1)
    player_tlbr_2 = tf.gather(box_tlbrs, tf.maximum(pairs[:,:,1]-1, 0), axis=1, batch_dims=1) # batch, num_pair, 4
    w_2 = player_tlbr_2[:,:,3] - player_tlbr_2[:,:,1]
    h_2 = player_tlbr_2[:,:,2] - player_tlbr_2[:,:,0]
    w_rate = tf.cast(width, tf.float32) / w_2
    h_rate = tf.cast(height, tf.float32) / h_2
    shift_x = w_rate * (player_tlbr_2[:,:,1] - player_tlbr_1[:,:,1])
    shift_y = h_rate * (player_tlbr_2[:,:,0] - player_tlbr_1[:,:,0])
    
    images = tf.reshape(player_cm_2[...,1:], [batch*num_pair, height, width, 1])  # use only mask
    shifts = tf.reshape(tf.stack([shift_y, shift_x], axis=-1), [batch*num_pair, 2])
    #player_m_2_shift = tf_shift_image_bilinear(images, shifts)
    
    player_m_2_shift = tf_shift_image_nearest(images, shifts)
    
    player_m_2_shift = tf.reshape(player_m_2_shift, [batch, num_pair, height, width, 1])
    contact_btw_players = player_cm_1[...,:1] * player_cm_1[...,1:2] * player_m_2_shift
    is_ground = tf.cast(pairs[:,:,1]==0, tf.float32)[:,:,tf.newaxis,tf.newaxis,tf.newaxis] # ground
    all_contacts = contact_btw_players * (1.-is_ground) + contact_w_ground * is_ground
    return all_contacts    



def build_model_explicit_distance_shift(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             from_scratch=False,
             return_feature_ext=False,
             size="SS",
             map_model=None):
    """
    TODO 寸法大丈夫かな？？フルスケールでまわるか？
    対地面の場合、扱いがムズイかも。。
    ボクセル的な奥行きを出力して三次元奥行きで交わり具合を出すのもいい気がする。
    
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    input_rgb = Input(input_shape, name="input_rgb")#256,256,3
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    input_player_positions = Input(shape=[num_boxes,2], name="input_player_positions")
    enc_in = input_rgb
    """
    if map_model is not None:
        averagesize = Lambda(average_box_size)(input_boxes)
        
        player_pos, out_map = map_model([input_rgb, input_boxes, averagesize])
        player_depth = Lambda(lambda x: x[...,1:])(player_pos) # [batch, num_p, 1]
        depth = Lambda(lambda x: x[...,1:])(out_map)
        depth = UpSampling2D(4)(depth)
        enc_in = Lambda(lambda x: tf.concat(x, axis=-1))([input_rgb, depth])
    """
    
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])

    use_coord_conv = False

    if use_coord_conv:
        print("use coords")
        
        x = Lambda(add_coords, name="add_coords")(x)
        x = Lambda(add_high_freq_coords, name="add_high_freq_coords")(x)
    
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)
    decoder_out = outs[-1]
    x = outs[-1]
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    ground_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="ground_mask",)(x)
    
    if size=="SS":
        roi_size = 72
        num_cbr = 3
    elif size=="SM":
        roi_size = 72
        num_cbr = 6
    elif size=="MM":
        roi_size = 108
        num_cbr = 6
        
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    
    features_w_cg = Lambda(lambda x: tf.concat(x, axis=-1))([contact_map, ground_mask, features])
    
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features_w_cg, input_boxes])
    feature_w_mask, box_tlbr_coords = Lambda(larger_crop_resize_layer_2, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1+2, 
                              "crop_size": [roi_size,roi_size], 
                              "add_crop_mask": True,
                              "wide_mode": True,
                              "unbatch": False,
                              })([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    contact_map_player = Lambda(lambda x: x[...,:1])(feature_w_mask)
    contact_map_ground = Lambda(lambda x: x[...,1:2])(feature_w_mask)
    feature_w_mask = Lambda(lambda x: x[...,2:])(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)
    
    for layer_idx in range(num_cbr):#7x3
        feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
    
    player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                       name="player_mask")(feature_w_mask)
    
    player_contact_w_mask = Lambda(lambda x: tf.concat(x, axis=-1))([contact_map_player, player_mask])
    #ground_contact_w_mask = Lambda(lambda x: tf.concat(x, axis=-1))([contact_map_ground, player_mask])
    """
    player_contact_w_ground = Lambda(lambda x: x[0] * x[1])([contact_map_ground, player_mask])
    #all_masks_include_contact = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([player_contact_w_ground, player_contact_w_mask])
    
    pairs_contact = Lambda(contact_btw_selected_pairs_shift, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([player_contact_w_ground, player_contact_w_mask, input_pairs, box_tlbr_coords]) 
            
    """
    
    ### second stage modeling
    pairs_contact, next_inputs = Lambda(contact_btw_selected_pairs_shift_return_next_inputs, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map_ground, player_mask, player_contact_w_mask, input_pairs, box_tlbr_coords]) 
    
    
    pos_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(input_player_positions)
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    image_contact_reduced_stopgrad = Lambda(lambda x: tf.stop_gradient(x[...,tf.newaxis]), name="stop_grad")(pairs_contact_reduced)
    # image_contact_reduced_stopgrad_filled = Lambda(fill_zero_with_average, name="player_ref_fill")(pairs_contact_reduced)
    pairs_xyd = Lambda(extract_xyd_features)([pos_w_dummy, input_pairs])
    ground_mask = Lambda(lambda x: tf.cast(x[:,:,1:2]==0, tf.float32))(input_pairs)
    
    #pairs_xydconf = Lambda(lambda x: tf.concat([tf.math.log(tf.clip_by_value(x[0]/(1-x[0]),1e-7,1e7)), x[1]], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    pairs_xydconf = Lambda(lambda x: tf.concat([x[0]-0.5, x[1]*10], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    
    
    ### 0108 一時的
    #pairs_features_compare = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([pairs_xydconf, pairs_features_gap_multi])
    # pairs_features_compare = Lambda(lambda x: tf.concat([x[0], tf.stop_gradient(x[1])], axis=-1))([pairs_xydconf, output_next_model])
    pairs_features = pairs_xydconf
    
    # predict by image and positions
    pairs_features = Dense(128, activation="relu", name="dense0")(pairs_features)
    pairs_features = Dense(128, activation="relu", name="dense1")(pairs_features)
    pairs_features = Dropout(0.2)(pairs_features)
    player_pred = Dense(1, activation="sigmoid", name="output_contact_label_player")(pairs_features)
    
    # predict by image and features    
    
    player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    #output_contact_label_total = Lambda(lambda x: (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    
    
    
    #player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    output_contact_label_total = Lambda(lambda x: #x[0]*x[2] + x[1]*(1.-x[2])[...,0]
                                        (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    
    #total_pred = Lambda(lambda x:x[...,0], name="output_contact_label_penetration")(total_pred)# 一時的な名称
    
    
    inputs = [input_rgb, input_boxes, input_pairs, input_player_positions, 
              #input_player_in_other_video,
              ]
    outputs = [pairs_contact_reduced, 
               player_pred, 
               output_contact_label_total, 
               contact_map,
               ]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_player": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_total": bce_loss,
              "contact_map": l2_regularization,
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "output_contact_label_player": 1.,#{"z_error": 1e-4,
                    "output_contact_label_total":1,
                    "contact_map": 0.01,
                    }
    metrics = {"output_contact_label": [matthews_correlation_best],
                "output_contact_label_player": [matthews_correlation_best],
                "output_contact_label_total": [matthews_correlation_best, matthews_correlation_03],
                }
    
    
    
    
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, 
                               output_contact_label_total,
                               ])
    return model, sub_model, losses, loss_weights, metrics


def build_model_explicit_distance_multi_view_shift(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             size="SS",
             feature_ext_weight=""):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """

    model, sub_model, model_feature_ext = build_model_explicit_distance(input_shape=input_shape,
                                         backbone=backbone, 
                                         minimum_stride=minimum_stride, 
                                         max_stride = max_stride,
                                         is_train=False,
                                         num_boxes = num_boxes,
                                         from_scratch=True,
                                         size=size,
                                         return_feature_ext=True)
    model.load_weights(feature_ext_weight)
    model.trainable = False
    model_feature_ext.trainable = False
    
    input_rgb = Input(input_shape, name="input_rgb")
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    input_player_positions = Input(shape=[num_boxes,2], name="input_player_positions")
    input_player_in_other_video = Input(shape=[num_boxes], name="input_player_in_other_video", dtype=tf.int32)
    
    #input_warp_p = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_p")
    #input_warp_n = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_n")
    
    features = model_feature_ext(input_rgb, training=False)
    
    x = features
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    ground_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="ground_mask",)(x)
    
    if size=="SS":
        roi_size = 72
        num_cbr = 3
    elif size=="SM":
        roi_size = 72
        num_cbr = 6
    elif size=="MM":
        roi_size = 108
        num_cbr = 10
        
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    
    features_w_cg = Lambda(lambda x: tf.concat(x, axis=-1))([contact_map, ground_mask, features])
    
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features_w_cg, input_boxes])
    feature_w_mask, box_tlbr_coords = Lambda(larger_crop_resize_layer_2, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1+2, 
                              "crop_size": [roi_size,roi_size], 
                              "add_crop_mask": True,
                              "wide_mode": True,
                              "unbatch": False,
                              })([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    contact_map_player = Lambda(lambda x: x[...,:1])(feature_w_mask)
    contact_map_ground = Lambda(lambda x: x[...,1:2])(feature_w_mask)
    feature_w_mask = Lambda(lambda x: x[...,2:])(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)
    
    for layer_idx in range(num_cbr):#7x3
        feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
    
    player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                       name="player_mask")(feature_w_mask)
    
    player_contact_w_mask = Lambda(lambda x: tf.concat(x, axis=-1))([contact_map_player, player_mask])
    #ground_contact_w_mask = Lambda(lambda x: tf.concat(x, axis=-1))([contact_map_ground, player_mask])
    
    ### second stage modeling
    pairs_contact, next_inputs = Lambda(contact_btw_selected_pairs_shift_return_next_inputs, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map_ground, player_mask, player_contact_w_mask, input_pairs, box_tlbr_coords]) 
    
    #next_inputs = Lambda(lambda x: tf.stop_gradient(x))(next_inputs)
    next_x = Lambda(lambda x: tf.reshape(x[0], [tf.shape(x[1])[0]*tf.shape(x[1])[1], roi_size, roi_size, 3]))([next_inputs, input_pairs])
    
    # next_x, skip_connections = effv2_encoder(next_x, is_train, from_scratch=False, model_name = "b2")
    
    for i in range(4):
        next_x = cbr(next_x, 64, kernel=5, stride=2, name=f"nectencoder_cbr{i}")
    
    # next_inputs = Lambda(lambda x: tf.reshape(x[0], [tf.shape(x[1])[0], tf.shape(x[1])[1], roi_size, roi_size, 3]))([next_inputs, input_player_positions])
    next_x = GlobalAveragePooling2D()(next_x)
    next_x = Dense(128, activation="relu", name="dense_next1")(next_x)
    next_x = Dropout(0.2)(next_x)
    next_x = Dense(1, activation="sigmoid")(next_x)
    output_next_model = Lambda(lambda x: tf.reshape(x[0], [tf.shape(x[1])[0], tf.shape(x[1])[1], 1]), name="output_next_model")([next_x, input_pairs])
    output_next_model_stop_grad = Lambda(lambda x: tf.stop_gradient(x), name="stop_next_out")(output_next_model)
    ###




    
    
    
    #### 0108 feature ext
    ### 0108 一時的
    ext_ch = 8
    feature_w_mask_stop = Lambda(lambda x: tf.stop_gradient(x))(feature_w_mask)
    
    feature_for_ext = cbr(feature_w_mask_stop, ext_ch, kernel=7, stride=1, name="player_cbr_ext")
    # player_gap_features = GlobalAveragePooling2D()(feature_for_ext) # batch, num_player, features
    # player_gap_features = Lambda(lambda x: tf.reshape(x[0], [tf.shape(x[1])[0], tf.shape(x[1])[1], ext_ch]))([player_gap_features, input_boxes])
    player_gap_features = Lambda(lambda x: tf.reduce_mean(x, axis=[2,3]))(feature_for_ext)
    
    
    pos_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(input_player_positions)
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    image_contact_reduced_stopgrad = Lambda(lambda x: tf.stop_gradient(x[...,tf.newaxis]), name="stop_grad")(pairs_contact_reduced)
    # image_contact_reduced_stopgrad_filled = Lambda(fill_zero_with_average, name="player_ref_fill")(pairs_contact_reduced)
    pairs_xyd = Lambda(extract_xyd_features)([pos_w_dummy, input_pairs])
    ground_mask = Lambda(lambda x: tf.cast(x[:,:,1:2]==0, tf.float32))(input_pairs)
    
    #pairs_xydconf = Lambda(lambda x: tf.concat([tf.math.log(tf.clip_by_value(x[0]/(1-x[0]),1e-7,1e7)), x[1]], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    pairs_xydconf = Lambda(lambda x: tf.concat([x[0]-0.5, x[1]*10], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    
    ### 0108 一時的
    ## pairs_features = Lambda(lambda x: tf.concat(x, axis=-1))([pairs_features, pairs_xydconf])
    #pairs_features = pairs_xydconf
    
    
    # split batch into two side and end and use features of other side
    """
    player_gap_features_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(player_gap_features)
    
    player_gap_features_w_dummy_s, player_gap_features_w_dummy_e = Lambda(lambda x: tf.split(x, 2, axis=0))(player_gap_features_w_dummy)
    other_video_player_s, other_video_player_e = Lambda(lambda x: tf.split(x, 2, axis=0))(input_player_in_other_video)
    
    # pairs_features_gap_e_for_s = Lambda(extract_xyd_features, arguments={"num_ch": ext_ch})([player_gap_features_w_dummy_e, other_video_player_s])
    # pairs_features_gap_s_for_e = Lambda(extract_xyd_features, arguments={"num_ch": ext_ch})([player_gap_features_w_dummy_s, other_video_player_e])
    player_gap_features_e_for_s = Lambda(lambda x: tf.gather(x[0], x[1], batch_dims=1))([player_gap_features_w_dummy_e, other_video_player_s])
    player_gap_features_s_for_e = Lambda(lambda x: tf.gather(x[0], x[1], batch_dims=1))([player_gap_features_w_dummy_s, other_video_player_e])

    player_gap_features_other_side = Lambda(lambda x: tf.concat(x, axis=0))([player_gap_features_e_for_s, player_gap_features_s_for_e])
    player_gap_features_other_side_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(player_gap_features_other_side)
    
    pairs_features_gap = Lambda(extract_pair_features, arguments={"num_ch": ext_ch})([player_gap_features_w_dummy, input_pairs])
    pairs_features_gap_other_side = Lambda(extract_pair_features, arguments={"num_ch": ext_ch})([player_gap_features_other_side_w_dummy, input_pairs])
    pairs_features_gap_multi = Lambda(lambda x: x[0] * x[1])([pairs_features_gap, pairs_features_gap_other_side])
    """
    ### 0108 一時的
    #pairs_features_compare = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([pairs_xydconf, pairs_features_gap_multi])
    pairs_features_compare = Lambda(lambda x: tf.concat([x[0], tf.stop_gradient(x[1])], axis=-1))([pairs_xydconf, output_next_model])
    pairs_features = pairs_xydconf
    
    # predict by image and positions
    pairs_features = Dense(128, activation="relu", name="dense0")(pairs_features)
    pairs_features = Dense(128, activation="relu", name="dense1")(pairs_features)
    pairs_features = Dropout(0.2)(pairs_features)
    player_pred = Dense(1, activation="sigmoid", name="output_contact_label_player")(pairs_features)
    
    # predict by image and features
    pairs_features_compare = Dense(128, activation="relu", name="dense0c")(pairs_features_compare)
    pairs_features_compare = Dense(128, activation="relu", name="dense1c")(pairs_features_compare)
    pairs_features_compare = Dropout(0.2)(pairs_features_compare)
    player_pred_compare = Dense(1, activation="sigmoid", name="output_contact_label_playerc")(pairs_features_compare)
    
    
    player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    player_pred_compare_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p_compare")(player_pred_compare)
    #output_contact_label_total = Lambda(lambda x: (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    
    
    
    #player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    output_contact_label_total = Lambda(lambda x: #x[0]*x[2] + x[1]*(1.-x[2])[...,0]
                                        (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    output_contact_label_total_compare = Lambda(lambda x: #x[0]*x[2] + x[1]*(1.-x[2])[...,0]
                                        (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_totalc")([output_next_model_stop_grad, player_pred_compare_stopgrad, ground_mask])
    
    #total_pred = Lambda(lambda x:x[...,0], name="output_contact_label_penetration")(total_pred)# 一時的な名称
    
    
    inputs = [input_rgb, input_boxes, input_pairs, input_player_positions, 
              #input_player_in_other_video,
              ]
    outputs = [pairs_contact_reduced, 
               player_pred, 
               player_pred_compare,
               output_contact_label_total, 
               contact_map,
               output_next_model,
               output_contact_label_total_compare]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_player": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_playerc": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_total": bce_loss,
              "output_contact_label_totalc": bce_loss,
              "contact_map": l2_regularization,
              "output_next_model": bce_loss}
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "output_contact_label_player": 1.,#{"z_error": 1e-4,
                    "output_contact_label_playerc": 1.,#{"z_error": 1e-4,
                    "output_contact_label_total":1,
                    "output_contact_label_totalc":1,
                    "contact_map": 0.01,
                    "output_next_model": 1
                    }
    metrics = {"output_contact_label": [matthews_correlation_best],
                "output_next_model": [matthews_correlation_best, matthews_correlation_03],
                "output_contact_label_player": [matthews_correlation_best],
                "output_contact_label_playerc": [matthews_correlation_best],
                "output_contact_label_total": [matthews_correlation_best, matthews_correlation_03],
                "output_contact_label_totalc": [matthews_correlation_best, matthews_correlation_03],
                }
    
    
    
    
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, 
                               output_contact_label_total,
                               output_contact_label_total_compare,
                               ])
    return model, sub_model, losses, loss_weights, metrics
    

def build_model_multiframe_explicit_distance(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             size="SS",
             feature_ext_weight=""):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """

    model, sub_model, model_feature_ext = build_model_explicit_distance(input_shape=input_shape,
                                         backbone=backbone, 
                                         minimum_stride=minimum_stride, 
                                         max_stride = max_stride,
                                         is_train=False,
                                         num_boxes = num_boxes,
                                         from_scratch=True,
                                         size=size,
                                         return_feature_ext=True)
    model.load_weights(feature_ext_weight)
    model.trainable = False
    model_feature_ext.trainable = False
    
    
    input_rgb_pp = Input(input_shape, name="input_rgb_pp")
    input_rgb_p = Input(input_shape, name="input_rgb_p")
    input_rgb_c = Input(input_shape, name="input_rgb_c")
    input_rgb_n = Input(input_shape, name="input_rgb_n")
    input_rgb_nn = Input(input_shape, name="input_rgb_nn")
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    input_player_positions = Input(shape=[num_boxes,2], name="input_player_positions")
    
    #input_warp_p = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_p")
    #input_warp_n = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_n")
    
    pp_features = model_feature_ext(input_rgb_pp, training=False)
    p_features = model_feature_ext(input_rgb_p, training=False)
    c_features = model_feature_ext(input_rgb_c, training=False)
    n_features = model_feature_ext(input_rgb_n, training=False)
    nn_features = model_feature_ext(input_rgb_nn, training=False)
    
    #p_features = Lambda(bilinear_warp_layer)([p_features, input_warp_p])
    #n_features = Lambda(bilinear_warp_layer)([n_features, input_warp_n])
        
    #pcn_features = Lambda(lambda x: tf.concat(x, axis=-1))([pp_features, p_features, c_features, n_features, nn_features])
    pcn_features = Lambda(lambda x: tf.stack(x, axis=-2))([pp_features, p_features, c_features, n_features, nn_features])
    
    
    # TODO 一個ぐらい昆布かませる　以降のカーネル5大丈夫か？
    
    contact_map = Conv3D(1, activation="sigmoid", kernel_size=5, strides=1, 
                        padding="same", 
                        name="contact_map_pre",)(pcn_features)
    contact_map = Lambda(lambda x: x[...,2,:], name = "contact_map")(contact_map) # middle frame
    ground_mask = Conv3D(1, activation="sigmoid", kernel_size=5, strides=1, 
                        padding="same", 
                        name="ground_mask_pre",)(pcn_features)
    ground_mask = Lambda(lambda x: x[...,2,:])(ground_mask)
    
    if size=="SS":
        roi_size = 72
        num_cbr = 3
    elif size=="SM":
        roi_size = 72
        num_cbr = 6
    elif size=="MM":
        roi_size = 108
        num_cbr = 6
        
    num_feature_ch = 24
    features = Conv3D(num_feature_ch, activation="relu", kernel_size=5, strides=1, padding="same", 
                          name="rgb_features")(pcn_features)
    features = Lambda(lambda x: x[...,2,:])(features)
    """
    
    x = c_features
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    ground_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="ground_mask",)(x)
    
    if size=="SS":
        roi_size = 72
        num_cbr = 3
    elif size=="SM":
        roi_size = 72
        num_cbr = 6
    elif size=="MM":
        roi_size = 108
        num_cbr = 6
        
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    
    
    """
    
    
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [roi_size,roi_size], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)

    for layer_idx in range(num_cbr):#7x3
        feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
 
    player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                       name="player_mask")(feature_w_mask)
    player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
               arguments={"num_ch": 1, 
                          "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
    
    
    pos_w_dummy = Lambda(lambda x: tf.concat([tf.zeros_like(x[:,0:1]), x],axis=1))(input_player_positions)
    
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    
    pairs_contact = Lambda(contact_btw_selected_pairs, name="contact_btw_selected_pairs",
                   )([contact_map, all_masks, input_pairs]) 
        
    # TODO ピークの場所に対応する特徴量を引っ張り出してconcatなど。
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    image_contact_reduced_stopgrad = Lambda(lambda x: tf.stop_gradient(x[...,tf.newaxis]), name="stop_grad")(pairs_contact_reduced)
    # image_contact_reduced_stopgrad_filled = Lambda(fill_zero_with_average, name="player_ref_fill")(pairs_contact_reduced)
    pairs_xyd = Lambda(extract_xyd_features)([pos_w_dummy, input_pairs])
    ground_mask = Lambda(lambda x: tf.cast(x[:,:,1:2]==0, tf.float32))(input_pairs)
    
    #pairs_xydconf = Lambda(lambda x: tf.concat([tf.math.log(tf.clip_by_value(x[0]/(1-x[0]),1e-7,1e7)), x[1]], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    pairs_xydconf = Lambda(lambda x: tf.concat([x[0]-0.5, x[1]*10], axis=-1))([image_contact_reduced_stopgrad, pairs_xyd])
    
    ### 0108 一時的
    ## pairs_features = Lambda(lambda x: tf.concat(x, axis=-1))([pairs_features, pairs_xydconf])
    pairs_features = pairs_xydconf
    
    # predict by image and positions
    pairs_features = Dense(128, activation="relu", name="dense0")(pairs_features)
    pairs_features = Dense(128, activation="relu", name="dense1")(pairs_features)
    pairs_features = Dropout(0.2)(pairs_features)
    player_pred = Dense(1, activation="sigmoid", name="output_contact_label_player")(pairs_features)
    #total_pred = Lambda(lambda x: x[0], name="output_contact_label_penetration")([total_pred, player_mask])# 一時的な名称
    
    player_pred_stopgrad = Lambda(lambda x: tf.stop_gradient(x), name="stop_grad_p")(player_pred)
    output_contact_label_total = Lambda(lambda x: #x[0]*x[2] + x[1]*(1.-x[2])[...,0]
                                        (x[0]*x[2] + x[1]*(1.-x[2]))[...,0], name="output_contact_label_total")([image_contact_reduced_stopgrad, player_pred_stopgrad, ground_mask])
    
    inputs = [input_rgb_pp, input_rgb_p, input_rgb_c, input_rgb_n, input_rgb_nn, 
              input_boxes, input_pairs, input_player_positions]
    # inputs = [input_rgb, input_boxes, input_pairs, input_player_positions]
    outputs = [pairs_contact_reduced, 
               player_pred, 
               output_contact_label_total, 
               contact_map]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_player": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_total": bce_loss,
              "contact_map": l2_regularization,
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "output_contact_label_player": 1.,#{"z_error": 1e-4,
                    "output_contact_label_total":1,
                    "contact_map": 0.01,
                    }
    metrics = {"output_contact_label": [matthews_correlation_best],
                "output_contact_label_player": [matthews_correlation_best],
                "output_contact_label_total": [matthews_correlation_best],
                }
    
    model = Model(inputs, outputs)
    sub_model = Model(inputs, [pairs_contact, 
                               pairs_contact_reduced,
                               output_contact_label_total,
                               ])
    return model, sub_model, losses, loss_weights, metrics
    
    


def build_model_multiply_contact(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             from_scratch=False,
             return_feature_ext=False):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    input_rgb = Input(input_shape, name="input_rgb")#256,256,3
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    enc_in = input_rgb
    
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])

    use_coord_conv = False

    if use_coord_conv:
        print("use coords")
        
        x = Lambda(add_coords, name="add_coords")(x)
        x = Lambda(add_high_freq_coords, name="add_high_freq_coords")(x)
    
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)
    decoder_out = outs[-1]
    x = outs[-1]
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    
    num_feature_contact = 8
    ground_mask = Conv2D(num_feature_contact, activation="linear", 
                         kernel_size=3, strides=1, 
                        padding="same", 
                        name="ground_mask",)(x)
    
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [72,72], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)
    mode="direct_mask"
    if mode=="direct_mask":
        #x_0 = feature_w_mask
        #x_1 = AveragePooling2D(2)(x_0)
        #x_2 = AveragePooling2D(2)(x_1)
        for layer_idx in range(3):#7x3
            feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_0 = cbr(x_0, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_1 = cbr(x_1, 32, kernel=7, stride=1, name=f"player_cbrs{layer_idx}")
            #x_2 = cbr(x_2, 32, kernel=7, stride=1, name=f"player_cbrss{layer_idx}")
        
        #x_1 = UpSampling2D(2)(x_1)
        #x_2 = UpSampling2D(4)(x_2)
        #feature_w_mask = Lambda(lambda x: tf.concat(x, axis=-1))([x_0,x_1,x_2])
        
        
        #feature_w_mask = cbr(feature_w_mask, 48, kernel=7, stride=1, name="player_cbr0")
        #for layer_idx in range(3):
        #    feature_w_mask = resblock(feature_w_mask, 48, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        #"""
        player_mask = Conv2D(num_feature_contact, activation="linear", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": num_feature_contact, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        """#12/20一時的に変更。裏表オクルージョンモデル。
        player_mask = Conv2D(2, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 2, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        # is_contactを弱めに学習してもいいのかもしれない？？
        # もしくは共通のグランドマスクを使う。
        #"""
        
        
    elif mode=="ch_attention":
        x = feature_w_mask
        #for layer_idx in range(3):
        #    x = cbr(x, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
        x = cbr(x, 48, kernel=7, stride=1, name="player_cbr0")
        for layer_idx in range(3):
            x = resblock(x, 32, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        
        x = GlobalAveragePooling2D()(x)
        attention_weight = Dense(num_feature_ch, activation="sigmoid", name="ch_attention")(x)
        attention_feature = Lambda(lambda x: x[0][...,:num_feature_ch] * tf.reshape(x[1], [-1,1,1,num_feature_ch]), name="mul_attention")([feature_w_mask, attention_weight])
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(attention_feature)
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                                              arguments={"num_ch": 1, 
                                                         "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        #Reshape((1,1,num_feature_ch), name="ch_attention_reshape")(attention_weight)
        #x_out = Multiply()([features, attention_weight])
        
        
    #"""
    # concat masks, [batch, num_player+1(ground), h, w, 1]
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    
    pairs_contact = Lambda(contact_btw_selected_pairs_logit, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    """#12/20一時的に変更。uraomoteモデル
    #all_masks = Lambda(lambda x: tf.concat([x[...,0:1], x[...,1:2]], axis=1))(player_mask) # concat ground and player_mask at 2nd axis
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    pairs_contact = Lambda(contact_btw_selected_pairs_uraomote, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    #pairs_contact = Lambda(contact_btw_selected_pairs_v2, name="contact_btw_selected_pairs",
    #               #arguments={"num_ch": 1, 
    #               #           "wide_mode": True},
    #               )([contact_map, all_masks, input_pairs]) 
    
    #"""
    
    
    
    # ペア予測がないと、ABC三選手が画像上重なる場合に、A-B, B-Cのみが干渉するケースに対応できないかも…。
    # マルチなロスにする方が自然かな…。この特徴が強すぎると詰むかも？
    add_pairwise_mask = False
    if add_pairwise_mask:
        num_pair_feature = 4
        feature_g = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_g")(features)
        feature_p = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_p")(feature_w_mask)
        feature_p = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize_features",
                                              arguments={"num_ch": num_pair_feature, 
                                                         "wide_mode": True})([feature_p, input_boxes, ground_mask]) 
        features_gp = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([feature_g, feature_p])
        pairs_feature = Lambda(contact_btw_selected_pairs_nomask, name="feature_btw_selected_pairs",
                       )([features_gp, input_pairs]) 
        pairwise_prediction = Conv2D(1, activation="sigmoid", kernel_size=1, strides=1, padding="same", 
                              name="pairwise_prediction")(pairs_feature)
        pairs_contact = Lambda(lambda x: x[0] * x[1], name="multiply_final_preds")([pairs_contact, pairwise_prediction])
    
    
    
    
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    
    inputs = [input_rgb, input_boxes, input_pairs]
    outputs = [pairs_contact_reduced, contact_map]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "contact_map": l2_regularization,
              #"zoom_dev_abs": weighted_dummy_loss
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "contact_map": 0.01,
                    #"zoom_dev_abs": 0.1*4
                    }
    metrics = {"output_contact_label": [matthews_correlation_best]}
    
    
    
    
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, pairs_contact_reduced])
    if not return_feature_ext:
        return model, sub_model, losses, loss_weights, metrics
    else:
        model_feature_ext = Model([input_rgb], [decoder_out])
        return model, sub_model, model_feature_ext

import tensorflow_addons as tfa

def bilinear_warp_layer(inputs):
    features, warp_coords = inputs
    
    def warp_bilinear_tfa(features, warp_coords):
        batch, height, width, _ = tf.unstack(tf.shape(features))
        x_idx = warp_coords[...,:1]
        y_idx = warp_coords[...,1:2]
        #inside_frame_x = tf.math.logical_and(x_idx >= 0., x_idx <= tf.cast(width-1, tf.float32))
        #inside_frame_y = tf.math.logical_and(y_idx >= 0., y_idx <= tf.cast(height-1, tf.float32))
        #inside_frame = tf.math.logical_and(inside_frame_x, inside_frame_y)
        x_idx = tf.clip_by_value(x_idx, 0., tf.cast(width-1, tf.float32))
        y_idx = tf.clip_by_value(y_idx, 0., tf.cast(height-1, tf.float32))
        warp_coords = tf.concat([x_idx, y_idx], axis=-1)
        warped = tfa.image.resampler(features, warp_coords)#easiest way is use tfa library
        return warped#, inside_frame
    
    """def rgbreconst_by_flow(data, default_grid, input_shape):
        #origin_size = tf.cast(tf.stack([data["flow_width"], data["flow_height"]]), tf.float32)
        #target_size = tf.cast(tf.stack([data["img_width"], data["img_height"]]), tf.float32)
        #rate = target_size / origin_size
        #resized_flow = tf.image.resize(tf.stack([data["flow_21"],data["flow_12"]]), 
        #                              (data["img_height"], data["img_width"]), method="bilinear")
        #resized_flow = resized_flow * rate[tf.newaxis, tf.newaxis, tf.newaxis, :] #multiply flow scale
        warp_coord = default_grid + tf.stack([data["flow_21"],data["flow_12"]])
        reconst_rgb = warp_bilinear_tfa(tf.stack([data["rgb_p"],data["rgb_n"]]), warp_coord)#add batch dim
        #data["rgb_reconst_from_p"] = tf.reshape(reconst_rgb[0], [data["img_height"], data["img_width"], 3])# - data["rgb"]
        #data["rgb_reconst_from_n"] = tf.reshape(reconst_rgb[1], [data["img_height"], data["img_width"], 3])# - data["rgb"]
        data["rgb_reconst_from_p"] = tf.reshape(reconst_rgb[0], [input_shape[0], input_shape[1], 3])# - data["rgb"]
        data["rgb_reconst_from_n"] = tf.reshape(reconst_rgb[1], [input_shape[0], input_shape[1], 3])# - data["rgb"]
        
        #data["rgb_reconst_prev"] = data["rgb_prev"]
        #data["rgb_reconst_next"] = data["rgb_next"]
        return data
    """
    
    reconst_features = warp_bilinear_tfa(features, warp_coords)
    reconst_features = tf.reshape(reconst_features, tf.shape(features))
    return reconst_features


def build_model_multiframe(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             feature_ext_weight=""):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    
    model, sub_model, model_feature_ext = build_model(input_shape=input_shape,
                                         backbone=backbone, 
                                         minimum_stride=2, 
                                         max_stride = 64,
                                         is_train=False,
                                         num_boxes = num_boxes,
                                         from_scratch=True,
                                         return_feature_ext=True)
    model.load_weights(feature_ext_weight)
    model.trainable = False
    model_feature_ext.trainable = False
    
    
    input_rgb_pp = Input(input_shape, name="input_rgb_pp")
    input_rgb_p = Input(input_shape, name="input_rgb_p")
    input_rgb_c = Input(input_shape, name="input_rgb_c")
    input_rgb_n = Input(input_shape, name="input_rgb_n")
    input_rgb_nn = Input(input_shape, name="input_rgb_nn")
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    
    #input_warp_p = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_p")
    #input_warp_n = Input([input_shape[0]//minimum_stride, input_shape[1]//minimum_stride, 2], name="input_warp_n")
    
    pp_features = model_feature_ext(input_rgb_pp, training=False)
    p_features = model_feature_ext(input_rgb_p, training=False)
    c_features = model_feature_ext(input_rgb_c, training=False)
    n_features = model_feature_ext(input_rgb_n, training=False)
    nn_features = model_feature_ext(input_rgb_nn, training=False)
    
    #p_features = Lambda(bilinear_warp_layer)([p_features, input_warp_p])
    #n_features = Lambda(bilinear_warp_layer)([n_features, input_warp_n])
        
    #pcn_features = Lambda(lambda x: tf.concat(x, axis=-1))([pp_features, p_features, c_features, n_features, nn_features])
    pcn_features = Lambda(lambda x: tf.stack(x, axis=-2))([pp_features, p_features, c_features, n_features, nn_features])
    
    contact_map = Conv3D(1, activation="sigmoid", kernel_size=5, strides=1, 
                        padding="same", 
                        name="contact_map_pre",)(pcn_features)
    contact_map = Lambda(lambda x: x[...,2,:], name = "contact_map")(contact_map) # middle frame
    ground_mask = Conv3D(1, activation="sigmoid", kernel_size=5, strides=1, 
                        padding="same", 
                        name="ground_mask_pre",)(pcn_features)
    ground_mask = Lambda(lambda x: x[...,2,:])(ground_mask)
    num_feature_ch = 24
    features = Conv3D(num_feature_ch, activation="relu", kernel_size=5, strides=1, padding="same", 
                          name="rgb_features")(pcn_features)
    features = Lambda(lambda x: x[...,2,:])(features)
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [72,72], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)
    mode="direct_mask"
    if mode=="direct_mask":
        #x_0 = feature_w_mask
        #x_1 = AveragePooling2D(2)(x_0)
        #x_2 = AveragePooling2D(2)(x_1)
        
        for layer_idx in range(3):
            feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_0 = cbr(x_0, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_1 = cbr(x_1, 32, kernel=7, stride=1, name=f"player_cbrs{layer_idx}")
            #x_2 = cbr(x_2, 32, kernel=7, stride=1, name=f"player_cbrss{layer_idx}")
        
        #x_1 = UpSampling2D(2)(x_1)
        #x_2 = UpSampling2D(4)(x_2)
        #feature_w_mask = Lambda(lambda x: tf.concat(x, axis=-1))([x_0,x_1,x_2])
        
        
        #feature_w_mask = cbr(feature_w_mask, 48, kernel=7, stride=1, name="player_cbr0")
        #for layer_idx in range(3):
        #    feature_w_mask = resblock(feature_w_mask, 48, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        #"""
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 1, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        """#12/20一時的に変更。裏表オクルージョンモデル。
        player_mask = Conv2D(2, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 2, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        # is_contactを弱めに学習してもいいのかもしれない？？
        # もしくは共通のグランドマスクを使う。
        #"""
        
        
    elif mode=="ch_attention":
        x = feature_w_mask
        #for layer_idx in range(3):
        #    x = cbr(x, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
        x = cbr(x, 48, kernel=7, stride=1, name="player_cbr0")
        for layer_idx in range(3):
            x = resblock(x, 32, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        
        x = GlobalAveragePooling2D()(x)
        attention_weight = Dense(num_feature_ch, activation="sigmoid", name="ch_attention")(x)
        attention_feature = Lambda(lambda x: x[0][...,:num_feature_ch] * tf.reshape(x[1], [-1,1,1,num_feature_ch]), name="mul_attention")([feature_w_mask, attention_weight])
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(attention_feature)
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                                              arguments={"num_ch": 1, 
                                                         "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        #Reshape((1,1,num_feature_ch), name="ch_attention_reshape")(attention_weight)
        #x_out = Multiply()([features, attention_weight])
        
        
    #"""
    # concat masks, [batch, num_player+1(ground), h, w, 1]
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    
    pairs_contact = Lambda(contact_btw_selected_pairs, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    """#12/20一時的に変更。uraomoteモデル
    #all_masks = Lambda(lambda x: tf.concat([x[...,0:1], x[...,1:2]], axis=1))(player_mask) # concat ground and player_mask at 2nd axis
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    pairs_contact = Lambda(contact_btw_selected_pairs_uraomote, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    #pairs_contact = Lambda(contact_btw_selected_pairs_v2, name="contact_btw_selected_pairs",
    #               #arguments={"num_ch": 1, 
    #               #           "wide_mode": True},
    #               )([contact_map, all_masks, input_pairs]) 
    
    #"""
    
    
    
    # ペア予測がないと、ABC三選手が画像上重なる場合に、A-B, B-Cのみが干渉するケースに対応できないかも…。
    # マルチなロスにする方が自然かな…。この特徴が強すぎると詰むかも？
    add_pairwise_mask = False
    if add_pairwise_mask:
        num_pair_feature = 4
        feature_g = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_g")(features)
        feature_p = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_p")(feature_w_mask)
        feature_p = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize_features",
                                              arguments={"num_ch": num_pair_feature, 
                                                         "wide_mode": True})([feature_p, input_boxes, ground_mask]) 
        features_gp = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([feature_g, feature_p])
        pairs_feature = Lambda(contact_btw_selected_pairs_nomask, name="feature_btw_selected_pairs",
                       )([features_gp, input_pairs]) 
        pairwise_prediction = Conv2D(1, activation="sigmoid", kernel_size=1, strides=1, padding="same", 
                              name="pairwise_prediction")(pairs_feature)
        pairs_contact = Lambda(lambda x: x[0] * x[1], name="multiply_final_preds")([pairs_contact, pairwise_prediction])
    
    
    
    
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    
    inputs = [input_rgb_pp, input_rgb_p, input_rgb_c, input_rgb_n, input_rgb_nn, input_boxes, input_pairs,
              ]
    outputs = [pairs_contact_reduced, contact_map]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "contact_map": l2_regularization,
              #"zoom_dev_abs": weighted_dummy_loss
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "contact_map": 0.01,
                    #"zoom_dev_abs": 0.1*4
                    }
    metrics = {"output_contact_label": [matthews_correlation_best]}
    
        
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, pairs_contact_reduced])
    return model, sub_model, losses, loss_weights, metrics




def build_model_multi(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             from_scratch=False):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    input_rgb = Input(input_shape, name="input_rgb")#256,256,3
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    enc_in = input_rgb
    
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])

    use_coord_conv = False

    if use_coord_conv:
        print("use coords")
        
        x = Lambda(add_coords, name="add_coords")(x)
        x = Lambda(add_high_freq_coords, name="add_high_freq_coords")(x)
    
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)

    x = outs[-1]
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    ground_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="ground_mask",)(x)
    
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [72,72], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)
    mode="direct_mask"
    if mode=="direct_mask":
        #x_0 = feature_w_mask
        #x_1 = AveragePooling2D(2)(x_0)
        #x_2 = AveragePooling2D(2)(x_1)
        
        for layer_idx in range(3):
            feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_0 = cbr(x_0, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
            #x_1 = cbr(x_1, 32, kernel=7, stride=1, name=f"player_cbrs{layer_idx}")
            #x_2 = cbr(x_2, 32, kernel=7, stride=1, name=f"player_cbrss{layer_idx}")
        
        #x_1 = UpSampling2D(2)(x_1)
        #x_2 = UpSampling2D(4)(x_2)
        #feature_w_mask = Lambda(lambda x: tf.concat(x, axis=-1))([x_0,x_1,x_2])
        
        
        #feature_w_mask = cbr(feature_w_mask, 48, kernel=7, stride=1, name="player_cbr0")
        #for layer_idx in range(3):
        #    feature_w_mask = resblock(feature_w_mask, 48, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        #"""
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 1, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        """#12/20一時的に変更。裏表オクルージョンモデル。
        player_mask = Conv2D(2, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 2, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        # is_contactを弱めに学習してもいいのかもしれない？？
        # もしくは共通のグランドマスクを使う。
        #"""
        
        
    elif mode=="ch_attention":
        x = feature_w_mask
        #for layer_idx in range(3):
        #    x = cbr(x, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
        x = cbr(x, 48, kernel=7, stride=1, name="player_cbr0")
        for layer_idx in range(3):
            x = resblock(x, 32, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        
        x = GlobalAveragePooling2D()(x)
        attention_weight = Dense(num_feature_ch, activation="sigmoid", name="ch_attention")(x)
        attention_feature = Lambda(lambda x: x[0][...,:num_feature_ch] * tf.reshape(x[1], [-1,1,1,num_feature_ch]), name="mul_attention")([feature_w_mask, attention_weight])
        player_mask = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(attention_feature)
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                                              arguments={"num_ch": 1, 
                                                         "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        #Reshape((1,1,num_feature_ch), name="ch_attention_reshape")(attention_weight)
        #x_out = Multiply()([features, attention_weight])
        
        
    #"""
    # concat masks, [batch, num_player+1(ground), h, w, 1]
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    
    pairs_contact = Lambda(contact_btw_selected_pairs, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    """#12/20一時的に変更。uraomoteモデル
    #all_masks = Lambda(lambda x: tf.concat([x[...,0:1], x[...,1:2]], axis=1))(player_mask) # concat ground and player_mask at 2nd axis
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    pairs_contact = Lambda(contact_btw_selected_pairs_uraomote, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    #pairs_contact = Lambda(contact_btw_selected_pairs_v2, name="contact_btw_selected_pairs",
    #               #arguments={"num_ch": 1, 
    #               #           "wide_mode": True},
    #               )([contact_map, all_masks, input_pairs]) 
    
    #"""
    
    
    
    # ペア予測がないと、ABC三選手が画像上重なる場合に、A-B, B-Cのみが干渉するケースに対応できないかも…。
    # マルチなロスにする方が自然かな…。この特徴が強すぎると詰むかも？
    add_pairwise_mask = True
    if add_pairwise_mask:
        num_pair_feature = 6
        feature_g = Conv2D(num_pair_feature, activation="relu", kernel_size=7, strides=1, padding="same", 
                              name="features_g")(features)
        #add white(mask)
        feature_g = Lambda(lambda x: tf.concat([x, tf.ones_like(x[...,:1])], axis=-1))(feature_g)
        
        feature_p = Conv2D(num_pair_feature, activation="relu", kernel_size=7, strides=1, padding="same", 
                              name="features_p")(feature_w_mask)
        #add white(mask)
        feature_p = Lambda(lambda x: tf.concat([x, tf.ones_like(x[...,:1])], axis=-1))(feature_p)
        feature_p = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize_features",
                                              arguments={"num_ch": num_pair_feature+1, 
                                                         "wide_mode": True})([feature_p, input_boxes, ground_mask]) 
        features_gp = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([feature_g, feature_p])
        #pairs_feature = Lambda(contact_btw_selected_pairs_nomask, name="feature_btw_selected_pairs",
        #               )([features_gp, input_pairs]) 
        pairs_feature, player_12_mask = Lambda(contact_btw_selected_pairs_feature_only_exist, name="feature_btw_selected_pairs",
                       )([features_gp, input_pairs]) 
        
        
        

        
        pairwise_prediction_2 = Conv2D(1, activation="sigmoid", kernel_size=1, strides=1, padding="same", 
                              name="pairwise_prediction")(pairs_feature)
        
        pairwise_prediction_2 = Lambda(lambda x: x[0] * x[1])([pairwise_prediction_2, player_12_mask])
    
    
    
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)

    pairs_contact_reduced_concat = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label_concat")(pairwise_prediction_2)

    
    inputs = [input_rgb, input_boxes, input_pairs]
    outputs = [pairs_contact_reduced, pairs_contact_reduced_concat, contact_map]
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              "output_contact_label_concat": bce_loss,#"z_error": weighted_dummy_loss,
              "contact_map": l2_regularization,
              #"zoom_dev_abs": weighted_dummy_loss
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    "output_contact_label_concat": 1.,#{"z_error": 1e-4,
                    "contact_map": 0.01,
                    #"zoom_dev_abs": 0.1*4
                    }
    metrics = {"output_contact_label": [matthews_correlation_best],
               "output_contact_label_concat": [matthews_correlation_best]}
    
    
    
    
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, pairs_contact_reduced_concat])
    return model, sub_model, losses, loss_weights, metrics

def _build_model(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             from_scratch=False):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    input_rgb = Input(input_shape, name="input_rgb")#256,256,3
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    enc_in = input_rgb
    
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])

    use_coord_conv = False

    if use_coord_conv:
        print("use coords")
        
        x = Lambda(add_coords, name="add_coords")(x)
        x = Lambda(add_high_freq_coords, name="add_high_freq_coords")(x)
    
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)

    x = outs[-1]
    pcontact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="pcontact_map",)(x)
    gcontact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="gcontact_map",)(x)
    
    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [72,72], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)
    mode="direct_mask"
    if mode=="direct_mask":
        for layer_idx in range(3):
            feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
        #feature_w_mask = cbr(feature_w_mask, 48, kernel=7, stride=1, name="player_cbr0")
        #for layer_idx in range(3):
        #    feature_w_mask = resblock(feature_w_mask, 48, kernel=7, stride=1, name=f"player_resblock{layer_idx}", use_se=False)
        
        #"""
        player_mask = Conv2D(2, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 2, 
                              "wide_mode": True})([player_mask, input_boxes, gcontact_map]) 
        
        """#12/20一時的に変更。裏表オクルージョンモデル。
        player_mask = Conv2D(2, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="player_mask")(feature_w_mask)
        # resize back to original scale, and reshape from [batch*num_box, h, w, 1] to [batch, num_box, h, w, 1]
        player_mask = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize",
                   arguments={"num_ch": 2, 
                              "wide_mode": True})([player_mask, input_boxes, ground_mask]) 
        
        # is_contactを弱めに学習してもいいのかもしれない？？
        # もしくは共通のグランドマスクを使う。
        #"""
        

    #"""
    # concat masks, [batch, num_player+1(ground), h, w, 1]
    all_masks_g = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1][...,:1]], axis=1))([gcontact_map, player_mask])
    all_masks_p = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1][...,1:2]], axis=1))([pcontact_map, player_mask]) # not use 1st player
    
    pairs_contact = Lambda(contact_btw_selected_pairs_each, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([all_masks_p, all_masks_g, input_pairs]) 
    
    """#12/20一時的に変更。uraomoteモデル
    #all_masks = Lambda(lambda x: tf.concat([x[...,0:1], x[...,1:2]], axis=1))(player_mask) # concat ground and player_mask at 2nd axis
    all_masks = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([ground_mask, player_mask])
    pairs_contact = Lambda(contact_btw_selected_pairs_uraomote, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([contact_map, all_masks, input_pairs]) 
    
    #pairs_contact = Lambda(contact_btw_selected_pairs_v2, name="contact_btw_selected_pairs",
    #               #arguments={"num_ch": 1, 
    #               #           "wide_mode": True},
    #               )([contact_map, all_masks, input_pairs]) 
    
    #"""
    
    
    
    # ペア予測がないと、ABC三選手が画像上重なる場合に、A-B, B-Cのみが干渉するケースに対応できないかも…。
    # マルチなロスにする方が自然かな…。この特徴が強すぎると詰むかも？
    add_pairwise_mask = False
    if add_pairwise_mask:
        num_pair_feature = 4
        feature_g = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_g")(features)
        feature_p = Conv2D(num_pair_feature, activation="relu", kernel_size=3, strides=1, padding="same", 
                              name="features_p")(feature_w_mask)
        feature_p = Lambda(inv_larger_crop_resize_layer, name="inv_wide_crop_resize_features",
                                              arguments={"num_ch": num_pair_feature, 
                                                         "wide_mode": True})([feature_p, input_boxes, ground_mask]) 
        features_gp = Lambda(lambda x: tf.concat([x[0][:,tf.newaxis], x[1]], axis=1))([feature_g, feature_p])
        pairs_feature = Lambda(contact_btw_selected_pairs_nomask, name="feature_btw_selected_pairs",
                       )([features_gp, input_pairs]) 
        pairwise_prediction = Conv2D(1, activation="sigmoid", kernel_size=1, strides=1, padding="same", 
                              name="pairwise_prediction")(pairs_feature)
        pairs_contact = Lambda(lambda x: x[0] * x[1], name="multiply_final_preds")([pairs_contact, pairwise_prediction])
    
    
    
    
    pairs_contact_reduced = Lambda(lambda x: tf.math.reduce_max(x, axis=[2,3,4]), name="output_contact_label")(pairs_contact)
    
    inputs = [input_rgb, input_boxes, input_pairs]
    outputs = [pairs_contact_reduced]#, contact_map]
    
    losses = {"output_contact_label": bce_loss,#"z_error": weighted_dummy_loss,
              #"contact_map": l2_regularization,
              #"zoom_dev_abs": weighted_dummy_loss
              }
    loss_weights = {"output_contact_label": 1.,#{"z_error": 1e-4,
                    #"contact_map": 0.01,
                    #"zoom_dev_abs": 0.1*4
                    }
    metrics = {"output_contact_label": [matthews_correlation_best]}
    
    
    
    
    model = Model(inputs, outputs)
    
    sub_model = Model(inputs, [pairs_contact, pairs_contact_reduced])
    return model, sub_model, losses, loss_weights, metrics

def build_model_nomask(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=2, 
             max_stride = 64,
             is_train=True,
             num_boxes = None,
             from_scratch=False):
    """
    mask model is better than simple dense
    """
    input_rgb = Input(input_shape, name="input_rgb")#256,256,3
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    input_pairs = Input(shape=[None,2], name="input_pairs", dtype=tf.int32)
    enc_in = input_rgb
    
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])

    use_coord_conv = False

    if use_coord_conv:
        print("use coords")
        
        x = Lambda(add_coords, name="add_coords")(x)
        x = Lambda(add_high_freq_coords, name="add_high_freq_coords")(x)
    
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)

    x = outs[-1]
    contact_map = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="contact_map",)(x)
    #ground_mask = Conv2D(1, activation="sigmoid", kernel_size=3, strides=1, 
    #                    padding="same", 
    #                    name="ground_mask",)(x)
    ground_feature = GlobalAveragePooling2D()(x)
    ground_feature = Dense(32, activation="relu", name="ground_feature")(ground_feature)
    ground_feature = Lambda(lambda x: tf.reshape(x, [-1,1,32]))(ground_feature)

    num_feature_ch = 24
    features = Conv2D(num_feature_ch, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
    feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, 
                            name="add_box_mask")([features, input_boxes])
    feature_w_mask = Lambda(larger_crop_resize_layer, name="wide_crop_resize",
                   arguments={"num_ch": num_feature_ch+1, 
                              "crop_size": [96,64], 
                              "add_crop_mask": True,
                              "wide_mode": True})([feature_w_mask, input_boxes]) 
    feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
    # ch = num_feature_ch + 2(one is self_mask, the other is other_mask)
    for layer_idx in range(3):
            feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
    player_feature = GlobalAveragePooling2D()(feature_w_mask)
    player_feature = Dense(32, activation="relu", name="player_feature")(player_feature)
    player_feature = Lambda(lambda x: tf.reshape(x, [-1,20,32]))(player_feature)

    all_features =  Lambda(lambda x: tf.concat(x, axis=1))([ground_feature, player_feature])
    pairs_contact = Lambda(contact_btw_selected_pairs_nomask, name="contact_btw_selected_pairs",
                   #arguments={"num_ch": 1, 
                   #           "wide_mode": True},
                   )([all_features, input_pairs]) 
    
    pairs_contact = Lambda(lambda x: tf.reshape(x, [-1,40,64]))(pairs_contact)
    pairs_contact = Dense(1, activation="sigmoid", name="pairs_contact")(pairs_contact)
    pairs_contact = Lambda(lambda x: tf.reshape(x, [-1,40]), name="output_contact_label")(pairs_contact)
    
    
    inputs = [input_rgb, input_boxes, input_pairs]
    outputs = [pairs_contact]
    losses = {"output_contact_label": bce_loss}#"z_error": weighted_dummy_loss,
              #"xy_error": weighted_dummy_loss,
              #"zoom_dev_abs": weighted_dummy_loss}
    loss_weights = {"output_contact_label": 1.}#{"z_error": 1e-4,
                    #"xy_error": 100.0*4,
                    #"zoom_dev_abs": 0.1*4}
    metrics = {"output_contact_label": [matthews_correlation_best]}
    
    
    model = Model(inputs, outputs)
    
    sub_model = None#Model(inputs, [pairs_contact, pairs_contact_reduced])
    return model, sub_model, losses, loss_weights, metrics


if __name__ == "__main__":

    model, _, _, _ = build_model(from_scratch=True)
    print(model.summary())
    
    
    
