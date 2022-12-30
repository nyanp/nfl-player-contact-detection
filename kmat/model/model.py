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
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv2D, Conv2DTranspose, BatchNormalization, Activation, GlobalAveragePooling2D, Lambda, Input, Concatenate, Add, UpSampling2D, LeakyReLU, ZeroPadding2D,Multiply, DepthwiseConv2D, MaxPooling2D, LayerNormalization
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

def matthews_correlation_fixed(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred>threshold, y_pred.dtype)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.-y_pred))
    fp = tf.reduce_sum((1.-y_true) * y_pred)
    tn = tf.reduce_sum((1.-y_true) * (1.-y_pred))
    score = (tp*tn - fp*fn) / tf.math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-7)
    return score

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
    
    

def build_model(input_shape=(256,256,3),
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
        for layer_idx in range(3):
            feature_w_mask = cbr(feature_w_mask, 32, kernel=7, stride=1, name=f"player_cbr{layer_idx}")
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