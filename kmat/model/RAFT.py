# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:55:09 2021

@author: k_mat
"""
import os
import sys
import math
from collections import OrderedDict
import time

#import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv2D, Conv2DTranspose, BatchNormalization, Activation, GlobalAveragePooling2D, Lambda, Input, Concatenate, Add, UpSampling2D, LeakyReLU, ZeroPadding2D,Multiply, DepthwiseConv2D, MaxPooling2D, LayerNormalization
from tensorflow.keras.layers import GRU, Bidirectional
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

#RAFT, DROID SLAM likeに
#TODO Flowの大きさを一方向に制限する?
#validエリアのストップグラッド
#flow初期値与える。

#loss decay ミス修正
#コルマット高速。二次元化。
#census loss追加。
#?? maskはstop gradient したけど、分子側は残す方がいい可能性もある？？？


def select_normalization(norm_type="batch"):
    if norm_type=="instance":
        Norm = tfa.layers.InstanceNormalization
    elif norm_type=="batch":
        Norm = BatchNormalization
    else:
        raise Exception("to be imple")
    return Norm

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, norm_type='batch', strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_type = norm_type
        self.strides = strides

        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')
        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')

        Norm = select_normalization(norm_type)
        
        self.norm1 = Norm()
        self.norm2 = Norm()
        if strides == 1:
            self.downsample = None
        else:
            self.downsample = tf.keras.Sequential([Conv2D(filters, kernel_size=1, strides=strides),
                                                   Norm(),
                                                   ])

    def call(self, inputs, training):
        fx = inputs
        fx = tf.nn.relu(self.norm1(self.conv1(fx), training=training))
        fx = tf.nn.relu(self.norm2(self.conv2(fx), training=training))

        if self.downsample:
            inputs = self.downsample(inputs, training=training)

        return tf.nn.relu(inputs + fx)

class Res_Encoder(tf.keras.layers.Layer):#Model):
    def __init__(self, dims=128, norm_type="batch", **kwargs):
        super().__init__(**kwargs)
        
        Norm = select_normalization(norm_type)
        
        self.dims = dims
        self.stem = Conv2D(64, 7, 2, 'same', name="stem_conv")
        self.norm_1 = Norm(name="setm_norm")
        self.act_1 = Activation("relu", name="stem_act")
        self.res_blocks = [ResBlock(64, norm_type, strides=1, name="resblock_1a"),
                           ResBlock(64, norm_type, strides=1, name="resblock_1b"),
                           ResBlock(96, norm_type, strides=2, name="resblock_2a"),
                           ResBlock(96, norm_type, strides=1, name="resblock_2b"),
                           ResBlock(128, norm_type, strides=2, name="resblock_3a"),
                           ResBlock(128, norm_type, strides=1, name="resblock_3b"),
                           ]
        self.conv_out = Conv2D(dims, 1)        
        
    def call(self, inputs, training):
        x = self.stem(inputs)
        x = self.norm_1(x)
        x = self.act_1(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.conv_out(x)
        return x


class ConvGRU(tf.keras.layers.Layer):
    def __init__(self, filters=128, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.convz = Conv2D(filters, 3, 1, 'same')
        self.convr = Conv2D(filters, 3, 1, 'same')
        self.convq = Conv2D(filters, 3, 1, 'same')
    
    def call(self, inputs):
        h, x = inputs
        hx = tf.concat([h, x], axis=-1)
        z = tf.nn.sigmoid(self.convz(hx))
        r = tf.nn.sigmoid(self.convr(hx))
        q = tf.nn.tanh(self.convq(tf.concat([r*h, x], axis=-1)))
        h = (1-z)*h + z*q
        return h

def warp_bilinear_lessint(features, warp_coords, only_mask=False):
    """
    int使用すると妙に遅い。最小限にとどめると少し早くなる。なぜ・・？
    gather_ndも遅い。
    tfaのresamplerもなぜか遅い。最新バージョンだと気にしない速さの様子＠kaggle notebook
    """
    batch, height, width, num_ch = tf.unstack(tf.shape(features))
    batch, height_w, width_w, _ = tf.unstack(tf.shape(warp_coords))
    height_f32 = tf.cast(height, tf.float32)
    width_f32 = tf.cast(width, tf.float32)
    
    x_idx = warp_coords[...,:1]
    y_idx = warp_coords[...,1:2]
    
    inside_frame_x = tf.math.logical_and(x_idx >= 0., x_idx <= tf.cast(width-1, tf.float32))
    inside_frame_y = tf.math.logical_and(y_idx >= 0., y_idx <= tf.cast(height-1, tf.float32))
    inside_frame = tf.math.logical_and(inside_frame_x, inside_frame_y)
    if only_mask:
        return inside_frame
    
    x_idx = tf.clip_by_value(x_idx, 0., tf.cast(width-1, tf.float32))
    y_idx = tf.clip_by_value(y_idx, 0., tf.cast(height-1, tf.float32))

    left_idx = tf.floor(x_idx)
    right_idx = tf.math.ceil(x_idx)
    top_idx = tf.floor(y_idx)
    bottom_idx = tf.math.ceil(y_idx)

    #left_idx = tf.clip_by_value(left_idx, 0., tf.cast(width-1, tf.float32))
    #right_idx = tf.clip_by_value(right_idx, 0., tf.cast(width-1, tf.float32))
    #top_idx = tf.clip_by_value(top_idx, 0., tf.cast(height-1, tf.float32))
    #bottom_idx = tf.clip_by_value(bottom_idx, 0., tf.cast(height-1, tf.float32))
    
    left_weight = right_idx - x_idx
    right_weight = 1.0 - left_weight
    top_weight = bottom_idx - y_idx
    bottom_weight = 1.0 - top_weight
    
    tl_weight = top_weight * left_weight
    tr_weight = top_weight * right_weight
    bl_weight = bottom_weight * left_weight
    br_weight = bottom_weight * right_weight
    
    tl_idx = top_idx * width_f32 + left_idx
    tr_idx = top_idx * width_f32 + right_idx
    bl_idx = bottom_idx * width_f32 + left_idx
    br_idx = bottom_idx * width_f32 + right_idx
    
    features = tf.reshape(features, [batch, height*width, num_ch])
    tl_idx = tf.reshape(tl_idx, [batch, height_w*width_w])
    tr_idx = tf.reshape(tr_idx, [batch, height_w*width_w])
    bl_idx = tf.reshape(bl_idx, [batch, height_w*width_w])
    br_idx = tf.reshape(br_idx, [batch, height_w*width_w])
    
    tl_weight = tf.reshape(tl_weight, [batch, height_w*width_w, 1])
    tr_weight = tf.reshape(tr_weight, [batch, height_w*width_w, 1])
    bl_weight = tf.reshape(bl_weight, [batch, height_w*width_w, 1])
    br_weight = tf.reshape(br_weight, [batch, height_w*width_w, 1])
    
    warp_indices = tf.cast(tf.concat([tl_idx, tr_idx, bl_idx, br_idx], axis=-1), tf.int32)
    
    tl_feature, tr_feature, bl_feature, br_feature = tf.split(tf.gather(features, warp_indices, batch_dims=1), 4, axis=1)

    warped = tl_weight * tl_feature + tr_weight * tr_feature + bl_weight * bl_feature + br_weight * br_feature
    warped = tf.reshape(warped, [batch, height_w, width_w, num_ch])
    return warped, inside_frame


def warp_bilinear(features, warp_coords, only_mask=False):
    """
    features : float
        [batch, height, width, num_features]
    warp_coords : float
        [batch, height_warp, width_warp, 2(x,y)]
    
    bilnear sampling
        
    メモ：レンジ外のマスクもアウトプットする。
    return:        
        warped_feature [batch, height_warp, width_warp, num_features]
        inside_frame_mask [batch, height_warp, width_warp, 1]
    """
    batch, height, width, _ = tf.unstack(tf.shape(features))
    x_idx = warp_coords[...,:1]
    y_idx = warp_coords[...,1:2]
    inside_frame_x = tf.math.logical_and(x_idx >= 0., x_idx <= tf.cast(width-1, tf.float32))
    inside_frame_y = tf.math.logical_and(y_idx >= 0., y_idx <= tf.cast(height-1, tf.float32))
    inside_frame = tf.math.logical_and(inside_frame_x, inside_frame_y)
    if only_mask:
        return inside_frame

    x_idx = tf.clip_by_value(x_idx, 0., tf.cast(width-1, tf.float32))
    y_idx = tf.clip_by_value(y_idx, 0., tf.cast(height-1, tf.float32))

    left_idx = tf.floor(x_idx)
    right_idx = tf.math.ceil(x_idx)
    top_idx = tf.floor(y_idx)
    bottom_idx = tf.math.ceil(y_idx)

    #left_idx = tf.clip_by_value(left_idx, 0., tf.cast(width-1, tf.float32))
    #right_idx = tf.clip_by_value(right_idx, 0., tf.cast(width-1, tf.float32))
    #top_idx = tf.clip_by_value(top_idx, 0., tf.cast(height-1, tf.float32))
    #bottom_idx = tf.clip_by_value(bottom_idx, 0., tf.cast(height-1, tf.float32))
    
    left_weight = right_idx - x_idx
    right_weight = 1.0 - left_weight
    top_weight = bottom_idx - y_idx
    bottom_weight = 1.0 - top_weight
    
    tl_weight = top_weight * left_weight
    tr_weight = top_weight * right_weight
    bl_weight = bottom_weight * left_weight
    br_weight = bottom_weight * right_weight
    
    top_idx = tf.cast(top_idx, tf.int32)
    left_idx = tf.cast(left_idx, tf.int32)
    bottom_idx = tf.cast(bottom_idx, tf.int32)
    right_idx = tf.cast(right_idx, tf.int32)

    top_left = tf.concat([top_idx, left_idx], axis=-1)
    top_right = tf.concat([top_idx, right_idx], axis=-1)
    bottom_left = tf.concat([bottom_idx, left_idx], axis=-1)
    bottom_right = tf.concat([bottom_idx, right_idx], axis=-1)
    #print(top_left)
    tl_feature = tf.gather_nd(features, top_left, batch_dims=1)
    tr_feature = tf.gather_nd(features, top_right, batch_dims=1)
    bl_feature = tf.gather_nd(features, bottom_left, batch_dims=1)
    br_feature = tf.gather_nd(features, bottom_right, batch_dims=1)

    warped = tl_weight * tl_feature + tr_weight * tr_feature + bl_weight * bl_feature + br_weight * br_feature
    return warped, inside_frame

def warp_bilinear_tfa(features, warp_coords):
    batch, height, width, _ = tf.unstack(tf.shape(features))
    x_idx = warp_coords[...,:1]
    y_idx = warp_coords[...,1:2]
    inside_frame_x = tf.math.logical_and(x_idx >= 0., x_idx <= tf.cast(width-1, tf.float32))
    inside_frame_y = tf.math.logical_and(y_idx >= 0., y_idx <= tf.cast(height-1, tf.float32))
    inside_frame = tf.math.logical_and(inside_frame_x, inside_frame_y)
    x_idx = tf.clip_by_value(x_idx, 0., tf.cast(width-1, tf.float32))
    y_idx = tf.clip_by_value(y_idx, 0., tf.cast(height-1, tf.float32))
    warp_coords = tf.concat([x_idx, y_idx], axis=-1)
    warped = tfa.image.resampler(features, warp_coords)#easiest way is use tfa library
    return warped, inside_frame

def warp_bilinear_x_coord(features, warp_coords):
    """
    features : float
        [batch, height, width, num_features]
    warp_coords : float
        [batch, height, width_warp, 2(x,y)]
    
    bilnear sampling
        
    メモ：レンジ外のマスクもアウトプットする。
    return:        
        warped_feature [batch, height_warp, width_warp, num_features]
        inside_frame_mask [batch, height_warp, width_warp, 1]
        
    USE ONLY X idx (no change in y axis)    
    
    """
    batch, height, width, _ = tf.unstack(tf.shape(features))
    x_idx = warp_coords[...,1]
    #y_idx = warp_coords[...,1:2]
    inside_frame = tf.math.logical_and(x_idx >= 0., x_idx <= tf.cast(width-1, tf.float32))
    #inside_frame_y = tf.math.logical_and(y_idx >= 0., y_idx <= tf.cast(height-1, tf.float32))
    #inside_frame = tf.math.logical_and(inside_frame_x, inside_frame_y)

    x_idx = tf.clip_by_value(x_idx, 0., tf.cast(width-1, tf.float32))
    #y_idx = tf.clip_by_value(y_idx, 0., tf.cast(height-1, tf.float32))

    left_idx = tf.floor(x_idx)
    right_idx = tf.math.ceil(x_idx)
    #top_idx = tf.floor(y_idx)
    #bottom_idx = tf.math.ceil(y_idx)

    #left_idx = tf.clip_by_value(left_idx, 0., tf.cast(width-1, tf.float32))
    #right_idx = tf.clip_by_value(right_idx, 0., tf.cast(width-1, tf.float32))
    #top_idx = tf.clip_by_value(top_idx, 0., tf.cast(height-1, tf.float32))
    #bottom_idx = tf.clip_by_value(bottom_idx, 0., tf.cast(height-1, tf.float32))
    
    l_weight = right_idx - x_idx
    r_weight = 1.0 - l_weight
    #top_weight = bottom_idx - y_idx
    #bottom_weight = 1.0 - top_weight
    
    #tl_weight = top_weight * left_weight
    #tr_weight = top_weight * right_weight
    #bl_weight = bottom_weight * left_weight
    #br_weight = bottom_weight * right_weight
    
    #top_idx = tf.cast(top_idx, tf.int32)
    left_idx = tf.cast(left_idx, tf.int32)
    #bottom_idx = tf.cast(bottom_idx, tf.int32)
    right_idx = tf.cast(right_idx, tf.int32)

    #top_left = tf.concat([top_idx, left_idx], axis=-1)
    #top_right = tf.concat([top_idx, right_idx], axis=-1)
    #bottom_left = tf.concat([bottom_idx, left_idx], axis=-1)
    #bottom_right = tf.concat([bottom_idx, right_idx], axis=-1)
    #print(top_left)
    l_feature = tf.gather(features, left_idx, batch_dims=2)
    r_feature = tf.gather(features, right_idx, batch_dims=2)
    #bl_feature = tf.gather_nd(features, bottom_left, batch_dims=1)
    #br_feature = tf.gather_nd(features, bottom_right, batch_dims=1)

    warped = l_weight * l_feature + r_weight * r_feature
    return warped, inside_frame



def warp_bilinear_flat(data, warp, name='flat_resampler'):
  """Resampler that avoids gather_nd which can be expensive on TPU.
  Computing gradients of gather_nd requires calling scatter_nd
  which is very slow on TPU and causes a large memory blowup.
  Empirically, this resampler produces a much lower memory footprint
  and faster inference time on the TPU by avoding gather_nd and instead
  using a flat gather. See tfa.image.resampler for more documentation.
  Args:
    data: float tf Tensor of shape b H W c, The source to differentiably
      resample from.
    warp: float tf Tensor of shape b h w 2, The set of coordinates to sample
      from data.
    name: str scope to put operations under.
  Returns:
    resampled_data: float tf Tensor of shape b h w c, The result of sampling
      data with warp.
  """
  with tf.name_scope(name):
    b, data_h, data_w, c = tf.unstack(tf.shape(data))
    _, warp_h, warp_w, _ = tf.unstack(tf.shape(warp))
    warp_x, warp_y = tf.unstack(warp, axis=-1)

    warp_shape = tf.shape(warp_x)
    warp_batch = tf.range(warp_shape[0], dtype=tf.int32)
    warp_batch = tf.reshape(warp_batch, (warp_shape[0], 1, 1))
    warp_batch = tf.broadcast_to(warp_batch, (b, warp_h, warp_w))
    warp_batch = tf.reshape(warp_batch, [-1])
    warp_x = tf.reshape(warp_x, [-1])
    warp_y = tf.reshape(warp_y, [-1])
    warp_floor_x = tf.math.floor(warp_x)
    warp_floor_y = tf.math.floor(warp_y)

    right_warp_weight = warp_x - warp_floor_x
    down_warp_weight = warp_y - warp_floor_y
    left_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, right_warp_weight.dtype), right_warp_weight)
    up_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, down_warp_weight.dtype), down_warp_weight)

    warp_floor_x = tf.cast(warp_floor_x, tf.int32)
    warp_floor_y = tf.cast(warp_floor_y, tf.int32)
    warp_ceil_x = tf.cast(tf.math.ceil(warp_x), tf.int32)
    warp_ceil_y = tf.cast(tf.math.ceil(warp_y), tf.int32)

    left_warp_weight = tf.expand_dims(left_warp_weight, -1)
    right_warp_weight = tf.expand_dims(right_warp_weight, -1)
    up_warp_weight = tf.expand_dims(up_warp_weight, -1)
    down_warp_weight = tf.expand_dims(down_warp_weight, -1)

    def flatten_warp(warp_y, warp_x):
      """Converts the warps from a 2D index to a 1D index."""
      output = tf.reshape(
          warp_batch * data_w * data_h + warp_y * data_w + warp_x, [-1])
      # Get a mask of the coordinates which go out of bounds.
      mask_y = tf.cast(
          tf.logical_and(warp_y >= 0, warp_y <= data_h - 1), dtype=data.dtype)
      mask_x = tf.cast(
          tf.logical_and(warp_x >= 0, warp_x <= data_w - 1), dtype=data.dtype)
      output = tf.clip_by_value(output, 0, b * data_h * data_w - 1)
      return output, tf.expand_dims(mask_y * mask_x, -1)

    up_left_warp, mask_up_left = flatten_warp(warp_floor_y, warp_floor_x)
    up_right_warp, mask_up_right = flatten_warp(warp_floor_y, warp_ceil_x)
    down_left_warp, mask_down_left = flatten_warp(warp_ceil_y, warp_floor_x)
    down_right_warp, mask_down_right = flatten_warp(warp_ceil_y, warp_ceil_x)
    flat_data = tf.reshape(data, (-1, c))

    up_left = tf.gather(flat_data, up_left_warp, axis=0) * mask_up_left
    up_right = tf.gather(flat_data, up_right_warp, axis=0) * mask_up_right
    down_left = tf.gather(flat_data, down_left_warp, axis=0) * mask_down_left
    down_right = tf.gather(flat_data, down_right_warp, axis=0) * mask_down_right
    result = (up_left * left_warp_weight + up_right * right_warp_weight
             ) * up_warp_weight + (down_left * left_warp_weight + down_right *
                                   right_warp_weight) * down_warp_weight
    return tf.reshape(result, (b, warp_h, warp_w, c))


#4,6500,6500

class _CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, num_feature=256):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.num_feature = num_feature
        self.num_levels = num_levels
        self.radius = radius

        corr = self.correlation(fmap1, fmap2)
        batch_size, h1, w1, _, h2, w2 = tf.unstack(tf.shape(corr))
        corr = tf.reshape(corr, (batch_size*h1*w1, h2, w2, 1))

        # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
        self.corr_pyramid = [corr] 
        for _ in range(num_levels - 1):
            corr = tf.nn.avg_pool2d(corr, 2, 2, padding='VALID')
            self.corr_pyramid.append(corr)

    def retrieve(self, coords):
        ''' Retrieve correlation values specified by coordinates
        Args:
          coords: coordinates tensor, shape (batch_size, h, w, 2)
        
        Returns:
          A tensor contains multiscale correlation
            with shape (bs, h, w, 81*num_levels)
        '''
        r = self.radius
        bs, h, w, _ = tf.unstack(tf.shape(coords))

        out_pyramid = []
        for i in range(self.num_levels):
            # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
            corr = self.corr_pyramid[i]
            # (2r+1, 2r+1)x2
            d = tf.range(-r, r+1, dtype=tf.float32)
            dy, dx = tf.meshgrid(d, d, indexing='ij')
            # (2r+1, 2r+1, 2)
            delta = tf.stack([dy, dx], axis=-1)
            # -> (1, 2r+1, 2r+1, 2)
            delta_lvl = tf.reshape(delta, (1, 2*r+1, 2*r+1, 2))

            # reshape and scale -> (bs*h*w, 1, 1, 2)
            centroid_lvl = tf.reshape(coords, (bs*h*w, 1, 1, 2)) / 2**i
            # add -> (bs*h*w, 2r+1, 2r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            # output: (bs*h*w, 2r+1, 2r+1, dim)
            #corr, _ = warp_bilinear(corr, coords_lvl)
            corr, _ = warp_bilinear_tfa(corr, coords_lvl)

            # -> (bs, h, w, (2r+1)*(2r+1)*nch)
            num_ch = (2*r+1) * (2*r+1)
            corr = tf.reshape(corr, (bs, h, w, num_ch))
            out_pyramid.append(corr)

        out = tf.concat(out_pyramid, axis=-1)
        return out

    def correlation(self, fmap1, fmap2):#52, 125
        b, h, w, ch = tf.unstack(tf.shape(fmap1))
        fmap1 = tf.reshape(fmap1, (b, h*w, ch))
        fmap2 = tf.reshape(fmap2, (b, h*w, ch))

        # shape (batch_size, h*w, h*w)
        corr = tf.matmul(fmap1, fmap2, transpose_b=True)
        corr = tf.reshape(corr, (b, h, w, 1, h, w))
        return corr / tf.sqrt(tf.cast(ch, dtype=tf.float32))  

class _CorrBlockStereo:
    """
    for stereo, reduce 1 rank -> B,W1,H2(=H1),W2
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, num_feature=256):
        
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.num_feature = num_feature
        self.num_levels = num_levels
        self.radius = radius

        corr = self.correlation(fmap1, fmap2)
        b, w1, h, w2 = tf.unstack(tf.shape(corr))
        corr = tf.reshape(corr, (b*w1, h, w2, 1))

        # (bs*w, h, w, 1), (bs*w, h/2, w/2, 1), ..., (bs*w, h/8, w/8, 1)
        self.corr_pyramid = [corr] 
        for _ in range(num_levels - 1):
            corr = tf.nn.avg_pool2d(corr, 2, 2, padding='VALID')
            self.corr_pyramid.append(corr)

    def retrieve(self, coords):
        ''' Retrieve correlation values specified by coordinates
        Args:
          coords: coordinates tensor, shape (batch_size, h, w, 2)
        
        Returns:
          A tensor contains multiscale correlation
            with shape (bs, h, w, 81*num_levels)
        '''
        r = self.radius
        b, h, w, _ = tf.unstack(tf.shape(coords))
        coords_tr = tf.transpose(coords, [0,2,1,3])#b,w,h,c

        out_pyramid = []
        for i in range(self.num_levels):
            # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
            corr = self.corr_pyramid[i]
            # (2r+1, 2r+1)x2
            d = tf.range(-r, r+1, dtype=tf.float32)
            dy, dx = tf.meshgrid(d, d, indexing='ij')
            # (2r+1, 2r+1, 2)
            delta = tf.stack([dy, dx], axis=-1)
            # -> (1, 2r+1, 2r+1, 2)
            delta_lvl = tf.reshape(delta, (1, 1, 2*r+1, 2*r+1, 2))

            # reshape and scale -> (b*w, h, 1, 1, 2)
            centroid_lvl = tf.reshape(coords_tr, (b*w, h, 1, 1, 2)) / 2**i
            # add -> (b*w, h, 2r+1, 2r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            # output: (bs*h*w, h, 2r+1, 2r+1, dim)
            num_ch = (2*r+1) * (2*r+1)
            coords_lvl = tf.reshape(coords_lvl, (b*w, h, num_ch, 2))
            
            # tf.gather_nd is slow? memory consuming? 
            # use simple tf.gather or tfa.image.resampler instead. 
            
            #corr,_ = warp_bilinear(corr, coords_lvl)#slow?
            #corr = warp_bilinear_flat(corr, coords_lvl)#smurf is using this function?
            #corr = tfa.image.resampler(corr, coords_lvl)#easiest way is use tfa library
            corr, _ = warp_bilinear_tfa(corr, coords_lvl)
            # -> (bs, h, w, (2r+1)*(2r+1)*nch)
            corr = tf.reshape(corr, (b, w, h, num_ch))
            corr = tf.transpose(corr, [0,2,1,3])
            """
            centroid_lvl = tf.reshape(coords_tr, (b*w*h, 2))/tf.cast(h, tf.float32)# / 2**i
            tlbrs = tf.concat([centroid_lvl-0.1, centroid_lvl+0.1], axis=-1)
            num_ch = (2*r+1) * (2*r+1)
            #print(tlbrs.shape, tf.range(b*w*h).shape)
            corr = tf.image.crop_and_resize(corr, tlbrs, 
                                     box_indices=tf.range(b*w*h)//h, 
                                     crop_size=(9,9))
            corr = tf.reshape(corr, (b, w, h, num_ch))
            corr = tf.transpose(corr, [0,2,1,3])
            """
            
            out_pyramid.append(corr)


        out = tf.concat(out_pyramid, axis=-1)
        return out

    def correlation(self, fmap1, fmap2):#52, 125
        b, h, w, ch = tf.unstack(tf.shape(fmap1))
        
        corr = tf.einsum('bhwi, bhxi->bhwx', fmap1, fmap2)
        corr = tf.transpose(corr, [0,2,1,3])
        
              

        # shape (batch_size, h*w, h*w)
        #corr = tf.matmul(fmap1, fmap2, transpose_b=True)
        #corr = tf.reshape(corr, (b, h, w, 1, h, w))
        return corr / tf.sqrt(tf.cast(ch, dtype=tf.float32))  

class CorrBlockStereo:
    """
    for stereo, reduce 1 rank -> B,W1,H2(=H1),W2
    """
    def __init__(self, num_levels=4, radius=4, num_feature=256):
        
        self.num_feature = num_feature
        self.num_levels = num_levels
        self.radius = radius

        
    def build_pyramid(self, fmap1, fmap2):
        corr = self.correlation(fmap1, fmap2)
        b, w1, h, w2 = tf.unstack(tf.shape(corr))
        corr = tf.reshape(corr, (b*w1, h, w2, 1))

        # (bs*w, h, w, 1), (bs*w, h/2, w/2, 1), ..., (bs*w, h/8, w/8, 1)
        self.corr_pyramid = [corr] 
        for _ in range(self.num_levels - 1):
            corr = tf.nn.avg_pool2d(corr, 2, 2, padding='VALID')
            self.corr_pyramid.append(corr)

    def retrieve(self, coords):
        ''' Retrieve correlation values specified by coordinates
        Args:
          coords: coordinates tensor, shape (batch_size, h, w, 2)
        
        Returns:
          A tensor contains multiscale correlation
            with shape (bs, h, w, 81*num_levels)
        '''
        r = self.radius
        b, h, w, _ = tf.unstack(tf.shape(coords))
        coords_tr = tf.transpose(coords, [0,2,1,3])#b,w,h,c

        out_pyramid = []
        for i in range(self.num_levels):
            # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
            corr = self.corr_pyramid[i]
            # (2r+1, 2r+1)x2
            d = tf.range(-r, r+1, dtype=tf.float32)
            dy, dx = tf.meshgrid(d, d, indexing='ij')
            # (2r+1, 2r+1, 2)
            delta = tf.stack([dy, dx], axis=-1)
            # -> (1, 2r+1, 2r+1, 2)
            delta_lvl = tf.reshape(delta, (1, 1, 2*r+1, 2*r+1, 2))

            # reshape and scale -> (b*w, h, 1, 1, 2)
            centroid_lvl = tf.reshape(coords_tr, (b*w, h, 1, 1, 2)) / 2**i
            # add -> (b*w, h, 2r+1, 2r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            # output: (bs*h*w, h, 2r+1, 2r+1, dim)
            num_ch = (2*r+1) * (2*r+1)
            coords_lvl = tf.reshape(coords_lvl, (b*w, h, num_ch, 2))
            
            # tf.gather_nd is slow? memory consuming? 
            # use simple tf.gather or tfa.image.resampler instead. 
            
            #corr,_ = warp_bilinear(corr, coords_lvl)#slow?
            #corr = warp_bilinear_flat(corr, coords_lvl)#smurf is using this function?
            #corr = tfa.image.resampler(corr, coords_lvl)#easiest way is use tfa library
            corr, _ = warp_bilinear_tfa(corr, coords_lvl)
            # -> (bs, h, w, (2r+1)*(2r+1)*nch)
            corr = tf.reshape(corr, (b, w, h, num_ch))
            corr = tf.transpose(corr, [0,2,1,3])
            """
            centroid_lvl = tf.reshape(coords_tr, (b*w*h, 2))/tf.cast(h, tf.float32)# / 2**i
            tlbrs = tf.concat([centroid_lvl-0.1, centroid_lvl+0.1], axis=-1)
            num_ch = (2*r+1) * (2*r+1)
            #print(tlbrs.shape, tf.range(b*w*h).shape)
            corr = tf.image.crop_and_resize(corr, tlbrs, 
                                     box_indices=tf.range(b*w*h)//h, 
                                     crop_size=(9,9))
            corr = tf.reshape(corr, (b, w, h, num_ch))
            corr = tf.transpose(corr, [0,2,1,3])
            """
            
            out_pyramid.append(corr)


        out = tf.concat(out_pyramid, axis=-1)
        return out

    def correlation(self, fmap1, fmap2):#52, 125
        b, h, w, ch = tf.unstack(tf.shape(fmap1))
        
        corr = tf.einsum('bhwi, bhxi->bhwx', fmap1, fmap2)
        corr = tf.transpose(corr, [0,2,1,3])
        
              

        # shape (batch_size, h*w, h*w)
        #corr = tf.matmul(fmap1, fmap2, transpose_b=True)
        #corr = tf.reshape(corr, (b, h, w, 1, h, w))
        return corr / tf.sqrt(tf.cast(ch, dtype=tf.float32))  


class CorrBlock:
    
    def __init__(self, num_levels=4, radius=4, num_feature=256):
        
        self.num_feature = num_feature
        self.num_levels = num_levels
        self.radius = radius
        """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, num_feature=256):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.num_feature = num_feature
        self.num_levels = num_levels
        self.radius = radius

        corr = self.correlation(fmap1, fmap2)
        batch_size, h1, w1, _, h2, w2 = tf.unstack(tf.shape(corr))
        corr = tf.reshape(corr, (batch_size*h1*w1, h2, w2, 1))

        # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
        self.corr_pyramid = [corr] 
        for _ in range(num_levels - 1):
            corr = tf.nn.avg_pool2d(corr, 2, 2, padding='VALID')
            self.corr_pyramid.append(corr)
        """
        
    def build_pyramid(self, fmap1, fmap2):
        corr = self.correlation(fmap1, fmap2)
        b, h1, w1, _, h2, w2 = tf.unstack(tf.shape(corr))
        corr = tf.reshape(corr, (b*h1*w1, h2, w2, 1))

        # (bs*w, h, w, 1), (bs*w, h/2, w/2, 1), ..., (bs*w, h/8, w/8, 1)
        self.corr_pyramid = [corr] 
        for _ in range(self.num_levels - 1):
            corr = tf.nn.avg_pool2d(corr, 2, 2, padding='VALID')
            self.corr_pyramid.append(corr)
            
    def retrieve(self, coords):
        ''' Retrieve correlation values specified by coordinates
        Args:
          coords: coordinates tensor, shape (batch_size, h, w, 2)
        
        Returns:
          A tensor contains multiscale correlation
            with shape (bs, h, w, 81*num_levels)
        '''
        r = self.radius
        bs, h, w, _ = tf.unstack(tf.shape(coords))

        out_pyramid = []
        for i in range(self.num_levels):
            # (bs*h*w, h, w, 1), (bs*h*w, h/2, w/2, 1), ..., (bs*h*w, h/8, w/8, 1)
            corr = self.corr_pyramid[i]
            # (2r+1, 2r+1)x2
            d = tf.range(-r, r+1, dtype=tf.float32)
            dy, dx = tf.meshgrid(d, d, indexing='ij')
            # (2r+1, 2r+1, 2)
            delta = tf.stack([dy, dx], axis=-1)
            # -> (1, 2r+1, 2r+1, 2)
            delta_lvl = tf.reshape(delta, (1, 2*r+1, 2*r+1, 2))

            # reshape and scale -> (bs*h*w, 1, 1, 2)
            centroid_lvl = tf.reshape(coords, (bs*h*w, 1, 1, 2)) / 2**i
            # add -> (bs*h*w, 2r+1, 2r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            # output: (bs*h*w, 2r+1, 2r+1, dim)
            #corr, _ = warp_bilinear(corr, coords_lvl)
            corr, _ = warp_bilinear_tfa(corr, coords_lvl)

            # -> (bs, h, w, (2r+1)*(2r+1)*nch)
            num_ch = (2*r+1) * (2*r+1)
            corr = tf.reshape(corr, (bs, h, w, num_ch))
            out_pyramid.append(corr)

        out = tf.concat(out_pyramid, axis=-1)
        return out
 
    def correlation(self, fmap1, fmap2):#52, 125
        b, h, w, ch = tf.unstack(tf.shape(fmap1))
        fmap1 = tf.reshape(fmap1, (b, h*w, ch))
        fmap2 = tf.reshape(fmap2, (b, h*w, ch))

        # shape (batch_size, h*w, h*w)
        corr = tf.matmul(fmap1, fmap2, transpose_b=True)
        corr = tf.reshape(corr, (b, h, w, 1, h, w))
        # TODO use einsum and transpose instead?
        
        return corr / tf.sqrt(tf.cast(ch, dtype=tf.float32))  
    
    
    

class UpdateBlockStereo(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=96, **kwargs):
        super().__init__(**kwargs)
        self.flow_convs = tf.keras.Sequential([Conv2D(64, 3, 1, activation="relu", padding='same'),
                                               Conv2D(32, 3, 1, activation="relu", padding='same')])
        self.cor_conv = Conv2D(96, 1, 1, activation="relu", padding='same')
        self.corflow_conv = Conv2D(80, 1, 1, activation="relu", padding='same')
        self.gru = ConvGRU(filters=hidden_dim)
        self.head =  tf.keras.Sequential([Conv2D(128, 3, 1, activation="relu", padding='same'),
                                          Conv2D(1, 3, 1, activation="linear", padding='same')])
        #1ch == ONLY X DIRECTION (stereo depth)

        self.mask = tf.keras.Sequential([Conv2D(256, 3, 1, activation="relu", padding='same'),
                                         Conv2D(64*9, 1, 1),
                                         ])

    def call(self, inputs):
        hidden, context, correlation, flow = inputs
        corflow_feature = tf.concat([self.cor_conv(correlation), self.flow_convs(flow)], axis=-1)
        corflow_feature = tf.concat([self.corflow_conv(corflow_feature), flow], axis=-1)
        
        #motion_features = self.encoder([flow, corr])
        gru_inputs = tf.concat([context, corflow_feature], axis=-1)

        next_hidden = self.gru([hidden, gru_inputs])
        delta_flow = self.head(next_hidden)
        delta_flow = tf.concat([delta_flow, tf.zeros_like(delta_flow)], axis=-1)#only X

        # scale mask to balance gradients
        mask = 0.25*self.mask(next_hidden)
        return next_hidden, mask, delta_flow

class UpdateBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=96, **kwargs):
        super().__init__(**kwargs)
        self.flow_convs = tf.keras.Sequential([Conv2D(64, 3, 1, activation="relu", padding='same'),
                                               Conv2D(32, 3, 1, activation="relu", padding='same')])
        self.cor_conv = Conv2D(96, 1, 1, activation="relu", padding='same')
        self.corflow_conv = Conv2D(80, 1, 1, activation="relu", padding='same')
        self.gru = ConvGRU(filters=hidden_dim)
        self.head =  tf.keras.Sequential([Conv2D(128, 3, 1, activation="relu", padding='same'),
                                          Conv2D(2, 3, 1, activation="linear", padding='same')])
        self.mask = tf.keras.Sequential([Conv2D(256, 3, 1, activation="relu", padding='same'),
                                         Conv2D(64*9, 1, 1),
                                         ])

    def call(self, inputs):
        hidden, context, correlation, flow = inputs
        corflow_feature = tf.concat([self.cor_conv(correlation), self.flow_convs(flow)], axis=-1)
        corflow_feature = tf.concat([self.corflow_conv(corflow_feature), flow], axis=-1)
        
        #motion_features = self.encoder([flow, corr])
        gru_inputs = tf.concat([context, corflow_feature], axis=-1)

        next_hidden = self.gru([hidden, gru_inputs])
        delta_flow = self.head(next_hidden)

        # scale mask to balance gradients
        mask = 0.25*self.mask(next_hidden)
        return next_hidden, mask, delta_flow
    
    def quick_inference(self, inputs):
        hidden, context, correlation, flow = inputs
        corflow_feature = tf.concat([self.cor_conv(correlation), self.flow_convs(flow)], axis=-1)
        corflow_feature = tf.concat([self.corflow_conv(corflow_feature), flow], axis=-1)
        
        #motion_features = self.encoder([flow, corr])
        gru_inputs = tf.concat([context, corflow_feature], axis=-1)

        next_hidden = self.gru([hidden, gru_inputs])
        delta_flow = self.head(next_hidden)

        return next_hidden, delta_flow


def loss_wrapper(loss_weights, gammma=0.8, batch_for_occ_mask=20000):                                     
    def loss_stereo_depth(rgb_1, rgb_2, 
                          #box_mask, box_mask_l, 
                          #dist_target,
                          endpoints_all_stage, 
                          batch_counter=0):
        """
        occlusion aware loss
    
        """
        use_occlusion = tf.cast(batch_counter > batch_for_occ_mask, tf.float32)
        rgbs = [rgb_1, rgb_2]
        #box_masks = [box_mask, box_mask_l]
        iterations = tf.cast(len(endpoints_all_stage), tf.float32)
        #loss_weights = {"rgb": 2.0,
        #                "ssim": 3.0,
        #                "smooth": 2.0}
        loss = {"rgb": 0.,
                "census": 0.,
                "ssim": 0.,
                "smooth": 0.,
                #"depth_l1": 0.,
                }
        loss_iter = [[],[]]
        for i in range(2):
            rgb = rgbs[i]
            #box_mask = box_masks[i]
            no_frame = 1 + i
            for j, ep in enumerate(endpoints_all_stage):
                decay_factor = gammma ** (iterations-1.-tf.cast(j, tf.float32))
                rgb_re = ep["rgb_{}_re".format(no_frame)]
                #depth = ep["depth_{}".format(no_frame)]
                #depth_re = ep["depth_{}_re".format(no_frame)]
                #mask_re = ep["mask_{}_re".format(no_frame)]
                mask = ep["occulusion_mask_for_{}".format(no_frame)] > 0.
                flow = ep["flow_{}".format(no_frame)]
                #dev_disparity = ep["devdisparity_{}".format(no_frame)]
                loss_rgb, loss_census, loss_ssim, loss_smooth = _loss(rgb, rgb_re, 
                                                                      #depth, depth_re, mask_re, 
                                                                      mask,
                                                                      flow, 
                                                                      #dev_disparity, 
                                                                      use_census=False, 
                                                                      use_occlusion=use_occlusion)#loss_weights["census"]!=0)
                #if i==0:
                #loss_smooth += box_smoothness_loss(flow, box_mask)
                #loss_depth_l1 = box_depth_l1(depth, dist_target, box_mask)
                loss["rgb"] = loss["rgb"] + loss_rgb * decay_factor
                loss["census"] = loss["census"] + loss_census * decay_factor
                loss["ssim"] = loss["ssim"] + loss_ssim * decay_factor
                loss["smooth"] = loss["smooth"] + loss_smooth * decay_factor
                #loss["depth_l1"] = loss["depth_l1"] + loss_depth_l1 * decay_factor
                ##iter_total = loss_weights["rgb"] * loss_rgb + loss_weights["census"]*loss_census + loss_weights["ssim"] * loss_ssim + loss_weights["smooth"] * loss_smooth + loss_weights["depth_l1"] * loss_depth_l1
                iter_total = loss_weights["rgb"] * loss_rgb + loss_weights["census"]*loss_census + loss_weights["ssim"] * loss_ssim + loss_weights["smooth"] * loss_smooth
                loss_iter[i].append(iter_total)
        loss["total"] = loss_weights["rgb"] * loss["rgb"] + loss_weights["census"] * loss["census"] + loss_weights["ssim"] * loss["ssim"] + loss_weights["smooth"] * loss["smooth"]
        
        return loss, loss_iter
    return loss_stereo_depth

def compute_occlusions_valid(depth, depth_re, simple=False):
    if simple:
        margin = 1e-7#マージンポジティブにすると狭間がロスゼロになる。ペナルティいるかも？？
        valid = ((depth+margin) >= depth_re)
    else:
        sq_mean = tf.math.sqrt((depth**2 + depth_re**2)/2.)
        #sq_dev = (depth-depth_re)**2
        #valid = (sq_dev > - sq_sum*0.01)
        #valid = ((depth-depth_re) <= sq_mean*0.01)
        valid = ((depth-depth_re) >= -sq_mean*0.01)
    return valid
    

def occlusion_penalty(nonoccluded):
    occluded_rate = tf.reduce_mean(1.0 - tf.cast(nonoccluded, tf.float32))
    penalty = tf.where(occluded_rate>0.5, (occluded_rate-0.5)**2, 0.0)
    return penalty

def box_smoothness_loss(flow, box_mask):
    #dy_flow, dx_flow = tf.image.image_gradients(flow)
    dx_flow = flow[:, :-1, 1:] - flow[:, :-1, :-1]
    dy_flow = flow[:, 1:,:-1] - flow[:, :-1, :-1]
    box_mask = box_mask[:,:-1,:-1]
    loss = tf.reduce_sum(box_mask*tf.reduce_sum((dy_flow**2 + dx_flow**2), axis=-1))/tf.reduce_sum(box_mask)
    return loss


def box_depth_l1(depth, gt_dist, box_mask):
    """
    pred_depth:
        float (batch, height, width, 1)
    car_mask:
        float. mask for depth (batch, height, width)
    gt_dist:
        distance between cars (batch)
    """
    depth = depth[:,:,:,0]
    depth = tf.clip_by_value(depth, gt_dist[:,tf.newaxis,tf.newaxis]*0.5, gt_dist[:,tf.newaxis,tf.newaxis]*1.5)
    #pred_dist = tf.reduce_sum(depth*box_mask, axis=[1,2]) / (tf.reduce_sum(box_mask, axis=[1,2]) + 1e-7)
    #relative_error = tf.reduce_mean(tf.math.abs(gt_dist - pred_dist)/gt_dist)
    
    relative_error = tf.math.abs(depth - gt_dist[:,tf.newaxis,tf.newaxis]) / gt_dist[:,tf.newaxis,tf.newaxis]
    relative_error = tf.reduce_sum(relative_error*box_mask, axis=[1,2])/(tf.reduce_sum(box_mask, axis=[1,2]) + 1e-7)
    
    #relative_error = absolute_error/gt_dist
    return tf.reduce_mean(relative_error)
        
def _loss(rgb, rgb_re, 
          #depth, depth_re, 
          mask, 
          flow, 
          #dev_disparity, 
          use_census=True, use_occlusion=0.):
    #valid_reconst = tf.cast(tf.ones_like(rgb[...,:1]), tf.float32)
    no_mask = tf.cast(tf.ones_like(rgb[...,:1]), tf.float32)
    mask = tf.cast(mask, tf.float32)
    
    mask = no_mask * (1.-use_occlusion) + mask * use_occlusion
    #mask_re = tf.cast(tf.ones_like(rgb[...,:1]), tf.float32)
    """
    valid_reconst = compute_occlusions_valid(depth, depth_re)
    ##occ_penalty = occlusion_penalty(valid_reconst)
    valid_reconst = tf.cast(tf.math.logical_and(valid_reconst, mask_re), tf.float32)
    valid_reconst = tf.stop_gradient(valid_reconst)    
    mask_re = tf.cast(mask_re, tf.float32)
    
    valid_reconst = valid_reconst * use_occlusion + tf.stop_gradient(mask_re) * (1.-use_occlusion)
    
    
    depth = tf.stop_gradient(depth)
    depth_re = tf.stop_gradient(depth_re)
    """
    loss_rgb = loss_rgb_L1(rgb, rgb_re, mask)
    if use_census:
        loss_census = loss_census_softhamming(rgb, rgb_re, mask)
    else:#computing census_loss is slow. so skip when you don't use it.
        loss_census = 0.#loss_census_softhamming(rgb, rgb_re, valid_reconst)
    #loss_census = loss_census + neg_disparity_loss(dev_disparity)
    #loss_ssim = loss_rgb_ssim(rgb, rgb_re, depth, depth_re, mask_re)
    loss_ssim = loss_rgb_ssim_simple(rgb, rgb_re, mask)
    loss_smooth = loss_edge_aware_flow_smoothness(flow, rgb)
    return loss_rgb, loss_census, loss_ssim, loss_smooth

def neg_disparity_loss(dev_disparity):
    return tf.reduce_mean((0.05 * dev_disparity)**2)
    
def loss_consistency():
    pass
    
def __loss_edge_aware_flow_smoothness(flow, rgb, lam=100., second_order=True):
    """
    TODO 2nd order -> stride for rgb image gradient should be 2
    """
    
    # assming the flow has only 1 dim (X disparity)
    dy_flow, dx_flow = tf.image.image_gradients(flow)
    if second_order:
        _, dx_flow = tf.image.image_gradients(dx_flow)
        dy_flow, _ = tf.image.image_gradients(dy_flow)
        dy_flow = tf.math.abs(dy_flow)[:,:-2,:]
        dx_flow = tf.math.abs(dx_flow)[:,:,:-2]
    else:
        dy_flow = tf.math.abs(dy_flow)[:,:-1,:]
        dx_flow = tf.math.abs(dx_flow)[:,:,:-1]
    
    dy_rgb, dx_rgb = tf.image.image_gradients(rgb)
    dy_rgb = tf.reduce_sum(tf.math.abs(dy_rgb), axis=-1, keepdims=True) / 3.
    dx_rgb = tf.reduce_sum(tf.math.abs(dx_rgb), axis=-1, keepdims=True) / 3.
    if second_order:
        dy_rgb = dy_rgb[:,:-2,:]
        dx_rgb = tf.math.abs(dx_rgb)[:,:,:-2]
    else:
        dy_rgb = dy_rgb[:,:-1,:]
        dx_rgb = tf.math.abs(dx_rgb)[:,:,:-1]
    
    loss_x = tf.reduce_mean(tf.math.exp(-lam*dx_rgb) * dx_flow)
    loss_y = tf.reduce_mean(tf.math.exp(-lam*dy_rgb) * dy_flow)
    
    return loss_x + loss_y
        

def loss_edge_aware_flow_smoothness(flow, rgb, lam=150., second_order=True, smooth_level=1):
    """
    smooth_level: Resolution level at which the smoothness loss should be applied.
    """
    for _ in range(smooth_level):#loss at low resolution
        b, h, w, _ = tf.unstack(tf.shape(rgb))
        rgb = tf.image.resize(rgb, size=[h//2, w//2])
        flow = tf.image.resize(flow, size=[h//2, w//2]) ##* 2.#flow should be changed? based on the scale
    
    def get_grad(imgs, stride):
        dx = imgs[:, :, stride:] - imgs[:, :, :-stride]
        dy = imgs[:, stride:] - imgs[:, :-stride]
        return dy, dx
        
    # assming the flow has only 1 dim (X disparity)
    dy_flow, dx_flow = get_grad(flow, 1)
    if second_order:
        _, dx_flow = get_grad(dx_flow, 1)
        dy_flow, _ = get_grad(dy_flow, 1)
        
    dy_flow = tf.math.abs(dy_flow)
    dx_flow = tf.math.abs(dx_flow)
    
    if second_order:
        dy_rgb, dx_rgb = get_grad(rgb, 2)
    else:
        dy_rgb, dx_rgb = get_grad(rgb, 1)
    dy_rgb = tf.reduce_sum(tf.math.abs(dy_rgb), axis=-1, keepdims=True) / 3.
    dx_rgb = tf.reduce_sum(tf.math.abs(dx_rgb), axis=-1, keepdims=True) / 3.
    
    loss_x = tf.reduce_mean(tf.math.exp(-lam*dx_rgb) * dx_flow)
    loss_y = tf.reduce_mean(tf.math.exp(-lam*dy_rgb) * dy_flow)
    
    return loss_x + loss_y


def loss_census_softhamming(rgb, rgb_re, mask, patch_size=7):
    """
    Compare the similarity of the census transform of two images.
    this implementation is from SMURF
    """
    
    def census_transform(image, patch_size):
        """
        take relative strength among neighbors.
        each pixel has positive or negative values [1,-1,1,1,-1,1,1,1,-1,-1,...]
        """
        intensities = tf.image.rgb_to_grayscale(image) * 255
        kernel = tf.reshape(
            tf.eye(patch_size * patch_size),
            (patch_size, patch_size, 1, patch_size * patch_size))#h,w,in,out_ch
        neighbors = tf.nn.conv2d(
            input=intensities, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
        diff = neighbors - intensities
        # Magic numbers taken from DDFlow
        diff_norm = diff / tf.sqrt(.81 + tf.square(diff))
        return diff_norm
    
    def soft_hamming(a_bhwk, b_bhwk, thresh=.1):
        """A soft hamming distance between tensor a_bhwk and tensor b_bhwk.
        Returns a tensor with approx. 1 in (h, w) locations that are significantly
        more different than thresh and approx. 0 if significantly less
        different than thresh.
        """
        sq_dist_bhwk = tf.square(a_bhwk - b_bhwk)
        soft_thresh_dist_bhwk = sq_dist_bhwk / (thresh + sq_dist_bhwk)
        return tf.reduce_sum(
            input_tensor=soft_thresh_dist_bhwk, axis=3, keepdims=True)#Reduce sumが正解？？
    
    def zero_mask_border(mask_bhw3, patch_size):
        """Used to ignore border effects from census_transform."""
        mask_padding = patch_size // 2
        mask = mask_bhw3[:, mask_padding:-mask_padding, mask_padding:-mask_padding, :]
        return tf.pad(
            tensor=mask,
            paddings=[[0, 0], [mask_padding, mask_padding],
                      [mask_padding, mask_padding], [0, 0]])
    
    census_rgb = census_transform(rgb, patch_size)
    census_rgb_re = census_transform(rgb_re, patch_size)
  
    hamming_dist = soft_hamming(census_rgb,
                                census_rgb_re)
  
    # set borders of mask to zero to ignore edge effects
    padded_mask = zero_mask_border(mask, patch_size)
    diff = tf.math.pow((tf.math.abs(hamming_dist) + 1e-2), 0.4)
    # originalの実装は後者のみstop gradientはいってた。この時点で既にマスクはストップしてるけど、分子は残すべきなのか？？
    loss_mean = tf.reduce_sum(diff * padded_mask) / (tf.reduce_sum(padded_mask) + 1e-7)
    return loss_mean/(patch_size**2)#originalは割ってない。


def loss_soft_hamming():
    # check the loss in the paper "SMURF"
    pass

def loss_depth_L1(depth, depth_re, mask):
    #デプスが大きい箇所のロスが主になるので悩ましい。検討中。
    pass

def loss_occlusion():
    pass

def loss_rgb_L1(rgb, rgb_re, mask):
    rgb_error = tf.math.abs(rgb - rgb_re)
    loss = tf.reduce_sum(rgb_error*mask) / (tf.reduce_sum(mask) + 1e-7)
    return loss/3.0 #rgb
    
def loss_rgb_ssim_simple(rgb, rgb_re, mask):
    ssim_error, average_weights = weighted_ssim(rgb, rgb_re, mask[...,0])#tf.ones_like(rgb[...,0]))
    #ssim_error_mean = tf.reduce_mean(tf.math.multiply_no_nan(ssim_error, avg_weight))
    ssim_loss = tf.reduce_mean(ssim_error * average_weights)
    
    ##ssim_error_mean = 1.0 - tf.reduce_mean(tf.image.ssim(frame2rgb_resampled, frame1rgb, max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03))
    return ssim_loss

def loss_rgb_ssim(rgb, rgb_re, depth, depth_re, mask):
    # loss weight 0.0 to 1.0 
    # weight increases when depth difference is small.
    # This idea comes from the paper "depth from videos in the wild"
    
    #maskでokエリアだけ見ているものの、無限遠とかの影響デカすぎるかも？逆数(視差)のがよくね？
    
    sqe_depth = (depth - depth_re)**2
    sqe_mean_error = tf.reduce_sum(sqe_depth*mask, axis=[1,2], keepdims=True) / (tf.reduce_sum(mask, axis=[1,2], keepdims=True) + 1e-7)
    ssim_weight = (sqe_mean_error / (sqe_depth + sqe_mean_error + 1e-7)) * mask    
    ssim_weight = tf.squeeze(ssim_weight, axis=-1)
    ssim_weight = tf.stop_gradient(ssim_weight)
    """
    ssim_weight = tf.squeeze(mask, axis=-1)
    """
    ssim_error, average_weights = weighted_ssim(rgb, rgb_re, ssim_weight)
    #ssim_error_mean = tf.reduce_mean(tf.math.multiply_no_nan(ssim_error, avg_weight))
    ssim_loss = tf.reduce_mean(ssim_error * average_weights)
    
    ##ssim_error_mean = 1.0 - tf.reduce_mean(tf.image.ssim(frame2rgb_resampled, frame1rgb, max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03))
    return ssim_loss


def weighted_ssim(x, y, weight, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01):
  """Computes a weighted structured image similarity measure.
  See https://en.wikipedia.org/wiki/Structural_similarity#Algorithm. The only
  difference here is that not all pixels are weighted equally when calculating
  the moments - they are weighted by a weight function.
  Args:
    x: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    y: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    weight: A tf.Tensor of shape [B, H, W], representing the weight of each
      pixel in both images when we come to calculate moments (means and
      correlations).
    c1: A floating point number, regularizes division by zero of the means.
    c2: A floating point number, regularizes division by zero of the second
      moments.
    weight_epsilon: A floating point number, used to regularize division by the
      weight.
  Returns:
    A tuple of two tf.Tensors. First, of shape [B, H-2, W-2, C], is scalar
    similarity loss oer pixel per channel, and the second, of shape
    [B, H-2. W-2, 1], is the average pooled `weight`. It is needed so that we
    know how much to weigh each pixel in the first tensor. For example, if
    `'weight` was very small in some area of the images, the first tensor will
    still assign a loss to these pixels, but we shouldn't take the result too
    seriously.
  """
  def _avg_pool3x3(x):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')
  if c1 == float('inf') and c2 == float('inf'):
    raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                     'likely unintended.')
  weight = tf.expand_dims(weight, -1)
  average_pooled_weight = _avg_pool3x3(weight)
  weight_plus_epsilon = weight + weight_epsilon
  inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

  def weighted_avg_pool3x3(z):
    wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
    return wighted_avg * inverse_average_pooled_weight

  mu_x = weighted_avg_pool3x3(x)
  mu_y = weighted_avg_pool3x3(y)
  sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
  sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
  sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
  if c1 == float('inf'):
    ssim_n = (2 * sigma_xy + c2)
    ssim_d = (sigma_x + sigma_y + c2)
  elif c2 == float('inf'):
    ssim_n = 2 * mu_x * mu_y + c1
    ssim_d = mu_x**2 + mu_y**2 + c1
  else:
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
  result = ssim_n / ssim_d
  return tf.clip_by_value((1 - result) / 2, 0, 1), average_pooled_weight    
    

def metrics_depth_acc(pred_depth, gt_dist, car_mask):
    """
    pred_depth:
        float (batch, height, width, 1)
    car_mask:
        float. mask for depth (batch, height, width, 1)
    gt_dist:
        distance between cars (batch)
    """
    pred_dist = tf.reduce_sum(pred_depth*car_mask, axis=[1,2,3]) / (tf.reduce_sum(car_mask, axis=[1,2,3]) + 1e-7)
    absolute_error = tf.reduce_mean(tf.math.abs(gt_dist - pred_dist))
    return absolute_error

def coords_grid(batch_size, height, width):
    ''' Generate coordinates (xy-order) from given info
    Args:
      batch_size, height, width: int values
    Returns:
      coordinates tensor with shape (bs, h, w, 2), xy-indexing
    '''
    # shape: (height, width)x2
    gy, gx = tf.meshgrid(tf.range(height, dtype=tf.float32),
                         tf.range(width, dtype=tf.float32),
                         indexing='ij')
    # -> (height, width, 2)
    coords = tf.stack([gx, gy], axis=-1)
    # -> (1, height, width, 2)
    coords = tf.expand_dims(coords, axis=0)
    # -> (batch_size, height, width, 2)
    coords = tf.tile(coords, (batch_size, 1, 1, 1))
    return coords    

def occlusion_mask_by_translation(flow_21, coords_2):
    # mask does not need gradient
    flow_21 = tf.stop_gradient(flow_21)
    coords_2 = tf.stop_gradient(coords_2)
    
    # make flatten indices
    batch, height, width, _ = tf.unstack(tf.shape(flow_21))
    warp_coords = coords_2 + flow_21
    warp_coords = tf.math.round(warp_coords)
    x_indices = tf.clip_by_value(warp_coords[:,:,:,0], 0., tf.cast(width, tf.float32)-1.)
    y_indices = tf.clip_by_value(warp_coords[:,:,:,1], 0., tf.cast(height, tf.float32)-1.) * tf.cast(width, tf.float32)
    b_indices = tf.range(batch, dtype=tf.float32)[:, tf.newaxis, tf.newaxis] * tf.cast(width*height, tf.float32)
    warp_indices = tf.reshape(tf.cast(x_indices + y_indices + b_indices, tf.int32), [-1])
    ones_mask = tf.ones((batch*height*width), tf.float32)
    
    #一個加えてソートのほうがはやいかな？
    translated_mask = tf.math.unsorted_segment_mean(ones_mask, warp_indices, num_segments=batch*height*width)
    # reshapeできるかな…？
    translated_mask = tf.reshape(translated_mask, (batch, height, width, 1))
    return translated_mask

def inverse_flow_quick(flow_21, coords_2, flow_rate=1):
    # mask does not need gradient
    flow_21 = tf.stop_gradient(flow_21)
    coords_2 = tf.stop_gradient(coords_2)
    
    # make flatten indices
    batch, height, width, _ = tf.unstack(tf.shape(flow_21))
    resolution_rate = 1
    target_h = height * resolution_rate
    target_w = width * resolution_rate
    warp_coords = (coords_2 + flow_21 / flow_rate) * resolution_rate
    warp_coords = tf.math.round(warp_coords)
    x_indices = tf.clip_by_value(warp_coords[:,:,:,0], 0., tf.cast(target_w, tf.float32)-1.)
    y_indices = tf.clip_by_value(warp_coords[:,:,:,1], 0., tf.cast(target_h, tf.float32)-1.) * tf.cast(target_w, tf.float32)
    b_indices = tf.range(batch, dtype=tf.float32)[:, tf.newaxis, tf.newaxis] * tf.cast(target_w*target_h, tf.float32)
    warp_indices = tf.reshape(tf.cast(x_indices + y_indices + b_indices, tf.int32), [-1])
    
    #flow12 = tf.math.unsorted_segment_sum(tf.reshape(-flow_21,(batch*height*width, 2)), warp_indices, num_segments=batch*target_h*target_w)
    #flow12_count = tf.math.unsorted_segment_sum(tf.ones((batch*height*width, 1), tf.float32), warp_indices, num_segments=batch*target_h*target_w)
    flow_ones = tf.concat([-flow_21, tf.ones((batch,height,width, 1), tf.float32)], axis=-1)
    flow_ones = tf.math.unsorted_segment_sum(tf.reshape(flow_ones,(batch*height*width, 3)), warp_indices, num_segments=batch*target_h*target_w)
    flow_ones = tf.reshape(flow_ones, (batch, target_h, target_w, 3))
    flow12 = flow_ones[...,:2]
    flow12_count = flow_ones[...,2:3]
    exist_mask = flow12_count>0.
    
    flowones_avg = tf.nn.avg_pool2d(flow_ones, 3, 1, padding='SAME')
    flow12_avg = flowones_avg[...,:2]#tf.nn.avg_pool2d(flow12, 3, 1, padding='SAME')
    flow12_count_avg = flowones_avg[...,2:3]#tf.nn.avg_pool2d(flow12_count, 3, 1, padding='SAME')
    
    # fill empty pixels by average values
    fill_value = flow12_avg / (flow12_count_avg+1e-7)
    fill_mask = tf.logical_and(flow12_count<1., flow12_count_avg>0.)
    flow12 = tf.cast(exist_mask, tf.float32) * (flow12 / (flow12_count+1e-7)) + tf.cast(fill_mask, tf.float32) * fill_value
    
    #隙間ができるし解像度が落ちるので、大き目にセグメントるのもいいのかも、sum 演算で除算する。
    
    return flow12    

class RAFT(tf.keras.Model):
    def __init__(self, drop_rate=0, iters=6, iters_pred=6, **kwargs):
        super().__init__(**kwargs)

        self.feature_dim = 128
        self.hidden_dim = 128
        self.context_dim = 128
        context_total_dims = self.hidden_dim + self.context_dim
        self.corr_levels = 4
        self.corr_radius = 4
        self.disparity_depth_coeff = 560.#/0.48
        #self.initial_disparity = 0.5 * (70./8.)/2.

        self.drop_rate = drop_rate

        self.iters = iters
        self.iters_pred = iters_pred

        self.fnet = Res_Encoder(dims=self.feature_dim,
                                 norm_type='instance',#'instance',
                                 )
        self.cnet = Res_Encoder(dims=context_total_dims,
                                 norm_type='batch',#'batch'
                                 )
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)
        self.train_step_counter = tf.Variable(0., trainable=False)
        self.correlation_12 = CorrBlock(num_levels=self.corr_levels,
                                              radius=self.corr_radius,
                                              num_feature=self.feature_dim)
        self.correlation_21 = CorrBlock(num_levels=self.corr_levels,
                                              radius=self.corr_radius,
                                              num_feature=self.feature_dim)
        
    def freeze_layers(self):    
        self.fnet.trainable=False
        self.cnet.trainable=False
        self.update_block.trainable=False
      
      
        
    def build(self, input_shape):
        del input_shape
    
    def flow_to_disparity(self, flow, inf_DPs, is_right=True):
        """
        flow: 
            shape (batch, height, width, 1)
            assuming the positive value. (right view on left coordinates).
        inf_DPs:
            shape (batch) 
        """
        if is_right:
            disparity = flow - tf.reshape(inf_DPs,[-1,1,1,1])
        else:
            disparity = -flow - tf.reshape(inf_DPs,[-1,1,1,1])
        disparity_clip = tf.maximum(disparity, 1e-7)
        dev_disparity = tf.math.abs(disparity - disparity_clip)
        return disparity_clip, dev_disparity
    
    def disparity_to_depth(self, disparity, resolution_rate):#, right_to_left):
        """
        disparity: 
            shape (batch, height, width, 1)
            assuming the positive value. (right view on left coordinates).
        """
        depth = (self.disparity_depth_coeff/tf.reshape(resolution_rate,[-1,1,1,1])) / disparity #0.48 is zoom ratio
        return depth
    
    def flow_to_depth(self, flow, inf_DPs, resolution_rate, is_right=True):
        """
        flow: 
            shape (batch, height, width, 1)
            assuming the positive value. (right view on left coordinates).
        inf_DPs:
            shape (batch) 
        """
        
        disparity, dev_disparity = self.flow_to_disparity(flow, inf_DPs, is_right=is_right)
        depth = self.disparity_to_depth(disparity, resolution_rate)
        #depth = (self.disparity_depth_coeff/tf.reshape(resolution_rate,[-1,1,1,1])) / disparity_clip #0.48 is zoom ratio
        return depth, dev_disparity
        
    
    def limit_flow(self, disparity, inf_DPs, is_right=True):
        if is_right:
            disparity_x = tf.maximum(disparity[...,:1], tf.reshape(inf_DPs,[-1,1,1,1]))
        else:
            disparity_x = tf.minimum(disparity[...,:1], -tf.reshape(inf_DPs,[-1,1,1,1]))
        return tf.concat([disparity_x, disparity[...,1:2]], axis=-1)

    
    def get_hidden_context(self, image, training):
        hidden_context = self.cnet(image, training=training)        
        hidden, context = tf.split(hidden_context, [self.hidden_dim, self.context_dim], axis=-1)
        hidden = tf.tanh(hidden)
        context = tf.nn.relu(context)
        return hidden, context
        

    def initialize_flow(self, image):
        b, h, w, _ = tf.unstack(tf.shape(image))
        
        #initial_flow_on_x = tf.maximum(tf.linspace(0.,1.,h//8) - 0.3, 0.)[tf.newaxis,:,tf.newaxis,tf.newaxis] # 70-0
        #initial_flow = self.initial_disparity * tf.concat([initial_flow_on_x, tf.zeros_like(initial_flow_on_x)], axis=-1)
        
        coords0 = coords_grid(b, h//8, w//8)
        coords1 = coords_grid(b, h//8, w//8)# + initial_flow
        coords2 = coords_grid(b, h//8, w//8)# - initial_flow
        coords = coords_grid(b, h, w)
        return coords0, coords1, coords2, coords

    def _upsample_flow(self, flow, mask):
        ''' Upsample flow (h, w, 2) -> (8xh, 8xw, 2) using convex combination
        Args:
          flow: tensor with shape (bs, h, w, 2)
          mask: tensor with shape (bs, h, w, 64x9), 64=8x8 is the upscale
                9 is the neighborhood pixels in unfolding
        
        Returns:
          upscaled flow with shape (bs, 8xh, 8xw, 2)
        '''
        # flow: (bs, h, w, 2), mask: (bs, h, w, 64*9)
        b, h, w, _ = tf.unstack(tf.shape(flow))
        mask = tf.reshape(mask, (b, h, w, 8, 8, 9, 1))
        mask = tf.nn.softmax(mask, axis=5)

        # flow: (bs, h, w, 2) -> (bs, h, w, 2*9)
        up_flow = tf.image.extract_patches(8*flow,
                                           sizes=(1, 3, 3, 1),
                                           strides=(1, 1, 1, 1),
                                           rates=(1, 1, 1, 1),
                                           padding='SAME')
        up_flow = tf.reshape(up_flow, (b, h, w, 1, 1, 9, 2))
        # (bs, h, w, 8, 8, 9, 2) -> (bs, h, w, 8, 8, 2)
        up_flow = tf.reduce_sum(mask*up_flow, axis=5)
        # (bs, h, w, 8, 8, 2) -> (bs, h, w, 8x8x2)
        up_flow = tf.reshape(up_flow, (b, h, w, -1))
        # (bs, h, w, 8x8x2) -> (bs, 8xh, 8xw, 2)
        return tf.nn.depth_to_space(up_flow, block_size=8)
    
    def upsample_flow(self, flow, mask):
        """Upsample flow [H/8, W/8, 2] -> [H, W, 2] using convex combination."""
        bs, height, width, _ = tf.unstack(tf.shape(flow))
        mask = tf.transpose(mask, perm=[0, 3, 1, 2])
        mask = tf.reshape(mask, [bs, 1, 9, 8, 8, height, width])
        mask = tf.nn.softmax(mask, axis=2)
    
        up_flow = tf.image.extract_patches(
            images=tf.pad(8 * flow, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]]),
            sizes=[1, 3, 3, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        up_flow = tf.reshape(up_flow, [bs, height, width, 1, 1, 9, 2])
        up_flow = tf.transpose(up_flow, [0, 6, 5, 4, 3, 1, 2])
    
        up_flow = tf.math.reduce_sum(mask * up_flow, axis=2)
        up_flow = tf.transpose(up_flow, perm=[0, 4, 2, 5, 3, 1])
        up_flow = tf.reshape(up_flow, [bs, height * 8, width * 8, 2])
        return up_flow
    
    def upsample_flow_bilinear(self, flow, mask):
        """Upsample bilinear [H/8, W/8, 2] -> [H, W, 2]."""
        bs, height, width, _ = tf.unstack(tf.shape(flow))
        return tf.image.resize(flow, size=[height * 8, width * 8]) * 8.0
        #up_flow = tf.reshape(up_flow, [bs, height * 8, width * 8, 2])
        #return up_flow    
    
    #@tf.function
    def call(self, inputs, training, is_inference=False):
        frame_1, frame_2 = inputs
        #image1 = 2*(image1/255.0) - 1.0
        #image2 = 2*(image2/255.0) - 1.0

        # feature extractor -> (bs, h/8, w/8, 256)x2
        fmap_1 = self.fnet(frame_1, training=training)
        fmap_2 = self.fnet(frame_2, training=training)

        # setup correlation values
        #C = CorrBlock
        #C = CorrBlockStereo
        self.correlation_12.build_pyramid(fmap_1, fmap_2)
        #(fmap_1, fmap_2,
        #                        num_levels=self.corr_levels,
        #                        radius=self.corr_radius,
        #                        num_feature=self.feature_dim)
        if not is_inference:
            self.correlation_21.build_pyramid(fmap_2, fmap_1)
            #,
            #                        num_levels=self.corr_levels,
            #                        radius=self.corr_radius,
            #                        num_feature=self.feature_dim)
        
        # context network -> (bs, h/8, w/8, hdim+cdim)
        hidden_1, context_1 = self.get_hidden_context(frame_1, training=training)
        if not is_inference:
            hidden_2, context_2 = self.get_hidden_context(frame_2, training=training)

        # (bs, h/8, w/8, 2)x2, xy-indexing
        coords_origin, coords_1, coords_2, coords_hres = self.initialize_flow(frame_1)

        predictions = []
        iters = self.iters if training else self.iters_pred        

        for i in range(iters):
            coords_1 = tf.stop_gradient(coords_1)#apply gradient on only delta 
            # FLOW from frame1 to 2
            # (bs, h, w, 81xnum_levels)
            corr = self.correlation_12.retrieve(coords_1)
            flow = coords_1 - coords_origin
            # (bs, h, w, *), net: hdim, up_mask: 64x9, delta_flow: 2
            hidden_1, up_mask_1, delta_flow = self.update_block([hidden_1, context_1, corr, flow])
            coords_1 += delta_flow
            #flow_limit = self.limit_flow(coords_1-coords_origin, inf_DPs/8, is_right=True)
            #coords_1 = coords_origin + flow_limit
            
            # upsample prediction
            #disparity_1_low_res, _ = self.flow_to_disparity(coords_1[...,:1] - coords_origin[...,:1],
            #                                             inf_DPs/8.0, is_right=True)            
            #for save in numpy format
            flow_1_low_res = coords_1[...,:2] - coords_origin[...,:2]
            
            #flow_up_1 = self.upsample_flow(coords_1 - coords_origin, up_mask_1)
            flow_up_1 = self.upsample_flow_bilinear(coords_1 - coords_origin, up_mask_1)
            #flow_up_1 = self.limit_flow(flow_up_1, inf_DPs, is_right=True)
            #frame_1_reconst_from_2, out_range_mask_12 = warp_bilinear(frame_2, coords_hres + flow_up_1)
            ##depth_1, dev_disparity_1 = self.flow_to_depth(flow_up_1[...,:1], inf_DPs, resolution_rate, is_right=True)#only x disparity is enough        
            
            
            
            if not is_inference:
                frame_1_reconst_from_2, _ = warp_bilinear_tfa(frame_2, coords_hres + flow_up_1)
                
                occulusion_mask_for_2 = occlusion_mask_by_translation(flow_up_1, coords_hres)
                
                coords_2 = tf.stop_gradient(coords_2)
                corr = self.correlation_21.retrieve(coords_2)
                flow = coords_2 - coords_origin
                hidden_2, up_mask_2, delta_flow = self.update_block([hidden_2, context_2, corr, flow])
                coords_2 += delta_flow
                #flow_limit = self.limit_flow(coords_2-coords_origin, inf_DPs/8, is_right=False)
                #coords_2 = coords_origin + flow_limit
                #disparity_2_low_res, _ = self.flow_to_disparity(coords_2[...,:1] - coords_origin[...,:1], 
                #                                             inf_DPs/8.0, is_right=False)
                #for save in numpy format
                flow_2_low_res = coords_2[...,:2] - coords_origin[...,:2]
                
                #disparity_2_low_res_warp, _ = warp_bilinear_tfa(disparity_2_low_res, coords_1)
                #flow_up_2 = self.upsample_flow(coords_2 - coords_origin, up_mask_2)
                flow_up_2 = self.upsample_flow_bilinear(coords_2 - coords_origin, up_mask_2)
                #flow_up_2 = self.limit_flow(flow_up_2, inf_DPs, is_right=False)
                
                #frame_2_reconst_from_1, out_range_mask_21 = warp_bilinear(frame_1, coords_hres + flow_up_2)
                frame_2_reconst_from_1, _ = warp_bilinear_tfa(frame_1, coords_hres + flow_up_2)
                occulusion_mask_for_1 = occlusion_mask_by_translation(flow_up_2, coords_hres)

                ##depth_2, dev_disparity_2 = self.flow_to_depth(flow_up_2[...,:1], inf_DPs, resolution_rate, is_right=False)#minus, ok?
                #mask_1_reconst_from_2 = warp_bilinear(depth_2, coords_hres + flow_up_1, only_mask=True)
                #mask_2_reconst_from_1 = warp_bilinear(depth_1, coords_hres + flow_up_2, only_mask=True)
                #depth_1_reconst_from_2 = tfa.image.resampler(depth_2, coords_hres + flow_up_1)
                #depth_2_reconst_from_1 = tfa.image.resampler(depth_1, coords_hres + flow_up_2)
                
                #オクルージョンマスク復活させる。
                ##depth_1_reconst_from_2, mask_1_reconst_from_2 = warp_bilinear_tfa(depth_2, coords_hres + flow_up_1)
                ##depth_2_reconst_from_1, mask_2_reconst_from_1 = warp_bilinear_tfa(depth_1, coords_hres + flow_up_2)
                
                predictions.append({"rgb_1_re": frame_1_reconst_from_2,
                                         "rgb_2_re": frame_2_reconst_from_1,
                                         #"depth_1": depth_1,
                                         #"depth_2": depth_2,
                                         "flow_1": flow_up_1,
                                         "flow_2": flow_up_2,
                                         "flow_1_low_res": flow_1_low_res,
                                         "flow_2_low_res": flow_2_low_res,
                                         #"up_mask_1": up_mask_1,
                                         #"up_mask_2": up_mask_2,
                                         #"mask_1_re": mask_1_reconst_from_2,
                                         #"mask_2_re": mask_2_reconst_from_1,
                                         #"devdisparity_1": dev_disparity_1,
                                         #"devdisparity_2": dev_disparity_2,
                                         "occulusion_mask_for_1": occulusion_mask_for_1,
                                         "occulusion_mask_for_2": occulusion_mask_for_2,
                                         })
            else:
                predictions.append({#"rgb_re": frame_1_reconst_from_2,
                                    "flow_1_low_res": flow_1_low_res,
                                    #"up_mask_1": up_mask_1,
                                    "flow_1": flow_up_1,
                                    })
                

        # flow_predictions[-1] is the finest output
        return predictions

    def quick_inference(self, inputs, training, both_flow=False):
        frame_1, frame_2 = inputs
        
        # feature extractor -> (bs, h/8, w/8, 256)x2
        fmap_1 = self.fnet(frame_1, training=training)
        fmap_2 = self.fnet(frame_2, training=training)
        self.correlation_12.build_pyramid(fmap_1, fmap_2)
        
        #if both_flow:
        #    self.correlation_21.build_pyramid(fmap_2, fmap_1)
            #,
            #                        num_levels=self.corr_levels,
            #                        radius=self.corr_radius,
            #                        num_feature=self.feature_dim)
        
        # context network -> (bs, h/8, w/8, hdim+cdim)
        hidden_1, context_1 = self.get_hidden_context(frame_1, training=training)
        #if both_flow:
        #    hidden_2, context_2 = self.get_hidden_context(frame_2, training=training)

        # (bs, h/8, w/8, 2)x2, xy-indexing
        coords_origin, coords_1, coords_2, coords_hres = self.initialize_flow(frame_1)

        predictions = []
        iters = self.iters if training else self.iters_pred  
        #return hidden_1, hidden_2, fmap_1, fmap_2
        
        for i in range(iters):
            #coords_1 = tf.stop_gradient(coords_1)#apply gradient on only delta 
            # FLOW from frame1 to 2
            # (bs, h, w, 81xnum_levels)
            corr = self.correlation_12.retrieve(coords_1)
            
            flow = coords_1 - coords_origin
            # (bs, h, w, *), net: hdim, up_mask: 64x9, delta_flow: 2
            hidden_1, delta_flow = self.update_block.quick_inference([hidden_1, context_1, corr, flow])
            coords_1 += delta_flow
            
            # upsample prediction
            #disparity_1_low_res, _ = self.flow_to_disparity(coords_1[...,:1] - coords_origin[...,:1],
            #                                             inf_DPs/8.0, is_right=True)            
            #for save in numpy format
            flow_1_low_res = coords_1[...,:2] - coords_origin[...,:2]
            
            ##flow_up_1 = self.upsample_flow_bilinear(coords_1 - coords_origin, up_mask_1)
            
            
            """
            if both_flow:
                #frame_1_reconst_from_2, _ = warp_bilinear_tfa(frame_2, coords_hres + flow_up_1)
                
                #occulusion_mask_for_2 = occlusion_mask_by_translation(flow_up_1, coords_hres)
                
                #coords_2 = tf.stop_gradient(coords_2)
                corr = self.correlation_21.retrieve(coords_2)
                flow = coords_2 - coords_origin
                hidden_2, delta_flow = self.update_block.quick_inference([hidden_2, context_2, corr, flow])
                coords_2 += delta_flow
                #flow_limit = self.limit_flow(coords_2-coords_origin, inf_DPs/8, is_right=False)
                #coords_2 = coords_origin + flow_limit
                #disparity_2_low_res, _ = self.flow_to_disparity(coords_2[...,:1] - coords_origin[...,:1], 
                #                                             inf_DPs/8.0, is_right=False)
                #for save in numpy format
                flow_2_low_res = coords_2[...,:2] - coords_origin[...,:2]
                
                #disparity_2_low_res_warp, _ = warp_bilinear_tfa(disparity_2_low_res, coords_1)
                #flow_up_2 = self.upsample_flow(coords_2 - coords_origin, up_mask_2)
                #flow_up_2 = self.upsample_flow_bilinear(coords_2 - coords_origin, up_mask_2)
                #flow_up_2 = self.limit_flow(flow_up_2, inf_DPs, is_right=False)
                
                #frame_2_reconst_from_1, out_range_mask_21 = warp_bilinear(frame_1, coords_hres + flow_up_2)
                #frame_2_reconst_from_1, _ = warp_bilinear_tfa(frame_1, coords_hres + flow_up_2)
                #occulusion_mask_for_1 = occlusion_mask_by_translation(flow_up_2, coords_hres)

                ##depth_2, dev_disparity_2 = self.flow_to_depth(flow_up_2[...,:1], inf_DPs, resolution_rate, is_right=False)#minus, ok?
                #mask_1_reconst_from_2 = warp_bilinear(depth_2, coords_hres + flow_up_1, only_mask=True)
                #mask_2_reconst_from_1 = warp_bilinear(depth_1, coords_hres + flow_up_2, only_mask=True)
                #depth_1_reconst_from_2 = tfa.image.resampler(depth_2, coords_hres + flow_up_1)
                #depth_2_reconst_from_1 = tfa.image.resampler(depth_1, coords_hres + flow_up_2)
                
                #オクルージョンマスク復活させる。
                ##depth_1_reconst_from_2, mask_1_reconst_from_2 = warp_bilinear_tfa(depth_2, coords_hres + flow_up_1)
                ##depth_2_reconst_from_1, mask_2_reconst_from_1 = warp_bilinear_tfa(depth_1, coords_hres + flow_up_2)
                
                predictions.append({
                                         "flow_1_low_res": flow_1_low_res,
                                         "flow_2_low_res": flow_2_low_res,
                                        })
            else:
            """
            predictions.append({
                                    "flow_1_low_res": flow_1_low_res,
                                    })
        if both_flow:
            predictions[-1]["flow_2_low_res"] = inverse_flow_quick(predictions[-1]["flow_1_low_res"], coords_1)

        return predictions


    def ____call(self, inputs, training, is_inference=False):
        frame_1, frame_2, inf_DPs = inputs
        #image1 = 2*(image1/255.0) - 1.0
        #image2 = 2*(image2/255.0) - 1.0

        # feature extractor -> (bs, h/8, w/8, 256)x2
        fmap_1 = self.fnet(frame_1, training=training)
        fmap_2 = self.fnet(frame_2, training=training)

        # setup correlation values
        correlation_12 = CorrBlock(fmap_1, fmap_2,
                                num_levels=self.corr_levels,
                                radius=self.corr_radius,
                                num_feature=self.feature_dim)
        if not is_inference:
            correlation_21 = CorrBlock(fmap_2, fmap_1,
                                    num_levels=self.corr_levels,
                                    radius=self.corr_radius,
                                    num_feature=self.feature_dim)

        # context network -> (bs, h/8, w/8, hdim+cdim)
        # split -> (bs, h/8, w/8, hdim), (bs, h/8, w/8, cdim)
        #hidden_context_1 = self.cnet(frame_1, training=training)        
        #hidden_1, context_1 = tf.split(hidden_context_1, [self.hidden_dim, self.context_dim], axis=-1)
        #hidden_1 = tf.tanh(hidden_1)
        #context_1 = tf.nn.relu(context_1)
        hidden_1, context_1 = self.get_hidden_context(frame_1, training=training)
        if not is_inference:
            hidden_2, context_2 = self.get_hidden_context(frame_2, training=training)

        # (bs, h/8, w/8, 2)x2, xy-indexing
        coords_origin, coords_1, coords_2, coords_hres = self.initialize_flow(frame_1)

        predictions = []
        iters = self.iters if training else self.iters_pred

        def cond(iter_num, coords_1, coords_2, hidden_1, hidden_2, predictions):
            return iter_num < iters
    
        def body(iter_num, coords_1, coords_2, hidden_1, hidden_2, predictions):
            coords_1 = tf.stop_gradient(coords_1)#apply gradient on only delta 
            # FLOW from frame1 to 2
            # (bs, h, w, 81xnum_levels)
            corr = correlation_12.retrieve(coords_1)
            flow = coords_1 - coords_origin
            # (bs, h, w, *), net: hdim, up_mask: 64x9, delta_flow: 2
            hidden_1, up_mask, delta_flow = self.update_block([hidden_1, context_1, corr, flow])
            coords_1 += delta_flow
            # upsample prediction
            flow_up_1 = self.upsample_flow(coords_1 - coords_origin, up_mask)
            frame_1_reconst_from_2, out_range_mask_12 = warp_bilinear(frame_2, coords_hres + flow_up_1)
            depth_1 = self.disparity_to_depth(flow_up_1[...,:1], inf_DPs)#only x disparity is enough
            
            coords_2 = tf.stop_gradient(coords_2)

            corr = correlation_21.retrieve(coords_2)
            flow = coords_2 - coords_origin
            hidden_2, up_mask, delta_flow = self.update_block([hidden_2, context_2, corr, flow])
            coords_2 += delta_flow
            flow_up_2 = self.upsample_flow(coords_2 - coords_origin, up_mask)
            frame_2_reconst_from_1, out_range_mask_21 = warp_bilinear(frame_1, coords_hres + flow_up_2)
            depth_2 = self.disparity_to_depth(-flow_up_2[...,:1], -inf_DPs)#minus, ok?
            
            depth_1_reconst_from_2, mask_1_reconst_from_2 = warp_bilinear(depth_2, coords_hres + flow_up_1)
            depth_2_reconst_from_1, mask_2_reconst_from_1 = warp_bilinear(depth_1, coords_hres + flow_up_2)
            
            predictions["rgb_1_re"] = predictions["rgb_1_re"].write(iter_num, frame_1_reconst_from_2)
            predictions["rgb_2_re"] = predictions["rgb_2_re"].write(iter_num, frame_2_reconst_from_1)
            predictions["depth_1"] = predictions["depth_1"].write(iter_num, depth_1)
            predictions["depth_2"] = predictions["depth_2"].write(iter_num, depth_2)
            predictions["flow_1"] = predictions["flow_1"].write(iter_num, flow_up_1)
            predictions["flow_2"] = predictions["flow_2"].write(iter_num, flow_up_2)
            predictions["depth_1_re"] = predictions["depth_1_re"].write(iter_num, depth_1_reconst_from_2)
            predictions["depth_2_re"] = predictions["depth_2_re"].write(iter_num, depth_2_reconst_from_1)
            predictions["mask_1_re"] = predictions["mask_1_re"].write(iter_num, mask_1_reconst_from_2)
            predictions["mask_2_re"] = predictions["mask_2_re"].write(iter_num, mask_2_reconst_from_1)
            
            iter_num += 1
            return [iter_num, coords_1, coords_2, hidden_1, hidden_2, predictions]
    
        
        predictions = ({"rgb_1_re": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "rgb_2_re": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "depth_1": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "depth_2": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "flow_1": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "flow_2": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "depth_1_re": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "depth_2_re": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "mask_1_re": tf.TensorArray(dtype=tf.bool, size=iters, dynamic_size=False),
                            "mask_2_re": tf.TensorArray(dtype=tf.bool, size=iters, dynamic_size=False),
                                         })
        iter_num = 0
        loop_vars = [iter_num, 
                     #frame_1, frame_2, coords_origin, 
                     coords_1, coords_2, 
                     #correlation_12, correlation_21, 
                     hidden_1, hidden_2,
                     #context_1, context_2, inf_DPs, 
                     predictions]
        _, _, _, _, _, predictions = tf.while_loop(cond,body,
                                                   loop_vars,
                                                   parallel_iterations=1,
                                                   maximum_iterations=iters)
        for key in predictions.keys():
            predictions[key] = predictions[key].stack()
            
        new_predictions = []
        for i in range(iters):
            new_predictions.append({"rgb_1_re": predictions["rgb_1_re"][i],
                                   "rgb_2_re": predictions["rgb_2_re"][i],
                                   "depth_1": predictions["depth_1"][i],
                                   "depth_2": predictions["depth_2"][i],
                                   "flow_1": predictions["flow_1"][i],
                                   "flow_2": predictions["flow_2"][i],
                                   "depth_1_re": predictions["depth_1_re"][i],
                                   "depth_2_re": predictions["depth_2_re"][i],
                                   "mask_1_re": predictions["mask_1_re"][i],
                                   "mask_2_re": predictions["mask_2_re"][i],
                                   })
        
        return new_predictions


    def compile(self, optimizer, clip_norm, 
                loss, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.loss = loss

        self.custom_metrics = OrderedDict({
            'loss': tf.keras.metrics.Mean(name='loss'),
            'loss_rgb': tf.keras.metrics.Mean(name='loss_rgb'),
            'loss_census': tf.keras.metrics.Mean(name='loss_census'),
            'loss_ssim': tf.keras.metrics.Mean(name='loss_ssim'),
            'loss_smooth': tf.keras.metrics.Mean(name='loss_smooth'),
            #'loss_depth_l1': tf.keras.metrics.Mean(name='loss_depth_l1'),
            'loss_iter_0': tf.keras.metrics.Mean(name='loss_iter_0'),
            'loss_iter_2': tf.keras.metrics.Mean(name='loss_iter_2'),
            'loss_iter_4': tf.keras.metrics.Mean(name='loss_iter_4'),
            'loss_iter_fin': tf.keras.metrics.Mean(name='loss_iter_fin'),
            'loss_f1': tf.keras.metrics.Mean(name='loss_f1'),
            'loss_f2': tf.keras.metrics.Mean(name='loss_f2'),
        })

    #@tf.function
    def train_step(self, data):
        self.train_step_counter.assign_add(1.0)
        #dataloaderに含める
        #image1, image2, flow, valid = data
        #image1 = tf.cast(image1, dtype=tf.float32)
        #image2 = tf.cast(image2, dtype=tf.float32)
        
        inputs = data#[frame_1, frame_2], [gt_dist, car_mask]
        
        with tf.GradientTape() as tape:
            predictions = self([inputs["frame_1"], inputs["frame_2"], 
                                #inputs["inf_DP"], inputs["res_rate"],
                                ], training=True)
            loss, loss_iter = self.loss(inputs["frame_1"], inputs["frame_2"], 
                                        ##inputs["box_mask"], inputs["box_mask_l"],
                                        ##targets["dist"],
                                        predictions, 
                                        self.train_step_counter)
        grads = tape.gradient(loss["total"], self.trainable_weights)
        ##grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.custom_metrics['loss'].update_state(loss["total"])
        self.custom_metrics['loss_rgb'].update_state(loss["rgb"])
        self.custom_metrics['loss_census'].update_state(loss["census"])
        self.custom_metrics['loss_ssim'].update_state(loss["ssim"])
        self.custom_metrics['loss_smooth'].update_state(loss["smooth"])
        #self.custom_metrics['loss_depth_l1'].update_state(loss["depth_l1"])
        self.custom_metrics['loss_iter_0'].update_state(loss_iter[0][0] + loss_iter[1][0])
        self.custom_metrics['loss_iter_2'].update_state(loss_iter[0][2] + loss_iter[1][2])
        self.custom_metrics['loss_iter_4'].update_state(loss_iter[0][4] + loss_iter[1][4])
        self.custom_metrics['loss_iter_fin'].update_state(loss_iter[0][-1] + loss_iter[1][-1])
        self.custom_metrics['loss_f1'].update_state(tf.reduce_sum(tf.stack(loss_iter[0])))
        self.custom_metrics['loss_f2'].update_state(tf.reduce_sum(tf.stack(loss_iter[1])))
        
        return {k: m.result() for k, m in self.custom_metrics.items()}

    def test_step(self, data):
        inputs = data#[frame_1, frame_2], [gt_dist, car_mask]
        
        predictions = self([inputs["frame_1"], inputs["frame_2"], 
                            #inputs["inf_DP"], inputs["res_rate"],
                            ], training=False)
        loss, loss_iter = self.loss(inputs["frame_1"], inputs["frame_2"], 
                                    #inputs["box_mask"], inputs["box_mask_l"],
                                    #targets["dist"],
                                    predictions)

        self.custom_metrics['loss'].update_state(loss["total"])
        self.custom_metrics['loss_rgb'].update_state(loss["rgb"])
        self.custom_metrics['loss_census'].update_state(loss["census"])
        self.custom_metrics['loss_ssim'].update_state(loss["ssim"])
        self.custom_metrics['loss_smooth'].update_state(loss["smooth"])
        #self.custom_metrics['loss_depth_l1'].update_state(loss["depth_l1"])
        self.custom_metrics['loss_iter_0'].update_state(loss_iter[0][0] + loss_iter[1][0])
        self.custom_metrics['loss_iter_2'].update_state(loss_iter[0][2] + loss_iter[1][2])
        self.custom_metrics['loss_iter_4'].update_state(loss_iter[0][4] + loss_iter[1][4])
        self.custom_metrics['loss_iter_fin'].update_state(loss_iter[0][-1] + loss_iter[1][-1])
        self.custom_metrics['loss_f1'].update_state(tf.reduce_sum(tf.stack(loss_iter[0])))
        self.custom_metrics['loss_f2'].update_state(tf.reduce_sum(tf.stack(loss_iter[1])))


        return {k: m.result() for k, m in self.custom_metrics.items()}

    def predict_step(self, data):
        frame_1, frame_2, *_ = data
        predictions = self([frame_1, frame_2], training=False, is_inference=True)
        return predictions[-1]

    def reset_metrics(self):
        for k, m in self.custom_metrics.items():
            m.reset_states()




class StereoDepth(tf.keras.Model):
    def __init__(self, drop_rate=0, iters=6, iters_pred=6, **kwargs):
        super().__init__(**kwargs)

        self.feature_dim = 128
        self.hidden_dim = 128
        self.context_dim = 128
        context_total_dims = self.hidden_dim + self.context_dim
        self.corr_levels = 4
        self.corr_radius = 4
        self.disparity_depth_coeff = 560.#/0.48
        self.initial_disparity = 0.5 * (70./8.)/2.

        self.drop_rate = drop_rate

        self.iters = iters
        self.iters_pred = iters_pred

        self.fnet = Res_Encoder(dims=self.feature_dim,
                                 norm_type='instance',#'instance',
                                 )
        self.cnet = Res_Encoder(dims=context_total_dims,
                                 norm_type='batch',#'batch'
                                 )
        self.update_block = UpdateBlockStereo(hidden_dim=self.hidden_dim)
        self.train_step_counter = tf.Variable(0., trainable=False)
        self.correlation_12 = CorrBlockStereo(num_levels=self.corr_levels,
                                              radius=self.corr_radius,
                                              num_feature=self.feature_dim)
        self.correlation_21 = CorrBlockStereo(num_levels=self.corr_levels,
                                              radius=self.corr_radius,
                                              num_feature=self.feature_dim)
        
    def freeze_layers(self):    
        self.fnet.trainable=False
        self.cnet.trainable=False
        self.update_block.trainable=False
      
      
        
    def build(self, input_shape):
        del input_shape
    
    def flow_to_disparity(self, flow, inf_DPs, is_right=True):
        """
        flow: 
            shape (batch, height, width, 1)
            assuming the positive value. (right view on left coordinates).
        inf_DPs:
            shape (batch) 
        """
        if is_right:
            disparity = flow - tf.reshape(inf_DPs,[-1,1,1,1])
        else:
            disparity = -flow - tf.reshape(inf_DPs,[-1,1,1,1])
        disparity_clip = tf.maximum(disparity, 1e-7)
        dev_disparity = tf.math.abs(disparity - disparity_clip)
        return disparity_clip, dev_disparity
    
    def disparity_to_depth(self, disparity, resolution_rate):#, right_to_left):
        """
        disparity: 
            shape (batch, height, width, 1)
            assuming the positive value. (right view on left coordinates).
        """
        depth = (self.disparity_depth_coeff/tf.reshape(resolution_rate,[-1,1,1,1])) / disparity #0.48 is zoom ratio
        return depth
    
    def flow_to_depth(self, flow, inf_DPs, resolution_rate, is_right=True):
        """
        flow: 
            shape (batch, height, width, 1)
            assuming the positive value. (right view on left coordinates).
        inf_DPs:
            shape (batch) 
        """
        
        disparity, dev_disparity = self.flow_to_disparity(flow, inf_DPs, is_right=is_right)
        depth = self.disparity_to_depth(disparity, resolution_rate)
        #depth = (self.disparity_depth_coeff/tf.reshape(resolution_rate,[-1,1,1,1])) / disparity_clip #0.48 is zoom ratio
        return depth, dev_disparity
        
    
    def limit_flow(self, disparity, inf_DPs, is_right=True):
        if is_right:
            disparity_x = tf.maximum(disparity[...,:1], tf.reshape(inf_DPs,[-1,1,1,1]))
        else:
            disparity_x = tf.minimum(disparity[...,:1], -tf.reshape(inf_DPs,[-1,1,1,1]))
        return tf.concat([disparity_x, disparity[...,1:2]], axis=-1)

    
    def get_hidden_context(self, image, training):
        hidden_context = self.cnet(image, training=training)        
        hidden, context = tf.split(hidden_context, [self.hidden_dim, self.context_dim], axis=-1)
        hidden = tf.tanh(hidden)
        context = tf.nn.relu(context)
        return hidden, context
        

    def initialize_flow(self, image):
        b, h, w, _ = tf.unstack(tf.shape(image))
        
        initial_flow_on_x = tf.maximum(tf.linspace(0.,1.,h//8) - 0.3, 0.)[tf.newaxis,:,tf.newaxis,tf.newaxis] # 70-0
        initial_flow = self.initial_disparity * tf.concat([initial_flow_on_x, tf.zeros_like(initial_flow_on_x)], axis=-1)
        
        coords0 = coords_grid(b, h//8, w//8)
        coords1 = coords_grid(b, h//8, w//8) + initial_flow
        coords2 = coords_grid(b, h//8, w//8) - initial_flow
        coords = coords_grid(b, h, w)
        return coords0, coords1, coords2, coords

    def _upsample_flow(self, flow, mask):
        ''' Upsample flow (h, w, 2) -> (8xh, 8xw, 2) using convex combination
        Args:
          flow: tensor with shape (bs, h, w, 2)
          mask: tensor with shape (bs, h, w, 64x9), 64=8x8 is the upscale
                9 is the neighborhood pixels in unfolding
        
        Returns:
          upscaled flow with shape (bs, 8xh, 8xw, 2)
        '''
        # flow: (bs, h, w, 2), mask: (bs, h, w, 64*9)
        b, h, w, _ = tf.unstack(tf.shape(flow))
        mask = tf.reshape(mask, (b, h, w, 8, 8, 9, 1))
        mask = tf.nn.softmax(mask, axis=5)

        # flow: (bs, h, w, 2) -> (bs, h, w, 2*9)
        up_flow = tf.image.extract_patches(8*flow,
                                           sizes=(1, 3, 3, 1),
                                           strides=(1, 1, 1, 1),
                                           rates=(1, 1, 1, 1),
                                           padding='SAME')
        up_flow = tf.reshape(up_flow, (b, h, w, 1, 1, 9, 2))
        # (bs, h, w, 8, 8, 9, 2) -> (bs, h, w, 8, 8, 2)
        up_flow = tf.reduce_sum(mask*up_flow, axis=5)
        # (bs, h, w, 8, 8, 2) -> (bs, h, w, 8x8x2)
        up_flow = tf.reshape(up_flow, (b, h, w, -1))
        # (bs, h, w, 8x8x2) -> (bs, 8xh, 8xw, 2)
        return tf.nn.depth_to_space(up_flow, block_size=8)
    
    def upsample_flow(self, flow, mask):
        """Upsample flow [H/8, W/8, 2] -> [H, W, 2] using convex combination."""
        bs, height, width, _ = tf.unstack(tf.shape(flow))
        mask = tf.transpose(mask, perm=[0, 3, 1, 2])
        mask = tf.reshape(mask, [bs, 1, 9, 8, 8, height, width])
        mask = tf.nn.softmax(mask, axis=2)
    
        up_flow = tf.image.extract_patches(
            images=tf.pad(8 * flow, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]]),
            sizes=[1, 3, 3, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        up_flow = tf.reshape(up_flow, [bs, height, width, 1, 1, 9, 2])
        up_flow = tf.transpose(up_flow, [0, 6, 5, 4, 3, 1, 2])
    
        up_flow = tf.math.reduce_sum(mask * up_flow, axis=2)
        up_flow = tf.transpose(up_flow, perm=[0, 4, 2, 5, 3, 1])
        up_flow = tf.reshape(up_flow, [bs, height * 8, width * 8, 2])
        return up_flow
    
    def _upsample_flow(self, flow, mask):
        """Upsample bilinear [H/8, W/8, 2] -> [H, W, 2]."""
        bs, height, width, _ = tf.unstack(tf.shape(flow))
        return tf.image.resize(flow, size=[height * 8, width * 8]) * 8.0
        #up_flow = tf.reshape(up_flow, [bs, height * 8, width * 8, 2])
        #return up_flow    
    
    #@tf.function
    def call(self, inputs, training, is_inference=False):
        frame_1, frame_2, inf_DPs, resolution_rate = inputs
        #image1 = 2*(image1/255.0) - 1.0
        #image2 = 2*(image2/255.0) - 1.0

        # feature extractor -> (bs, h/8, w/8, 256)x2
        fmap_1 = self.fnet(frame_1, training=training)
        fmap_2 = self.fnet(frame_2, training=training)

        # setup correlation values
        #C = CorrBlock
        #C = CorrBlockStereo
        self.correlation_12.build_pyramid(fmap_1, fmap_2)
        #(fmap_1, fmap_2,
        #                        num_levels=self.corr_levels,
        #                        radius=self.corr_radius,
        #                        num_feature=self.feature_dim)
        if not is_inference:
            self.correlation_21.build_pyramid(fmap_2, fmap_1)
            #,
            #                        num_levels=self.corr_levels,
            #                        radius=self.corr_radius,
            #                        num_feature=self.feature_dim)
        
        # context network -> (bs, h/8, w/8, hdim+cdim)
        hidden_1, context_1 = self.get_hidden_context(frame_1, training=training)
        if not is_inference:
            hidden_2, context_2 = self.get_hidden_context(frame_2, training=training)

        # (bs, h/8, w/8, 2)x2, xy-indexing
        coords_origin, coords_1, coords_2, coords_hres = self.initialize_flow(frame_1)

        predictions = []
        iters = self.iters if training else self.iters_pred        

        for i in range(iters):
            coords_1 = tf.stop_gradient(coords_1)#apply gradient on only delta 
            # FLOW from frame1 to 2
            # (bs, h, w, 81xnum_levels)
            corr = self.correlation_12.retrieve(coords_1)
            flow = coords_1 - coords_origin
            # (bs, h, w, *), net: hdim, up_mask: 64x9, delta_flow: 2
            hidden_1, up_mask, delta_flow = self.update_block([hidden_1, context_1, corr, flow])
            coords_1 += delta_flow
            #flow_limit = self.limit_flow(coords_1-coords_origin, inf_DPs/8, is_right=True)
            #coords_1 = coords_origin + flow_limit
            
            # upsample prediction
            #disparity_1_low_res, _ = self.flow_to_disparity(coords_1[...,:1] - coords_origin[...,:1],
            #                                             inf_DPs/8.0, is_right=True)            
            #for save in numpy format
            disparity_1_low_res = coords_1[...,:1] - coords_origin[...,:1]
            
            flow_up_1 = self.upsample_flow(coords_1 - coords_origin, up_mask)
            #flow_up_1 = self.limit_flow(flow_up_1, inf_DPs, is_right=True)
            #frame_1_reconst_from_2, out_range_mask_12 = warp_bilinear(frame_2, coords_hres + flow_up_1)
            frame_1_reconst_from_2, _ = warp_bilinear_tfa(frame_2, coords_hres + flow_up_1)
            depth_1, dev_disparity_1 = self.flow_to_depth(flow_up_1[...,:1], inf_DPs, resolution_rate, is_right=True)#only x disparity is enough        
                
            if not is_inference:
                coords_2 = tf.stop_gradient(coords_2)
                corr = self.correlation_21.retrieve(coords_2)
                flow = coords_2 - coords_origin
                hidden_2, up_mask, delta_flow = self.update_block([hidden_2, context_2, corr, flow])
                coords_2 += delta_flow
                #flow_limit = self.limit_flow(coords_2-coords_origin, inf_DPs/8, is_right=False)
                #coords_2 = coords_origin + flow_limit
                #disparity_2_low_res, _ = self.flow_to_disparity(coords_2[...,:1] - coords_origin[...,:1], 
                #                                             inf_DPs/8.0, is_right=False)
                #for save in numpy format
                disparity_2_low_res = -(coords_2[...,:1] - coords_origin[...,:1])
                
                #disparity_2_low_res_warp, _ = warp_bilinear_tfa(disparity_2_low_res, coords_1)
                flow_up_2 = self.upsample_flow(coords_2 - coords_origin, up_mask)
                #flow_up_2 = self.limit_flow(flow_up_2, inf_DPs, is_right=False)
                
                #frame_2_reconst_from_1, out_range_mask_21 = warp_bilinear(frame_1, coords_hres + flow_up_2)
                frame_2_reconst_from_1, _ = warp_bilinear_tfa(frame_1, coords_hres + flow_up_2)

                depth_2, dev_disparity_2 = self.flow_to_depth(flow_up_2[...,:1], inf_DPs, resolution_rate, is_right=False)#minus, ok?
                #mask_1_reconst_from_2 = warp_bilinear(depth_2, coords_hres + flow_up_1, only_mask=True)
                #mask_2_reconst_from_1 = warp_bilinear(depth_1, coords_hres + flow_up_2, only_mask=True)
                #depth_1_reconst_from_2 = tfa.image.resampler(depth_2, coords_hres + flow_up_1)
                #depth_2_reconst_from_1 = tfa.image.resampler(depth_1, coords_hres + flow_up_2)
                depth_1_reconst_from_2, mask_1_reconst_from_2 = warp_bilinear_tfa(depth_2, coords_hres + flow_up_1)
                depth_2_reconst_from_1, mask_2_reconst_from_1 = warp_bilinear_tfa(depth_1, coords_hres + flow_up_2)
                
                predictions.append({"rgb_1_re": frame_1_reconst_from_2,
                                         "rgb_2_re": frame_2_reconst_from_1,
                                         "depth_1": depth_1,
                                         "depth_2": depth_2,
                                         "flow_1": flow_up_1,
                                         "flow_2": flow_up_2,
                                         "disparity_1_low_res": disparity_1_low_res,
                                         "disparity_2_low_res": disparity_2_low_res,                                        "depth_1_re": depth_1_reconst_from_2,
                                         "depth_2_re": depth_2_reconst_from_1,
                                         "mask_1_re": mask_1_reconst_from_2,
                                         "mask_2_re": mask_2_reconst_from_1,
                                         "devdisparity_1": dev_disparity_1,
                                         "devdisparity_2": dev_disparity_2,
                                         })
            else:
                predictions.append({"rgb_re": frame_1_reconst_from_2,
                                    "depth": depth_1,
                                    })

        # flow_predictions[-1] is the finest output
        return predictions


    def ____call(self, inputs, training, is_inference=False):
        frame_1, frame_2, inf_DPs = inputs
        #image1 = 2*(image1/255.0) - 1.0
        #image2 = 2*(image2/255.0) - 1.0

        # feature extractor -> (bs, h/8, w/8, 256)x2
        fmap_1 = self.fnet(frame_1, training=training)
        fmap_2 = self.fnet(frame_2, training=training)

        # setup correlation values
        correlation_12 = CorrBlock(fmap_1, fmap_2,
                                num_levels=self.corr_levels,
                                radius=self.corr_radius,
                                num_feature=self.feature_dim)
        if not is_inference:
            correlation_21 = CorrBlock(fmap_2, fmap_1,
                                    num_levels=self.corr_levels,
                                    radius=self.corr_radius,
                                    num_feature=self.feature_dim)

        # context network -> (bs, h/8, w/8, hdim+cdim)
        # split -> (bs, h/8, w/8, hdim), (bs, h/8, w/8, cdim)
        #hidden_context_1 = self.cnet(frame_1, training=training)        
        #hidden_1, context_1 = tf.split(hidden_context_1, [self.hidden_dim, self.context_dim], axis=-1)
        #hidden_1 = tf.tanh(hidden_1)
        #context_1 = tf.nn.relu(context_1)
        hidden_1, context_1 = self.get_hidden_context(frame_1, training=training)
        if not is_inference:
            hidden_2, context_2 = self.get_hidden_context(frame_2, training=training)

        # (bs, h/8, w/8, 2)x2, xy-indexing
        coords_origin, coords_1, coords_2, coords_hres = self.initialize_flow(frame_1)

        predictions = []
        iters = self.iters if training else self.iters_pred

        def cond(iter_num, coords_1, coords_2, hidden_1, hidden_2, predictions):
            return iter_num < iters
    
        def body(iter_num, coords_1, coords_2, hidden_1, hidden_2, predictions):
            coords_1 = tf.stop_gradient(coords_1)#apply gradient on only delta 
            # FLOW from frame1 to 2
            # (bs, h, w, 81xnum_levels)
            corr = correlation_12.retrieve(coords_1)
            flow = coords_1 - coords_origin
            # (bs, h, w, *), net: hdim, up_mask: 64x9, delta_flow: 2
            hidden_1, up_mask, delta_flow = self.update_block([hidden_1, context_1, corr, flow])
            coords_1 += delta_flow
            # upsample prediction
            flow_up_1 = self.upsample_flow(coords_1 - coords_origin, up_mask)
            frame_1_reconst_from_2, out_range_mask_12 = warp_bilinear(frame_2, coords_hres + flow_up_1)
            depth_1 = self.disparity_to_depth(flow_up_1[...,:1], inf_DPs)#only x disparity is enough
            
            coords_2 = tf.stop_gradient(coords_2)

            corr = correlation_21.retrieve(coords_2)
            flow = coords_2 - coords_origin
            hidden_2, up_mask, delta_flow = self.update_block([hidden_2, context_2, corr, flow])
            coords_2 += delta_flow
            flow_up_2 = self.upsample_flow(coords_2 - coords_origin, up_mask)
            frame_2_reconst_from_1, out_range_mask_21 = warp_bilinear(frame_1, coords_hres + flow_up_2)
            depth_2 = self.disparity_to_depth(-flow_up_2[...,:1], -inf_DPs)#minus, ok?
            
            depth_1_reconst_from_2, mask_1_reconst_from_2 = warp_bilinear(depth_2, coords_hres + flow_up_1)
            depth_2_reconst_from_1, mask_2_reconst_from_1 = warp_bilinear(depth_1, coords_hres + flow_up_2)
            
            predictions["rgb_1_re"] = predictions["rgb_1_re"].write(iter_num, frame_1_reconst_from_2)
            predictions["rgb_2_re"] = predictions["rgb_2_re"].write(iter_num, frame_2_reconst_from_1)
            predictions["depth_1"] = predictions["depth_1"].write(iter_num, depth_1)
            predictions["depth_2"] = predictions["depth_2"].write(iter_num, depth_2)
            predictions["flow_1"] = predictions["flow_1"].write(iter_num, flow_up_1)
            predictions["flow_2"] = predictions["flow_2"].write(iter_num, flow_up_2)
            predictions["depth_1_re"] = predictions["depth_1_re"].write(iter_num, depth_1_reconst_from_2)
            predictions["depth_2_re"] = predictions["depth_2_re"].write(iter_num, depth_2_reconst_from_1)
            predictions["mask_1_re"] = predictions["mask_1_re"].write(iter_num, mask_1_reconst_from_2)
            predictions["mask_2_re"] = predictions["mask_2_re"].write(iter_num, mask_2_reconst_from_1)
            
            iter_num += 1
            return [iter_num, coords_1, coords_2, hidden_1, hidden_2, predictions]
    
        
        predictions = ({"rgb_1_re": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "rgb_2_re": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "depth_1": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "depth_2": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "flow_1": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "flow_2": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "depth_1_re": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "depth_2_re": tf.TensorArray(dtype=tf.float32, size=iters, dynamic_size=False),
                            "mask_1_re": tf.TensorArray(dtype=tf.bool, size=iters, dynamic_size=False),
                            "mask_2_re": tf.TensorArray(dtype=tf.bool, size=iters, dynamic_size=False),
                                         })
        iter_num = 0
        loop_vars = [iter_num, 
                     #frame_1, frame_2, coords_origin, 
                     coords_1, coords_2, 
                     #correlation_12, correlation_21, 
                     hidden_1, hidden_2,
                     #context_1, context_2, inf_DPs, 
                     predictions]
        _, _, _, _, _, predictions = tf.while_loop(cond,body,
                                                   loop_vars,
                                                   parallel_iterations=1,
                                                   maximum_iterations=iters)
        for key in predictions.keys():
            predictions[key] = predictions[key].stack()
            
        new_predictions = []
        for i in range(iters):
            new_predictions.append({"rgb_1_re": predictions["rgb_1_re"][i],
                                   "rgb_2_re": predictions["rgb_2_re"][i],
                                   "depth_1": predictions["depth_1"][i],
                                   "depth_2": predictions["depth_2"][i],
                                   "flow_1": predictions["flow_1"][i],
                                   "flow_2": predictions["flow_2"][i],
                                   "depth_1_re": predictions["depth_1_re"][i],
                                   "depth_2_re": predictions["depth_2_re"][i],
                                   "mask_1_re": predictions["mask_1_re"][i],
                                   "mask_2_re": predictions["mask_2_re"][i],
                                   })
        
        return new_predictions


    def compile(self, optimizer, clip_norm, 
                loss, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.loss = loss

        self.custom_metrics = OrderedDict({
            'loss': tf.keras.metrics.Mean(name='loss'),
            'loss_rgb': tf.keras.metrics.Mean(name='loss_rgb'),
            'loss_census': tf.keras.metrics.Mean(name='loss_census'),
            'loss_ssim': tf.keras.metrics.Mean(name='loss_ssim'),
            'loss_smooth': tf.keras.metrics.Mean(name='loss_smooth'),
            'loss_depth_l1': tf.keras.metrics.Mean(name='loss_depth_l1'),
            'loss_iter_0': tf.keras.metrics.Mean(name='loss_iter_0'),
            'loss_iter_2': tf.keras.metrics.Mean(name='loss_iter_2'),
            'loss_iter_4': tf.keras.metrics.Mean(name='loss_iter_4'),
            'loss_iter_fin': tf.keras.metrics.Mean(name='loss_iter_fin'),
            'loss_f1': tf.keras.metrics.Mean(name='loss_f1'),
            'loss_f2': tf.keras.metrics.Mean(name='loss_f2'),
        })

    #@tf.function
    def train_step(self, data):
        self.train_step_counter.assign_add(1.0)
        #dataloaderに含める
        #image1, image2, flow, valid = data
        #image1 = tf.cast(image1, dtype=tf.float32)
        #image2 = tf.cast(image2, dtype=tf.float32)
        
        inputs, targets = data#[frame_1, frame_2], [gt_dist, car_mask]
        
        with tf.GradientTape() as tape:
            predictions = self([inputs["frame_1"], inputs["frame_2"], inputs["inf_DP"], inputs["res_rate"]], training=True)
            loss, loss_iter = self.loss(inputs["frame_1"], inputs["frame_2"], 
                                        inputs["box_mask"], inputs["box_mask_l"],
                                        targets["dist"],
                                        predictions, self.train_step_counter)
        grads = tape.gradient(loss["total"], self.trainable_weights)
        ##grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.custom_metrics['loss'].update_state(loss["total"])
        self.custom_metrics['loss_rgb'].update_state(loss["rgb"])
        self.custom_metrics['loss_census'].update_state(loss["census"])
        self.custom_metrics['loss_ssim'].update_state(loss["ssim"])
        self.custom_metrics['loss_smooth'].update_state(loss["smooth"])
        self.custom_metrics['loss_depth_l1'].update_state(loss["depth_l1"])
        self.custom_metrics['loss_iter_0'].update_state(loss_iter[0][0] + loss_iter[1][0])
        self.custom_metrics['loss_iter_2'].update_state(loss_iter[0][2] + loss_iter[1][2])
        self.custom_metrics['loss_iter_4'].update_state(loss_iter[0][4] + loss_iter[1][4])
        self.custom_metrics['loss_iter_fin'].update_state(loss_iter[0][-1] + loss_iter[1][-1])
        self.custom_metrics['loss_f1'].update_state(tf.reduce_sum(tf.stack(loss_iter[0])))
        self.custom_metrics['loss_f2'].update_state(tf.reduce_sum(tf.stack(loss_iter[1])))

        return {k: m.result() for k, m in self.custom_metrics.items()}

    def test_step(self, data):
        inputs, targets = data#[frame_1, frame_2], [gt_dist, car_mask]
        
        predictions = self([inputs["frame_1"], inputs["frame_2"], inputs["inf_DP"], inputs["res_rate"]], training=False)
        loss, loss_iter = self.loss(inputs["frame_1"], inputs["frame_2"], 
                                    inputs["box_mask"], inputs["box_mask_l"],
                                    targets["dist"],
                                    predictions)

        self.custom_metrics['loss'].update_state(loss["total"])
        self.custom_metrics['loss_rgb'].update_state(loss["rgb"])
        self.custom_metrics['loss_census'].update_state(loss["census"])
        self.custom_metrics['loss_ssim'].update_state(loss["ssim"])
        self.custom_metrics['loss_smooth'].update_state(loss["smooth"])
        self.custom_metrics['loss_depth_l1'].update_state(loss["depth_l1"])
        self.custom_metrics['loss_iter_0'].update_state(loss_iter[0][0] + loss_iter[1][0])
        self.custom_metrics['loss_iter_2'].update_state(loss_iter[0][2] + loss_iter[1][2])
        self.custom_metrics['loss_iter_4'].update_state(loss_iter[0][4] + loss_iter[1][4])
        self.custom_metrics['loss_iter_fin'].update_state(loss_iter[0][-1] + loss_iter[1][-1])
        self.custom_metrics['loss_f1'].update_state(tf.reduce_sum(tf.stack(loss_iter[0])))
        self.custom_metrics['loss_f2'].update_state(tf.reduce_sum(tf.stack(loss_iter[1])))


        return {k: m.result() for k, m in self.custom_metrics.items()}

    def predict_step(self, data):
        frame_1, frame_2, *_ = data
        predictions = self([frame_1, frame_2], training=False, is_inference=True)
        return predictions[-1]

    def reset_metrics(self):
        for k, m in self.custom_metrics.items():
            m.reset_states()


def condition(i, items):
    print(i)
    # Tensor("while/Merge:0", shape=(), dtype=int32) Tensor("while/Merge_1:0", shape=(), dtype=int32)
    return i < 10


def update(i, items):
    print(i)
    items = items.write(i, tf.constant(i,tf.int32))

    # => Tensor("while/Identity:0", shape=(), dtype=int32) Tensor("while/Identity_1:0", shape=(), dtype=int32)
    return i+1, items





    
def check(inputs):
    a, b, *_ = inputs
    print(a,b)

if __name__=="__main__":
    
    flow_21_x = [[0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,-2,-1,0]]
    
    coords_2 = tf.stack(tf.meshgrid(tf.range(5),tf.range(3)), axis=2)
    flow_21_x = tf.constant(flow_21_x, tf.float32)[tf.newaxis,:,:,tf.newaxis]
    flow_21 = tf.concat([flow_21_x, tf.zeros_like(flow_21_x)], axis=-1)
    coords_2 = tf.cast(coords_2, tf.float32)[tf.newaxis,:,:,:]
    
    print(occlusion_mask_by_translation(flow_21, coords_2)[0,:,:,0])
    
    raise Exception()
    
    SS = 0.
    SS_RET = 0.
    #items = []
    """
    items = tf.TensorArray(dtype=tf.int32, size=10, dynamic_size=False)
    
    init_val = (0, items)
    i, final_val = tf.while_loop(cond=condition, body=update, loop_vars=init_val)
    final_val = final_val.stack()
    print(final_val)

    #seq = tf.keras.Sequential([
    #        ResBlock(filters=10, strides=2, name="res_1"),
    #        ResBlock(filters=10, strides=1, name="res_2")
    #    ])
    """
    inputs = Input((64,64,3))
    x = ResBlock(filters=10, strides=2, name="res_1")(inputs)
    x = ResBlock(filters=10, strides=1, name="res_2")(x)
    model = Model(inputs, x)
    
    model = RAFT()
    
    #model.compile(
    #                       optim, 0.01,
    #                       loss_wrapper(loss_weight, gammma=0.8, batch_for_occ_mask=20000),
    #                       )
        
    #print(model.summary())
    
    
    import time
    s = time.time()

    for i in range(1):
        outs = model([tf.ones((1,256,128,3), tf.float32),tf.ones((1,256,128,3), tf.float32)], is_inference=False)
    print(time.time()-s)
    
    s = time.time()
    for i in range(4):
        outs = model([tf.ones((1,256,128,3), tf.float32),tf.ones((1,256,128,3), tf.float32)], is_inference=False)
    print(time.time()-s)

    for key in outs[0].keys():
        print(key, outs[0][key].shape)
    
    
    # test train_step
    loss_weight = {"rgb": 3.0,
                   "census":0.0,
                   "ssim": 3.0,
                   "smooth": 1.5,
                   }
    optim = tf.keras.optimizers.Adam(lr=0.01, clipnorm=0.01)    
    model.compile(
                           clip_norm=0.01,
                           optimizer = optim,
                           loss = loss_wrapper(loss_weight, gammma=0.8, batch_for_occ_mask=20000),
                           )
    
    outs = model.train_step({"frame_1":tf.ones((1,256,128,3), tf.float32),
                             "frame_2":tf.ones((1,256,128,3), tf.float32)})
    
    for key in outs.keys():
        print(key, outs[key].shape)
    #loss,_ = loss_stereo_depth(tf.ones((1,64,64,3), tf.float32),tf.ones((1,64,64,3), tf.float32),outs)
    #for key in loss.keys():
    #    print(loss)
    #model.build(input_shape=[(None, 64, 64, 3), (None, 64, 64, 3)])
    #print(model.summary())
    
"""

def build_feaure_encoder():
    


d_inputs_0 = rgb_0
d_inputs_1 = rgb_1

depth_0 = depth_net(d_inputs_0) or uniform depth
depth_1 = depth_net(d_inputs_1)

feature_0 = feature_net(f_inputs_0)
feature_1 = feature_net(f_inputs_1)
cor_pyramid_0_1 = correlation_pyramid(feature_0, feature_1)
cor_pyramid_1_0 = transpose(correlation_0_1)


hidden_0, context_0 = context_net(c_inputs_0)
hidden_1, context_1 = context_net(c_inputs_1)

warp_idx_0to1 = get_warp_idx(depth_0, inf_DP, (coords, pose_matrix) )
warp_idx_1to0 = get_warp_idx(depth_1, inv_inf_DP, (coords, inv_pose_matrix) )


update_block(warp_idx, cor_pyramid, ), update_operator(warp_idx, cor_pyramid, )


def update_operator(warp_idx, cor_pyramid, hidden, context)

    corr_look_up(warp_idx, cor_mat)


    return upd_warp_idx, upd_hidden, 

# loss
rgb_0_warp_1 = warp(rgb_0, warp_idx_0to1)
rgb_loss = rgb_consistent(rgb_0_warp_1, rgb_1, occulusion_information)

"""

