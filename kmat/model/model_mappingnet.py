# -*- coding: utf-8 -*-
"""
@author: kmat
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

def se(x_in, layer_n,rate, name):
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

def intrinsics_head(inputs, img_height, img_width, maximum_focal_rate=10.0, minimum_focal_rate=1.0):
    """
    --- NOT USE ----
    
    
    Adds a head the preficts camera intrinsics.
    Args:
      bottleneck: A tf.Tensor of shape [B, 1, 1, C], typically the bottlenech
        features of a netrowk.
      image_height: A scalar tf.Tensor or an python scalar, the image height in
        pixels.
      image_width: A scalar tf.Tensor or an python scalar, the image width in
        pixels.
    image_height and image_width are used to provide the right scale for the focal
    length and the offest parameters.
    Returns:
      a tf.Tensor of shape [B, 3, 3], and type float32, where the 3x3 part is the
      intrinsic matrix: (fx, 0, x0), (0, fy, y0), (0, 0, 1).
    """
    bottleneck = GlobalAveragePooling2D(name="gap")(inputs)
    maximum_focal_rate = tf.constant(maximum_focal_rate, tf.float32)
    minimum_focal_rate = tf.constant(minimum_focal_rate, tf.float32)
    #  with tf.variable_scope('CameraIntrinsics'):
    #    # Since the focal lengths in pixels tend to be in the order of magnitude of
    #    # the image width and height, we multiply the network prediction by them.
    #focal_lengths = Dense(2, activation="softplus", name="preout_focal_length")(inputs)
    focal_lengths = Dense(2, activation="sigmoid", name="preout_focal_length")(bottleneck)
    focal_lengths = Lambda(lambda x: (x * (maximum_focal_rate - minimum_focal_rate) + minimum_focal_rate) * tf.convert_to_tensor([[img_width, img_height]], tf.float32), name="out_focal_length")(focal_lengths)    
    
    #center_offsets = Dense(2, activation="sigmoid", name="preout_center_offsets")(bottleneck)
    #center would be within the image
    #center_offsets = Lambda(lambda x: x*tf.convert_to_tensor([[img_width, img_height]], tf.float32), name="out_center_offsets")(center_offsets)
    center_offsets = Lambda(lambda x: 0.5*tf.ones_like(x)*tf.convert_to_tensor([[img_width, img_height]], tf.float32), name="out_center_offsets")(focal_lengths)
    def make_matrix(inputs):
        focal_lengths, center_offsets = inputs
        foci = tf.linalg.diag(focal_lengths)
        intrinsic_mat = tf.concat([foci, tf.expand_dims(center_offsets, -1)], axis=2)
        batch_size = tf.shape(focal_lengths)[0]
        last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
        intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
        return intrinsic_mat
    intrinsic_mat = Lambda(make_matrix, name="make_intrinsic")([focal_lengths, center_offsets])
    intrinsic_mat_inv = invert_intrinsics_matrix(intrinsic_mat)
    return intrinsic_mat_inv#, intrinsic_mat

def camera_ratios_head(bottleneck):
    
    #ratios = Dense(2, activation="relu", name="out_xy_ratio")(bottleneck)
    #TODO これシングルバリューにしないと、奥行きをこれで調整してしまいそう。
    #とりあえずそれはそれで試してみるか…。
    #
    k_init = tf.keras.initializers.RandomNormal(stddev=0.005)
    scale_ratio = Dense(1, activation="tanh", name="out_scale_ratio",
                        kernel_initializer=k_init,#'zeros', 
                        bias_initializer='zeros',
                        )(bottleneck)
    ##scale_ratio = Lambda(lambda x: tf.math.exp(4.5*x))(scale_ratio)
    scale_ratio = Lambda(lambda x: tf.math.exp(3.0*x))(scale_ratio)
    aspect_ratio = Dense(1, activation="tanh", 
                         kernel_initializer=k_init,#'zeros', 
                         bias_initializer='zeros',
                         name="out_aspect_ratio")(bottleneck)
    #aspect_ratio = Lambda(lambda x: tf.math.exp(2.0*(x+0.25)))(aspect_ratio)
    ##aspect_ratio = Lambda(lambda x: tf.math.exp(1.5*(x+0.1)))(aspect_ratio)
    aspect_ratio = Lambda(lambda x: tf.math.exp(2.0*(x+0.2)))(aspect_ratio)
    
    return scale_ratio, aspect_ratio


def invert_intrinsics_matrix(intrinsics_mat):
    """Inverts an intrinsics matrix.
    Args:
      intrinsics_mat: A tensor of shape [.... 3, 3], representing an intrinsics
        matrix `(in the last two dimensions).
    Returns:
      A tensor of the same shape containing the inverse of intrinsics_mat
    """
    intrinsics_mat = tf.convert_to_tensor(intrinsics_mat)
    intrinsics_mat_cols = tf.unstack(intrinsics_mat, axis=-1)
    if len(intrinsics_mat_cols) != 3:
        raise ValueError('The last dimension of intrinsics_mat should be 3, not '
                       '%d.' % len(intrinsics_mat_cols))

    fx, _, _ = tf.unstack(intrinsics_mat_cols[0], axis=-1)
    _, fy, _ = tf.unstack(intrinsics_mat_cols[1], axis=-1)
    x0, y0, _ = tf.unstack(intrinsics_mat_cols[2], axis=-1)

    zeros = tf.zeros_like(fx)
    ones = tf.ones_like(fx)

    row1 = tf.stack([1.0 / fx, zeros, zeros], axis=-1)
    row2 = tf.stack([zeros, 1.0 / fy, zeros], axis=-1)
    row3 = tf.stack([-x0 / fx, -y0 / fy, ones], axis=-1)

    return tf.stack([row1, row2, row3], axis=-1)


def xy_conbine_layer(inputs):
    xy_offsets, scale_ratio, aspect_ratio = inputs
    height, width = tf.unstack(tf.shape(xy_offsets))[1:3]
    grid = tf.stack(tf.meshgrid(tf.range(width), tf.range(height)), axis=0)
    grid = tf.transpose(tf.cast(grid, tf.float32), [1,2,0])/tf.cast([[[width,height]]],tf.float32)
    grid_x = grid[:,:,:1]
    grid_y = 1.0 - grid[:,:,1:2]#inverse y
    grid = tf.concat([grid_x, grid_y], axis=-1)
    
    grid = grid[tf.newaxis,:,:,:]
    scale = scale_ratio[:,tf.newaxis,tf.newaxis,:] * tf.concat([tf.ones_like(aspect_ratio), aspect_ratio], axis=-1)[:,tf.newaxis,tf.newaxis,:]
    xy = scale * (grid + xy_offsets)
    return xy


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
    images, boxes, met_size = inputs
    ##pad_size = 150//4
    img_height, img_width = tf.unstack(tf.shape(images))[1:3]
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
    average_size = met_size[:,:,tf.newaxis]#tf.reduce_mean(tf.math.sqrt(width*height), axis=1, keepdims=True)

    #if over range, zero padding automatically.
    if wide_mode:
        lr_length = 2
        t_length = 2
        b_length = 4.0        
    else:
        lr_length = 1.5#*2
        t_length = 1.5#*2
        b_length = 3.0#*2
    h_length = t_length + b_length
    w_length = lr_length * 2

    #if crop_size[0]!=(crop_size[1]*h_length/w_length):
    #    raise Exception("check aspect ratio is {}:{}".format(h_length, w_length))
    
    ts = (cy - average_size*t_length)
    ls = (cx - average_size*lr_length)
    bs = (cy + average_size*b_length)
    rs = (cx + average_size*lr_length)
    
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
        top_aft_resize = ((top - cy)/(h_length*average_size)) + t_length/h_length 
        left_aft_resize = ((left - cx)/(w_length*average_size)) + lr_length/w_length
        bottom_aft_resize = ((bottom - cy)/(h_length*average_size)) + t_length/h_length
        right_aft_resize = ((right - cx)/(w_length*average_size)) + lr_length/w_length
        
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
    


def points2points_fitting(targets, sources, num_iter=6, l2_reg=0.1):
    
    def get_transmatrix(k, rz, tx, ty):
        """
        k : log(zoom ratio).
        rz : rotation.
        tx : x offset.
        ty : z offset
        shape [batch]
        
        returns:
            transmatrix with shape [batch, 3, 3]
        """
        exp_k = tf.math.exp(k)
        sin = tf.math.sin(rz)
        cos = tf.math.cos(rz)
        mat = tf.stack([[exp_k*cos, -exp_k*sin, exp_k*tx],
                        [exp_k*sin, exp_k*cos, exp_k*ty],
                        [tf.zeros_like(k), tf.zeros_like(k), tf.ones_like(k)]])
        mat = tf.transpose(mat, [2,0,1])
        return mat
        
    def transform_points(transmatrix, points):
        x, y = tf.split(points, 2, axis=-1)
        xyones = tf.concat([x,y,tf.ones_like(x)], axis=-1)
        trans_points = tf.matmul(xyones, transmatrix, transpose_b=True)[...,:2]
        return trans_points
    
    def get_derivative_at(k, rz, tx, ty, points):
        dev = 1e-5
        original = transform_points(get_transmatrix(k, rz, tx, ty), points)
        dxy_dk = (transform_points(get_transmatrix(k+dev, rz, tx, ty), points) - original) / dev
        dxy_drz = (transform_points(get_transmatrix(k, rz+dev, tx, ty), points) - original) / dev
        dxy_dtx = (transform_points(get_transmatrix(k, rz, tx+dev, ty), points) - original) / dev
        dxy_dty = (transform_points(get_transmatrix(k, rz, tx, ty+dev), points) - original) / dev
        return original, dxy_dk, dxy_drz, dxy_dtx, dxy_dty
    
    def normal_equation(X, Y, gn_rate=0.9, gd_rate=0.1):
        #not in use
        #Gaussian Neuton combind with Gradient descent
        XtX = tf.matmul(X, X, transpose_a=True)
        #XtX_inv = np.linalg.inv(XtX)
        #XtX_inv = 1.0*XtX_inv + 0.5*np.eye(len(XtX))
        #results = np.dot(XtX_inv, np.dot(X.T, Y))
        
        results_neuton = tf.linalg.solve(XtX, tf.matmul(X, Y, transpose_a=True))
        results_gaussian = tf.matmul(tf.tile(tf.eye(XtX.shape[1])[tf.newaxis,...], [XtX.shape[0],1,1]), tf.matmul(X, Y, transpose_a=True))
        results = gn_rate*results_neuton + gd_rate*results_gaussian
        
        return results
    
    # initial_values
    batch, num_points = tf.unstack(tf.shape(targets))[:2]
    k = 0.0 * tf.ones((batch), tf.float32)#zoom ratio = exp(k)
    rz = 0.0 * tf.ones((batch), tf.float32)
    tx = 0.0 * tf.ones((batch), tf.float32)
    ty = 0.0 * tf.ones((batch), tf.float32)
    
    source_origin = sources#tf.stop_gradient(sources)
    for i in range(num_iter):
        currents, dxy_dk, dxy_rz, dxy_dtx, dxy_dty = get_derivative_at(k, rz, tx, ty, source_origin)
        b = tf.reshape(targets-currents, [batch, num_points*2, 1])#xy flatten
        a = tf.stack([dxy_dk, dxy_rz, dxy_dtx, dxy_dty], axis=-1)
        a = tf.reshape(a, [batch, num_points*2, 4])
        updates = tf.linalg.lstsq(a, b, l2_regularizer=l2_reg, fast=True)#batch, 4, 1
        k = k + updates[:,0,0]
        rz = rz + updates[:,1,0]
        tx = tx + updates[:,2,0]
        ty = ty + updates[:,3,0]
    trans_matrix = get_transmatrix(k, rz, tx, ty)
    trans_sources = transform_points(trans_matrix, sources)
    return trans_sources, trans_matrix, k, rz, tx, ty
    
def reduce_z_axis(xyz_points):
    """
    for depth and pointcloud estimation,
    reduce z axis and project players on 2D map
    (Not Adopted)
    """
    mean_pred = tf.reduce_mean(xyz_points, axis=1, keepdims=True)
    xyz_points = xyz_points - mean_pred
    xtx = tf.matmul(xyz_points, xyz_points, transpose_a=True)
    s,u,v = tf.linalg.svd(xtx)
    
    xy_points = tf.matmul(xyz_points, u[..., :2])
    normal_vec = tf.linalg.cross(v[...,0], v[...,1]) # first 2 axis are the major axis
    depth_direction = (0.5 + 0.5*tf.cast(tf.math.sign(normal_vec[...,2]), tf.float32))[:,tf.newaxis,tf.newaxis]
    xy_points = tf.concat([(1-depth_direction)*xy_points[...,0:1] + depth_direction*xy_points[..., 1:2], 
                           (1-depth_direction)*xy_points[...,1:2] + depth_direction*xy_points[..., 0:1]], axis=-1)
    
    # check z distribution
    normal_vec = v[...,2] # 3rd axis is the axis of smallest distribution (normal vector)
    xnv = tf.einsum('bnc,bc->bn', xyz_points, normal_vec)
    mse_z = tf.reduce_mean((xnv)**2.0, axis=-1, keepdims=True)
    return xy_points, mse_z
    

def xyz_fitting_layer(inputs, 
                      num_iter=80, 
                      l2_reg=0.5,
                      use_z_reduction=False,
                      mode="F1"):
    """
    
    """
    
    xy_gt, xyz_pred = inputs
    batch, num, ch = tf.unstack(tf.shape(xyz_pred))
    if use_z_reduction:
        xy_pred, mse_z = reduce_z_axis(xyz_pred)
    else:
        xy_pred = xyz_pred
        mse_z = tf.zeros((batch,1), tf.float32)
    transformed_pred, trans_matrix, zoom_dev, rot_dev, tx_dev, ty_dev = points2points_fitting(xy_gt, xy_pred, num_iter=num_iter, l2_reg=l2_reg)
    zoom_dev_abs = tf.math.abs(zoom_dev)

    if mode=="F0":
        scale_mode = "max"
        nonlinear_loss = False
        loss_clipping = True
        clip_value = 0.01
        add_relative_loss = False
        relative_loss_weight = 0.2
    elif mode=="F1":
        scale_mode = "average"
        nonlinear_loss = False
        loss_clipping = False
        clip_value = 0.01
        add_relative_loss = False
        relative_loss_weight = 0.2
    elif mode=="F2":
        scale_mode = "average"
        nonlinear_loss = True
        loss_clipping = False
        clip_value = 0.01
        add_relative_loss = False
        relative_loss_weight = 0.2
    elif mode=="F3":
        scale_mode = "average"
        nonlinear_loss = False
        loss_clipping = False
        clip_value = 0.01
        add_relative_loss = True
        relative_loss_weight = 0.025
    else:
        raise Exception("choose run type from [F0, F1, F2, F3]")

    if scale_mode == "max":    
        ref_scale = tf.reduce_max(xy_gt, axis=1, keepdims=True) - tf.reduce_min(xy_gt, axis=1, keepdims=True)
        ref_scale = tf.reduce_max(ref_scale, axis=-1, keepdims=True)+1e-7
    elif scale_mode == "average":
        xy_gt_dist_mat = tf.reduce_sum((xy_gt[:,:,tf.newaxis,:] - xy_gt[:,tf.newaxis,:,:])**2, axis=-1)
        ref_scale = tf.sqrt(tf.reduce_mean(xy_gt_dist_mat, axis=[1,2], keepdims=True)+1e-10)
    xy_gt_rescaled = xy_gt / ref_scale
    pred_rescaled = transformed_pred / ref_scale
    xy_error = (xy_gt_rescaled - pred_rescaled)**2
    
    if nonlinear_loss:
        xy_error = tf.reduce_sum(xy_error, axis=-1)
        xy_error = tf.where(xy_error<1.0, xy_error, 2.0*(xy_error/tf.sqrt(xy_error+1e-10))-1.0)
    else:
        xy_error = tf.reduce_mean(xy_error, axis=-1)#should be sum, mistake
    mse_xy = tf.reduce_mean(xy_error, axis=1)
    
    if add_relative_loss:
        xy_gt_dist_mat = tf.reduce_sum((xy_gt_rescaled[:,:,tf.newaxis,:] - xy_gt_rescaled[:,tf.newaxis,:,:])**2, axis=-1)
        pred_dist_mat = tf.reduce_sum((pred_rescaled[:,:,tf.newaxis,:] - pred_rescaled[:,tf.newaxis,:,:])**2, axis=-1)
        mask = tf.cast(tf.logical_and(xy_gt_dist_mat>0., xy_gt_dist_mat<1.), tf.float32)
        relative_loss = tf.reduce_sum(pred_dist_mat * mask, axis=[1,2])/(tf.reduce_sum(mask, axis=[1,2])+1e-7)
        mse_xy = mse_xy + (relative_loss_weight * relative_loss)
    
    if loss_clipping:
        mse_xy = tf.minimum(mse_xy, clip_value)
    
    return transformed_pred, mse_z, mse_xy, zoom_dev_abs, zoom_dev, rot_dev, tx_dev, ty_dev
    

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
    boxes_mask = tf.einsum('bnh,bnw->bhw', 
                tf.cast(height_inside, tf.float32), 
                tf.cast(width_inside, tf.float32))
    if only_overlap:#if>=2, overlap
        boxes_mask = tf.cast((boxes_mask>1.0), tf.float32)
    
    return tf.concat([imgs, boxes_mask[...,tf.newaxis]], axis=-1)

def get_dev_overlap(inputs, multi_mask=True):
    all_box_mask = inputs[...,-2:-1]
    crop_box_mask = inputs[...,-1:]
    other_box_mask  = all_box_mask - crop_box_mask
    other_box_mask = tf.where(other_box_mask<0.99, 0.0, 1.0)
    if multi_mask:
        return tf.concat([inputs[..., :-2], crop_box_mask, other_box_mask], axis=-1)
    else:
        return tf.concat([inputs[..., :-2], crop_box_mask], axis=-1)  

def add_scale_aspect_ch(inputs):
    x, scale_ratio, aspect_ratio = inputs
    batch_num, height, width = tf.unstack(tf.shape(x))[:3]
    scale_aspect = tf.concat([scale_ratio, aspect_ratio], axis=-1)
    scale_aspect = tf.stop_gradient(scale_aspect)
    scale_aspect = tf.tile(scale_aspect[:,tf.newaxis,tf.newaxis,:], [1,height,width,1])
    return tf.concat([x, scale_aspect], axis=-1)

def gaussian_average(inputs, size=[16,16], sigma=6.0):
    #batchxnum_boxes, h(16), w(16), ch
    h_shift = tf.range(size[0], dtype=tf.float32) + 0.5 - tf.cast(size[0]/2, tf.float32)
    w_shift = tf.range(size[1], dtype=tf.float32) + 0.5 - tf.cast(size[1]/2, tf.float32)
    sq_dist = h_shift[:,tf.newaxis]**2 + w_shift[tf.newaxis,:]**2
    gaussian_weight = tf.math.exp(-sq_dist/(2*(sigma**2)))
    total_weights = tf.reduce_sum(gaussian_weight)
    #outputs = tf.reduce_sum(inputs * gaussian_weight[tf.newaxis,tf.newaxis,:,:,tf.newaxis], axis=[2,3])/total_weights
    outputs = tf.reduce_sum(inputs * gaussian_weight[tf.newaxis,:,:,tf.newaxis], axis=[1,2])/total_weights
    return outputs

def attention_average(inputs):
    #batchxnum_boxes, h(16), w(16), ch
    img, attention = inputs
    #attention = tf.tile(attention,[1,1,1,2])
    total_weights = tf.reduce_sum(attention, axis=[1,2])
    outputs = tf.reduce_sum(img * attention, axis=[1,2])/(total_weights+1e-7)
    return outputs    
    

def similarity_head_simple(x, temperature=5., use_dropout=False):
    def get_matrix(vec):
        batch, num_box, _ = tf.unstack(tf.shape(vec))
        normal_vec = tf.linalg.normalize(vec, axis=-1)[0]
        vec_1 = tf.tile(normal_vec[:,tf.newaxis,:,:], [1,num_box,1,1])
        vec_2 = tf.tile(normal_vec[:,:,tf.newaxis,:], [1,1,num_box,1])
        simmat = tf.reduce_sum(vec_1 * vec_2, axis=-1, keepdims=True)
        #einsum may better
        return tf.math.sigmoid(temperature * simmat)
    if use_dropout:
        x = Dropout(0.3, noise_shape=(None, 1, None), name='simmat_drop')(x)
    x = Lambda(get_matrix, name="team_similarity")(x)
    return x



def similarity_head_arcface(x, epoch_dependent_value=1.0, min_temperature=20., max_temperature=40., min_margin=0.25, max_margin=0.4, temp_normal=5.0, use_dropout=False, is_train=False):
    def get_matrix(inputs):
        #margin = min_margin
        margin = min_margin + (max_margin - min_margin)*epoch_dependent_value
        temperature = min_temperature + (max_temperature - min_temperature)*epoch_dependent_value
        
        if is_train:
            vec, labels_matrix = inputs
        else:
            vec = inputs
        batch, num_box, _ = tf.unstack(tf.shape(vec))
        normal_vec = tf.linalg.normalize(vec, axis=-1)[0]
        vec_1 = tf.tile(normal_vec[:,:,tf.newaxis,:], [1,1,num_box,1])
        vec_2 = tf.tile(normal_vec[:,tf.newaxis,:,:], [1,num_box,1,1])
        cos_sim = tf.reduce_sum(vec_1 * vec_2, axis=-1, keepdims=True)

        if is_train:
            theta = tf.math.acos(tf.clip_by_value(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7))
            # for positive
            delay_mask = tf.math.logical_and((theta<(math.pi - margin)), (labels_matrix>0.5))
            theta = tf.where(delay_mask, theta + margin, theta)
            # for negative
            delay_mask = tf.math.logical_and((theta>(0. + margin)), (labels_matrix<0.5))
            theta = tf.where(delay_mask, theta - margin, theta)
            
            normal_predictions = tf.math.sigmoid(temp_normal * cos_sim)
            delay_predictions = tf.math.sigmoid(temperature * tf.math.cos(theta))
            predictions = tf.concat([normal_predictions, delay_predictions], axis=-1)
        else:
            predictions = tf.math.sigmoid(temp_normal * cos_sim)
        #einsum may better
        return predictions
    if use_dropout:
        x = Dropout(0.3, noise_shape=(None, 1, None), name='simmat_drop')(x)
    x = Lambda(get_matrix, name="team_similarity")(x)
    return x

def random_flip_layer(inputs, is_train=False):
    if is_train:
        outputs = tf.image.random_flip_up_down(tf.image.random_flip_left_right(inputs))
    else:
        outputs = inputs
    return outputs
 
def build_model_map_previous_competition(weight_file):
    
    input_shape = (512, 896, 3)
    output_shape = (128, 224)
    minimum_stride = int(input_shape[0] // output_shape[0])
    t_model, model, custom_losses, custom_loss_weights, custom_metrics = build_model_map(input_shape=input_shape, 
                                                                  minimum_stride=minimum_stride, 
                                                                  epoch_dependent_value=1,
                                                                  is_train=False,
                                                                  backbone="effv2s",
                                                                  from_scratch=True)
    t_model.load_weights(weight_file)
    t_model.trainable = False
    model.trainable = False
    return model
 
def build_model_map(input_shape=(256,256,3),
             backbone="effv2s", 
             minimum_stride=1, max_stride = 64,
             epoch_dependent_value=1.0,
             is_train=True,
             num_boxes = None,
             loss_type="F1",
             from_scratch=False):
    """
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    input_rgb = Input(input_shape, name="input_rgb")#256,256,3
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    enc_in = input_rgb
    
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])
     
    

    use_coord_conv = True

    # global mapping branch
    bottleneck = GlobalAveragePooling2D(name="gap_bottleneck")(x)

    met_sizes = Input(shape=[1,], name="met_sizes")
    met_sizes_rescale = Lambda(lambda x: x*10.0, name="rescale_met_size")(met_sizes)
    
    bottleneck = Concatenate()([bottleneck, met_sizes_rescale])
    scale_ratio, aspect_ratio = camera_ratios_head(bottleneck)
    
    if use_coord_conv:
        x = Lambda(add_coords, name="add_coords")(x)
    x = Lambda(add_scale_aspect_ch, name="add_scale_aspect")([x, scale_ratio, aspect_ratio])
        
    outs = decoder(x, skip_connections, use_batchnorm=True, 
                   num_channels=32, max_stride=max_stride, minimum_stride=minimum_stride)

    x = outs[-1]
    xy_offsets = Conv2D(2, activation="sigmoid", kernel_size=3, strides=1, 
                        padding="same", 
                        name="xy_offsets",)(x)
    xy_offsets = Lambda(lambda x: x*epoch_dependent_value, name="epoch_dependent")(xy_offsets)
    xy = Lambda(xy_conbine_layer, name="xy_conbine")([xy_offsets, scale_ratio, aspect_ratio])
    xy_all = xy
 
    mode_list = ["simple_gap", "met_attention", "player_attention"]
    mode = mode_list[2]
    if mode=="player_attention":
        xy = Lambda(larger_crop_resize_layer, name="crop_bbox",
                   arguments={"num_ch": 2, "crop_size": [24,16], "add_crop_mask": False})([xy, input_boxes, met_sizes]) 
        rgb_features = Conv2D(16, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(x)
        feature_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, name="add_box_mask")([rgb_features, input_boxes])
        feature_w_mask = Lambda(larger_crop_resize_layer, name="crop_bbox_attention",
                   arguments={"num_ch": 17, "crop_size": [24,16], "add_crop_mask": True})([feature_w_mask, input_boxes, met_sizes]) 
        feature_w_mask = Lambda(get_dev_overlap,
                                arguments={"multi_mask": True},
                                name="dev_mask")(feature_w_mask)
        attention = cbr(feature_w_mask, 32, kernel=7, stride=1, name="attention_cbr1")
        attention = cbr(attention, 32, kernel=7, stride=1, name="attention_cbr2")
        attention = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="xy_attention")(attention)
        xy_points = Lambda(attention_average, name="xy_attention_average")([xy, attention])
        
    elif mode=="met_attention":
        xy = Lambda(crop_resize_layer, name="crop_bbox",
                   arguments={"num_ch": 2})([xy, input_boxes])        
        
        # get attention considering overlap between bounding boxes
        feature_w_mask = Lambda(add_bbox_img, name="add_box_mask")([x, input_boxes])
        feature_w_mask = Conv2D(16, activation="relu", kernel_size=3, strides=1, padding="same", 
                          name="rgb_features")(feature_w_mask)
        feature_w_mask = Lambda(crop_resize_layer, name="crop_bbox_attention",
                                 arguments={"num_ch": 16})([feature_w_mask, input_boxes])
        attention = cbr(feature_w_mask, 32, kernel=7, stride=1, name="attention_cbr")
        attention = Conv2D(1, activation="sigmoid", kernel_size=7, strides=1, padding="same", 
                           name="xy_attention")(attention)
        
        xy_points = Lambda(attention_average, name="xy_attention_average")([xy, attention])
    elif mode=="simple_gap":
        xy = Lambda(crop_resize_layer, name="crop_bbox",
                   arguments={"num_ch": 2})([xy, input_boxes])
    
        use_gaussian_average = True
    
        if use_gaussian_average:
            xy_points = Lambda(gaussian_average, name="gaussian_aft_crop")(xy)
        else:
            xy_points = GlobalAveragePooling2D(name="gap_aft_crop")(xy)
        
    xy_points = Lambda(lambda x: tf.reshape(x[0], [tf.shape(x[1])[0],-1,2]), name= "xy_points")([xy_points, input_boxes])
    #move to center
    xy_selected = xy_points
    xy_points = Lambda(lambda x: x - tf.reduce_mean(x, axis=1, keepdims=True), name= "xy_adjust_center")(xy_points)
    
    
    #if is_train:
    gt_points = Input(shape=[num_boxes,2], name="gt_points")
    xy_points_trans, z_error, xy_error, zoom_dev_abs, zoom_dev, rot_dev, tx_dev, ty_dev = Lambda(xyz_fitting_layer, 
                                                                                                 arguments={"mode": loss_type},
                                                                                                 name="xyz_error")([gt_points, xy_points])
    
    #just rename
    z_error = Lambda(lambda x: x, name= "z_error")(z_error)
    xy_error = Lambda(lambda x: x, name= "xy_error")(xy_error)
    zoom_dev_abs = Lambda(lambda x: x, name= "zoom_dev_abs")(zoom_dev_abs)
    
    t_inputs = [input_rgb, input_boxes, met_sizes, gt_points]
    t_outputs = [z_error, xy_error, zoom_dev_abs]            

    inputs = [input_rgb, input_boxes, met_sizes]
    xy_all = Lambda(lambda xy: xy[0] - tf.reduce_mean(xy[1], axis=1, keepdims=True)[:,tf.newaxis,:,:], name= "xy_image")([xy_all, xy_selected])
    outputs = [xy_points, xy_all]
    
    losses = {"z_error": weighted_dummy_loss,
              "xy_error": weighted_dummy_loss,
              "zoom_dev_abs": weighted_dummy_loss}
    loss_weights = {"z_error": 1e-4,
                    "xy_error": 100.0*4,
                    "zoom_dev_abs": 0.1*4}
    metrics = {}
    
    t_model = Model(t_inputs, t_outputs)
    model = Model(inputs, outputs)
    
    return t_model, model, losses, loss_weights, metrics
        

def points_projection(xy_image, transmatrix, xy_points):
    """    
    xy_image : [height, width, xy(2)].
    transmatrix : [3, 3]
    xy_points : [num_points, 2]

    Returns:
        Distance from each points
        temporary [height, width, num_points]

    """
    xyones_image = tf.pad(xy_image, [[0,0],[0,0],[0,1]], mode='CONSTANT', constant_values=1.0)
    trans_image = tf.matmul(xyones_image, transmatrix, transpose_b=True)[...,:2]
    dist_images = tf.norm(trans_image[:,:,tf.newaxis,:]-xy_points[tf.newaxis,tf.newaxis,:,:], axis=-1)
    min_dist_image = tf.reduce_min(dist_images, axis=2)
    return dist_images, min_dist_image

def build_model_team(input_shape=(96,64,3),
             backbone="effv2s",
             epoch_dependent_value = 1.0,
             is_train=True,
             num_boxes = None,
             return_only_vector=False,
             from_scratch=False):
    """    
    model inputs:
        - normalized rgb(d)
        - boxes(normalized coordinates to show box location. top,left,bottom,right)
    """
    
    input_rgb = Input([None, None, 3], name="input_rgb")#256,256,3
    input_boxes = Input(shape=[num_boxes,4], name="input_boxes")
    met_sizes = Input(shape=[1,], name="met_sizes")
    
    img_w_mask = Lambda(add_bbox_img, arguments={"only_overlap": False}, name="add_box_mask_team")([input_rgb, input_boxes])
    player_imgs = Lambda(larger_crop_resize_layer, 
                         name="crop_bbox_highresolution",
                         arguments={"num_ch": input_shape[2]+1, 
                                    "crop_size": [input_shape[0],input_shape[1]], 
                                    "add_crop_mask": True,
                                    "wide_mode":False,
                                    })([img_w_mask, input_boxes, met_sizes])
    player_imgs_w_mask = Lambda(get_dev_overlap, name="dev_mask_team")(player_imgs)
    
    player_imgs_w_mask = Lambda(random_flip_layer, arguments={"is_train": is_train},  name="random_flip")(player_imgs_w_mask)
    enc_in = player_imgs_w_mask
    
    model_names = {"effv2s":"s", "effv2m":"m", "effv2l":"l", "effv2xl":"xl"}
    if backbone not in model_names.keys():
        raise Exception("check backbone name")
    x, skip_connections = effv2_encoder(enc_in, is_train, from_scratch, model_name = model_names[backbone])
            
        
    x = GlobalAveragePooling2D(name="team_gap")(x)
    x = Dense(128, activation="linear", name="dense128")(x)
    
    vec = Lambda(lambda x: tf.reshape(x[0], [tf.shape(x[1])[0],tf.shape(x[1])[1],128]), name= "vec")([x, input_boxes])
    if return_only_vector:
        model = Model([input_rgb, input_boxes], vec)
        return model
        
    if is_train:
        input_labels = Input(shape=[num_boxes,num_boxes,1], name="label")
        sim_mat = similarity_head_arcface([vec, input_labels], 
                                          epoch_dependent_value=epoch_dependent_value,
                                          use_dropout=False, 
                                          is_train=is_train)
        inputs = [input_rgb, input_boxes, met_sizes, input_labels]
        outputs = [sim_mat]
    else:
        sim_mat = similarity_head_arcface(vec, use_dropout=False, is_train=is_train)
        normal_vec = Lambda(lambda x: tf.linalg.normalize(x, axis=-1)[0])(vec)
        inputs = [input_rgb, input_boxes, met_sizes]
        outputs = [sim_mat, normal_vec]
    #outputs = [sim_mat]# + [TEMP_OUT]
    losses = {"team_similarity": team_similarity_loss}
    loss_weights = {"team_similarity": 1.}
    metrics = {"team_similarity": team_similarity_acc}
    model = Model(inputs, outputs)
    return model, losses, loss_weights, metrics    


def team_similarity_loss(y_true, y_pred):
    """
    if similar -> 1
    else 0
    """
    y_pred = y_pred[...,-1:]
    #binary_ce
    mask_eye = 1.0 - tf.eye(tf.shape(y_pred)[1], dtype=tf.float32)[tf.newaxis,...,tf.newaxis]
    mask = y_true[...,1:] * mask_eye
    y_true = tf.clip_by_value(y_true[...,:1],1e-7,1.0-1e-7)
    y_pred = tf.clip_by_value(y_pred,1e-7,1.0-1e-7)
    loss = - (y_true*tf.math.log(y_pred) + (1.-y_true)*tf.math.log(1.-y_pred))
    return tf.reduce_sum(loss*mask)/tf.reduce_sum(mask)

def team_similarity_acc(y_true, y_pred):
    """
    if similar -> 1
    else 0
    """
    y_pred = y_pred[...,:1]

    #binary_ce
    mask_eye = 1.0 - tf.eye(tf.shape(y_pred)[1], dtype=tf.float32)[tf.newaxis,...,tf.newaxis]
    mask = y_true[...,1:] * mask_eye
    y_true = y_true[...,:1]>0.5#tf.clip_by_value(y_true[...,:1],1e-7,1.0-1e-7)
    y_pred = y_pred>0.5#tf.clip_by_value(y_pred,1e-7,1.0-1e-7)
    correct = tf.cast(y_true==y_pred, tf.float32)
    return tf.reduce_sum(correct*mask)/tf.reduce_sum(mask)

def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred) 

def weighted_dummy_loss(y_true, y_pred):
    batch = tf.shape(y_true)[0]
    weight = tf.reshape(y_true, [batch,])
    y_pred = tf.reshape(y_pred, [batch,])
    return tf.reduce_sum(y_pred*weight)/tf.reduce_sum(weight) 






    
