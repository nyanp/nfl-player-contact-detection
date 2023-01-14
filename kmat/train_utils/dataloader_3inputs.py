# -*- coding: utf-8 -*-
"""
@author: kmat
dataloader in training phase
"""
import os
import glob
import json

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

AUTO = tf.data.experimental.AUTOTUNE

LOAD_FLOW = False            

####
#### --- COL MODEL ---
####

def chain_list(inputs):
    outputs = []
    for inp_list in inputs:
        outputs += inp_list
    return outputs

def split_to_list(inputs, sizes):
    outputs = []
    start = 0
    for size in sizes:
        end = start + size
        outputs += [inputs[start:end]]
        start = end
    return outputs
    
def load_dataset(path_list,
                 num_max=-1, 
                 frame_interval=1, 
                 raft_model=False,
                 gcn_model=False):
    dataset = []
    for path in path_list:
        if raft_model:
            data = load_data_raft(path,
                          num_max=num_max, 
                          frame_interval=frame_interval, 
                          )
        elif gcn_model:
            data = load_data_gcn(path)
        else:
            data = load_data_5frames(path,
                          num_max=num_max, 
                          frame_interval=frame_interval, 
                          )
        dataset += data
    return dataset

def load_data_3frames(path,
              num_max=-1, 
              frame_interval=1, 
              ):
    rgb_files = sorted(glob.glob(os.path.join(path, "*.jpg")))[::frame_interval]
    rgb_files_prev = rgb_files[:-2]
    rgb_files_current = rgb_files[1:-1]
    rgb_files_next = rgb_files[2:]
    
    annotation_files = sorted(glob.glob(os.path.join(path, "*_label.json")))[1:-1]
    flow_files_12 = [rgb_f.replace("train_img", "train_flow_img_512x896").replace(".jpg", "flow12.npy") for rgb_f in rgb_files_current]
    flow_files_21 = [rgb_f.replace("train_img", "train_flow_img_512x896").replace(".jpg", "flow21.npy") for rgb_f in rgb_files_current]
    
    num_files = len(rgb_files_current)
    dataset = []
    for i, [rgb_file_p, rgb_file_c, rgb_file_n, ann_file, flow_12, flow_21] in enumerate(zip(rgb_files_prev, rgb_files_current, rgb_files_next, annotation_files, flow_files_12, flow_files_21)):
            
        if i%(num_files//4)==0:
            print("\r----- loading dataset {}/{} -----".format(i+1, num_files), end="")

        ann = json.load(open(ann_file, 'r')) 
        rectangles = np.array(ann["rectangles"], np.float32)
        num_player = int(ann["num_player"])
        num_labels = np.array(ann["num_contact_labels"], np.int32).sum()
        player_id = np.array(ann["player_id_1"], np.int32)
        player_id_1 = np.array(chain_list([[pid] * num for pid, num in zip(ann["player_id_1"], ann["num_contact_labels"])]), np.int32)
        player_id_2 = np.array(chain_list(ann["player_id_2"]), np.int32)
        contact_labels = np.array(chain_list(ann["contact_labels"]), np.int32)
        contact_pairlabels = np.vstack([player_id_1, player_id_2, contact_labels]).T
        #contact_labels = np.array(ann["contact_labels"])#, np.int32)
        
        if num_labels == 0 :
            continue

        data = {"file_p": rgb_file_p,
                "file_c": rgb_file_c,
                "file_n": rgb_file_n,
                "rectangles": rectangles, 
                "player_id": player_id, 
                "contact_pairlabels": contact_pairlabels,
                "num_labels": num_labels,
                "num_player":num_player, 
                "img_height": 720,
                "img_width": 1280,
                }
        
        data["flow_12"] = flow_12
        data["flow_21"] = flow_21
        data["flow_width"] = 896//8
        data["flow_height"] = 512//8

        dataset.append(data)
    return dataset

def load_data_5frames(path,
              num_max=-1, 
              frame_interval=1, 
              ):
    rgb_files = sorted(glob.glob(os.path.join(path, "*.jpg")))[::frame_interval]
    rgb_files_pprev = rgb_files[4:]
    rgb_files_prev = rgb_files[3:-1]
    rgb_files_current = rgb_files[2:-2]
    rgb_files_next = rgb_files[1:-3]
    rgb_files_nnext = rgb_files[:-4]

    annotation_files = sorted(glob.glob(os.path.join(path, "*_label.json")))[1:-1]
    flow_files_12 = [rgb_f.replace("train_img", "train_flow_img_512x896").replace(".jpg", "flow12.npy") for rgb_f in rgb_files_current]
    flow_files_21 = [rgb_f.replace("train_img", "train_flow_img_512x896").replace(".jpg", "flow21.npy") for rgb_f in rgb_files_current]
    
    num_files = len(rgb_files_current)
    dataset = []
    for i, [rgb_file_pp, rgb_file_p, rgb_file_c, rgb_file_n,  rgb_file_nn, ann_file, flow_12, flow_21] in enumerate(zip(rgb_files_pprev, rgb_files_prev, rgb_files_current, rgb_files_next, rgb_files_nnext, annotation_files, flow_files_12, flow_files_21)):
            
        if i%(num_files//4)==0:
            print("\r----- loading dataset {}/{} -----".format(i+1, num_files), end="")

        ann = json.load(open(ann_file, 'r')) 
        rectangles = np.array(ann["rectangles"], np.float32)
        num_player = int(ann["num_player"])
        num_labels = np.array(ann["num_contact_labels"], np.int32).sum()
        player_id = np.array(ann["player_id_1"], np.int32)
        player_id_1 = np.array(chain_list([[pid] * num for pid, num in zip(ann["player_id_1"], ann["num_contact_labels"])]), np.int32)
        player_id_2 = np.array(chain_list(ann["player_id_2"]), np.int32)
        contact_labels = np.array(chain_list(ann["contact_labels"]), np.int32)
        contact_pairlabels = np.vstack([player_id_1, player_id_2, contact_labels]).T
        #contact_labels = np.array(ann["contact_labels"])#, np.int32)
        
        if num_labels == 0 :
            continue

        data = {"file_pp": rgb_file_pp,
                "file_p": rgb_file_p,
                "file_c": rgb_file_c,
                "file_n": rgb_file_n,
                "file_nn": rgb_file_nn,
                "rectangles": rectangles, 
                "player_id": player_id, 
                "contact_pairlabels": contact_pairlabels,
                "num_labels": num_labels,
                "num_player":num_player, 
                "img_height": 720,
                "img_width": 1280,
                }
        
        data["flow_12"] = flow_12
        data["flow_21"] = flow_21
        data["flow_width"] = 896//8
        data["flow_height"] = 512//8

        dataset.append(data)
    return dataset


def decode_image(dataset):
    def read_jpg(img_file):
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img, channels=3)
        return img
    dataset["rgb_pp"] = read_jpg(dataset["file_pp"])
    dataset["rgb_p"] = read_jpg(dataset["file_p"])
    dataset["rgb_c"] = read_jpg(dataset["file_c"])
    dataset["rgb_n"] = read_jpg(dataset["file_n"])
    dataset["rgb_nn"] = read_jpg(dataset["file_nn"])
    return dataset

def decode_flow_npy(file, height, width, num_ch=2):
    header_offset = 128#npy_header_offset(filename)
    dtype = tf.float32
    def npy_header_offset(npy_path):
        with open(str(npy_path), 'rb') as f:
            if f.read(6) != b'\x93NUMPY':
                raise ValueError('Invalid NPY file.')
            version_major, version_minor = f.read(2)
            if version_major == 1:
                header_len_size = 2
            elif version_major == 2:
                header_len_size = 4
            else:
                raise ValueError('Unknown NPY file version {}.{}.'.format(version_major, version_minor))
            header_len = sum(b << (8 * i) for i, b in enumerate(f.read(header_len_size)))
            header = f.read(header_len)
            if not header.endswith(b'\n'):
                raise ValueError('Invalid NPY file.')
            return f.tell()
    size = height * width * num_ch
    raw = tf.io.read_file(file)
    raw = tf.strings.substr(raw, pos=header_offset, len=size * dtype.size)
    output = tf.io.decode_raw(raw, dtype)#, fixed_length= size * dtype.size)
    return output    


def cast_and_reshape_dataset(dataset):
    
    height = dataset["img_height"]
    width = dataset["img_width"]
    #num_box = dataset["rectangle_num"]# +dataset["current_rectangle_num"] + dataset["next_rectangle_num"]
    #for key in ["previous_rgb","current_rgb","next_rgb"]:
    dataset["rgb_pp"] = tf.reshape(tf.cast(dataset["rgb_pp"], tf.float32),[height, width, 3])
    dataset["rgb_p"] = tf.reshape(tf.cast(dataset["rgb_p"], tf.float32),[height, width, 3])
    dataset["rgb_c"] = tf.reshape(tf.cast(dataset["rgb_c"], tf.float32),[height, width, 3])
    dataset["rgb_n"] = tf.reshape(tf.cast(dataset["rgb_n"], tf.float32),[height, width, 3])
    dataset["rgb_nn"] = tf.reshape(tf.cast(dataset["rgb_nn"], tf.float32),[height, width, 3])
    #dataset["rectangles"] = tf.reshape(tf.cast(dataset["rectangles"], tf.float32),[num_box, 4, 2])
    #dataset["locations"] = tf.reshape(tf.cast(dataset["locations"], tf.float32),[num_box, 1, 2]) 
    #dataset["team_labels"] = tf.reshape(tf.cast(dataset["team_labels"], tf.int32),[num_box, 1]) 
    #dataset["jersey_labels"] = tf.reshape(tf.cast(dataset["jersey_labels"], tf.int32),[num_box, 10])     
    dataset["num_player"] = tf.cast(dataset["num_player"], tf.int32)
    dataset["num_labels"] = tf.cast(dataset["num_labels"], tf.int32)
    dataset["contact_pairlabels"] = tf.reshape(tf.cast(dataset["contact_pairlabels"], tf.int32),[dataset["num_labels"], 3])
    dataset["player_id"] = tf.reshape(tf.cast(dataset["player_id"], tf.int32),[dataset["num_player"],])
    dataset["rectangles"] = tf.reshape(tf.cast(dataset["rectangles"], tf.float32),[dataset["num_player"], 4, 2])
    #dataset["player_id_2"] = split_to_list(dataset["player_id_2"], dataset["num_contact_labels"])
    
    if LOAD_FLOW:
        flow_12 = decode_flow_npy(dataset["flow_12"], 
                               dataset["flow_height"], 
                               dataset["flow_width"], 
                               num_ch=2)
        dataset["flow_12"] = tf.reshape(tf.cast(flow_12, tf.float32),
                                      [dataset["flow_height"], dataset["flow_width"], 2])
        
        flow_21 = decode_flow_npy(dataset["flow_21"], 
                               dataset["flow_height"], 
                               dataset["flow_width"], 
                               num_ch=2)
        dataset["flow_21"] = tf.reshape(tf.cast(flow_21, tf.float32),
                                      [dataset["flow_height"], dataset["flow_width"], 2])

    return dataset

def build_tf_dataset(original_dataset):
    def gen_wrapper(dataset, data_keys=None):
        def generator():
            for data in dataset:
                yield data
        return generator
    dataset = tf.data.Dataset.from_generator(gen_wrapper(original_dataset), output_types={"file_pp": tf.string,
                                                                                          "file_p": tf.string,
                                                                                          "file_c": tf.string,
                                                                                          "file_n": tf.string,
                                                                                          "file_nn": tf.string,
                                                                                          "rectangles": tf.float32, 
                                                                                          "player_id": tf.int32,
                                                                                          #"player_id_2": tf.int32,
                                                                                          "contact_pairlabels": tf.int32,
                                                                                          "num_labels": tf.int32,
                                                                                          "num_player": tf.int32, 
                                                                                          "img_height": tf.int32,
                                                                                          "img_width": tf.int32,
                                                                                          "flow_12": tf.string,
                                                                                          "flow_21": tf.string,
                                                                                          "flow_width": tf.int32,
                                                                                          "flow_height": tf.int32,
                                                                                          })
    
    dataset = dataset.map(decode_image, num_parallel_calls=AUTO)
    dataset = dataset.map(cast_and_reshape_dataset, num_parallel_calls=AUTO)
    return dataset

def resize_flow_to_rgb(data):
    if LOAD_FLOW:
        height, width = data["img_height"], data["img_width"]
        data["stacked_flow"] = tf.stack([data["flow_12"], data["flow_21"]])
        
        data["stacked_flow"]= tf.image.resize(data["stacked_flow"], (height, width), method="bilinear")
        origin_size = tf.cast(tf.stack([data["flow_width"], data["flow_height"]]), tf.float32)
        target_size = tf.cast(tf.stack([data["img_width"], data["img_height"]]), tf.float32)
        rate = tf.reshape(target_size / origin_size, [2,])
        data["stacked_flow"] = data["stacked_flow"] * rate#[tf.newaxis, tf.newaxis, tf.newaxis, :]
        data["flow_12"] = data["stacked_flow"][0]
        data["flow_21"] = data["stacked_flow"][1]
    return data


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
    """
    inside_frame_x = tf.math.logical_and(x_idx >= 0., x_idx <= tf.cast(width-1, tf.float32))
    inside_frame_y = tf.math.logical_and(y_idx >= 0., y_idx <= tf.cast(height-1, tf.float32))
    inside_frame = tf.math.logical_and(inside_frame_x, inside_frame_y)
    if only_mask:
        return inside_frame
    """
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
    return warped#, inside_frame

def coords_grid(height, width):    
    gy, gx = tf.meshgrid(tf.range(height, dtype=tf.float32),
                         tf.range(width, dtype=tf.float32),
                         indexing='ij')
    coords = tf.stack([gx, gy], axis=-1)
    return coords#[h,w,2]

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

def rgbreconst_by_flow(data, default_grid, input_shape):
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

def rgbrediff_by_flow(data):
    for suffix in ["p", "n"]:
        data[f"rgb_diff_{suffix}"] = data[f"rgb_reconst_{suffix}"] - data["rgb_c"]
    return data

def normalize_inputs_outputs(data):
    h, w = tf.unstack(tf.shape(data["rgb_c"]))[:2]
    #data["rgb_p"] = (data["rgb_c"] - data["rgb_reconst_from_p"])/255
    #data["rgb_n"] = (data["rgb_reconst_from_n"] - data["rgb_c"])/255
    #data["rgb_c"] = data["rgb_c"]/255
    
    
    #data["rgb_p"] = data["rgb_reconst_from_p"]/255
    #data["rgb_n"] = data["rgb_reconst_from_n"]/255
    #data["rgb_c"] = data["rgb_c"]/255
    
    data["rgb_nn"] = data["rgb_nn"]/255
    data["rgb_n"] = data["rgb_n"]/255
    data["rgb_c"] = data["rgb_c"]/255
    data["rgb_p"] = data["rgb_p"]/255
    data["rgb_pp"] = data["rgb_pp"]/255
    
    """
    if LOAD_FLOW:
        base_helmet_size = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum((data["rectangles"][:,0,:] - data["rectangles"][:,2,:])**2, axis=-1)))
        data["flow_12"] = data["flow_12"] / base_helmet_size
        data["flow_21"] = data["flow_21"] / base_helmet_size
        
        # minus average(camera?) 差分取りするときには注意するほうがいいかも（事前にとるべきかも）。
        data["flow_12"] = data["flow_12"] - tf.reduce_mean(data["flow_12"], axis=[0,1], keepdims=True)
        data["flow_21"] = data["flow_21"] - tf.reduce_mean(data["flow_21"], axis=[0,1], keepdims=True)
    """
    data["rectangles"] = data["rectangles"]/tf.cast(tf.stack([[[w,h]]]),tf.float32)
    return data

def resize_flow_to_warp_w_target_scale(data, target_shape):
    height, width = target_shape
    coords = coords_grid(height, width)

    flow_height, flow_width = tf.unstack(tf.shape(data["flow_12"]))[:2]
    data["stacked_flow"] = tf.stack([data["flow_12"], data["flow_21"]])
    
    data["stacked_flow"]= tf.image.resize(data["stacked_flow"], (height, width), method="bilinear")
    origin_size = tf.cast(tf.stack([flow_width, flow_height]), tf.float32)
    target_size = tf.cast(tf.stack([width, height]), tf.float32)
    rate = tf.reshape(target_size / origin_size, [2,])
    data["stacked_flow"] = data["stacked_flow"] * rate#[tf.newaxis, tf.newaxis, tf.newaxis, :]
    data["warp_n"] = coords + data["stacked_flow"][0]
    data["warp_p"] = coords + data["stacked_flow"][1]
    return data

def pair_labels_to_matrix(player_ids, pair_labels):
    """
    player_ids:
        player_ids exist on image
    pair_labels:
        pair and contact label [num_sample, 3(player_1, player_2, contact_label)]
    
    returns:
        contatc_label matrix [num_player, num_player]
        no_label == -1
        no_contact == 0
        contact == 1
    """
    p1_delta = player_ids[:,tf.newaxis] - pair_labels[tf.newaxis,:,0]
    p1_exist = (p1_delta == 0)[:,tf.newaxis,:] # shape[num_player, 1, num_sample]
    p2_delta = tf.pad(player_ids, [[0,1]], constant_values=0)[:,tf.newaxis] - pair_labels[tf.newaxis,:,1]
    #p2_exist = (tf.logical_or(p2_delta==0, pair_labels[tf.newaxis,:,1]==0))[tf.newaxis,:,:]
    p2_exist = (p2_delta == 0)[tf.newaxis,:,:] # shape[num_player, 1, num_sample]
    pair_exist = tf.cast(tf.logical_and(p1_exist, p2_exist), tf.int32) # shape[num_player, num_player, num_sample]
    
    label = pair_labels[tf.newaxis,tf.newaxis,:,2] + 1 # plus one will be removed later
    label_matrix = tf.reduce_sum(label * pair_exist, axis=2) - 1
    return label_matrix

def pair_labels_id2indices(player_ids, pair_labels, ground_as_zero=True):
    """
    player_ids:
        player_ids exist on image
    pair_labels:
        pair and contact label [num_sample, 3(player_1, player_2, contact_label)]
    
    returns:
         pair and contact label [num_sample, 3(player_1_indices, player_2_indices, contact_label)]
    """
    if ground_as_zero:
        player_ids = tf.pad(player_ids, [[1,0]], constant_values=0)
    p1_ind = tf.where((player_ids[tf.newaxis,:] - pair_labels[:,tf.newaxis,0])==0)[:,1:]
    p2_ind = tf.where((player_ids[tf.newaxis,:] - pair_labels[:,tf.newaxis,1])==0)[:,1:]
    p1_ind = tf.cast(p1_ind, tf.int32)
    p2_ind = tf.cast(p2_ind, tf.int32)
    return tf.concat([p1_ind, p2_ind, pair_labels[:,2:]], axis=-1)
    
#MAX_BOX_NUM = 20
#max_pair_num = 40

def box_xycoords_to_tlbr(data, shuffle_order=True, max_box_num=20, max_pair_num=40):
    left_tops = tf.reduce_min(data["rectangles"], axis=1)
    right_bottoms = tf.reduce_max(data["rectangles"], axis=1)
    box_tlbr = tf.concat([left_tops[:,::-1], right_bottoms[:,::-1]], axis=-1)
    box_tlbr = tf.clip_by_value(box_tlbr, tf.zeros((1,4),tf.float32), tf.ones((1,4),tf.float32))
    box_size = (box_tlbr[...,2] - box_tlbr[...,0]) * (box_tlbr[...,3] - box_tlbr[...,1])
    # neglect too small box
    mask = (box_size > 1e-6)
    box_tlbr = tf.boolean_mask(box_tlbr, mask)[:max_box_num]
    player_id = tf.boolean_mask(data["player_id"], mask)[:max_box_num]
    num_survive = tf.minimum(tf.reduce_sum(tf.cast(mask, tf.int32)), max_box_num)
    
    
    contact_ids = data["contact_pairlabels"]
    if shuffle_order:
        order = tf.random.shuffle(tf.range(num_survive))
        box_tlbr = tf.gather(box_tlbr, order)
        player_id = tf.gather(player_id, order)
        contact_ids = tf.random.shuffle(contact_ids)
    
    
    contact_pairs = contact_ids[tf.newaxis, :, :2]
    contact_ids_alive = tf.reduce_any((contact_pairs - player_id[:, tf.newaxis, tf.newaxis])==0, axis=[0])
    contact_ids_mask = tf.logical_and(contact_ids_alive[:,0], tf.logical_or(contact_ids_alive[:,1], contact_ids[:,1]==0))#groud or alive player
    contact_pairlabels = tf.boolean_mask(contact_ids, contact_ids_mask)
    
    num_survive_label = tf.minimum(tf.reduce_sum(tf.cast(contact_ids_mask, tf.int32)), max_pair_num)
    data["player_id"] = player_id
    data["num_player"] = num_survive
    data["rectangles"] = tf.reshape(box_tlbr, [num_survive,4])
    data["num_labels"] = num_survive_label
    data["contact_pairlabels_indices"] = pair_labels_id2indices(player_id, contact_pairlabels, ground_as_zero=True)
    data["contact_pairlabels_indices"] = data["contact_pairlabels_indices"][:max_pair_num]
    data["contact_pairlabels"] = contact_pairlabels[:max_pair_num]
    return data

def pad_box_and_labels_if_necessary(data, max_box_num=20, max_pair_num=40):
    invalid_label = tf.constant([[0,0,-1]], dtype=tf.int32)
    num_tile = max_pair_num - data["num_labels"]
    data["contact_pairlabels_indices"] = tf.concat([data["contact_pairlabels_indices"], tf.tile(invalid_label, [num_tile,1])], axis=0)
    data["contact_pairlabels"] = tf.concat([data["contact_pairlabels"], tf.tile(invalid_label, [num_tile,1])], axis=0)
    
    #num_tile = tf.math.ceil(tf.cast(max_box_num - data["num_player"], tf.float32) / tf.cast(data["num_player"], tf.float32))
    # 無意味にオーバーラップがでるのでやめるべき？、バッチノームの統計値が心配ではある
    num_tile = 1 + (max_box_num - data["num_player"]) // data["num_player"]
    data["rectangles"] = tf.concat([data["rectangles"], tf.tile(data["rectangles"], [num_tile,1])], axis=0)[:max_box_num]
    
    #invalid_label = tf.constant([[0.5,0.5,0.5+1e-7, 0.5+1e-7]], dtype=tf.float32)
    #num_tile = max_box_num - data["num_player"]
    #data["rectangles"] = tf.concat([data["rectangles"], tf.tile(invalid_label, [num_tile,1])], axis=0)
    
    return data
    
def assign_input_output(data):
    #if LOAD_FLOW:
    #    inputs = tf.concat([data["rgb"], data["flow_12"], data["flow_21"]], axis=-1)
    #else:
    #inputs = tf.concat([data["rgb_p"],data["rgb_c"],data["rgb_n"]], axis=-1)
        
    #inputs = {"input_rgb": inputs,
    #          "input_boxes": data["rectangles"],
    #          "input_pairs": data["contact_pairlabels_indices"][:,:2],
    #          }
    
    inputs = {"input_rgb_pp": data["rgb_p"],
              "input_rgb_p": data["rgb_p"],
              "input_rgb_c": data["rgb_c"],
              "input_rgb_n": data["rgb_n"],
              "input_rgb_nn": data["rgb_nn"],
              "input_boxes": data["rectangles"],
              "input_pairs": data["contact_pairlabels_indices"][:,:2],
              #"input_warp_p": data["warp_p"],
              #"input_warp_n": data["warp_n"],
                        
                    }
    
    
    outputs = {
              "output_contact_label": data["contact_pairlabels_indices"][:,2],
              "contact_map": tf.ones((1,1)), # dummy
              }

    return inputs, outputs

def assign_input_output_w_info(data):
    inputs = {"input_rgb": data["rgb"],
              "input_boxes": data["rectangles"],
              "input_pairs": data["contact_pairlabels_indices"][:,:2],
              }
    outputs = {
              "output_contact_label": data["contact_pairlabels_indices"][:,2],
              "contact_map": tf.ones((1,1)), # dummy
              }
    info_keys = ["num_labels", "num_player", "contact_pairlabels"]
    info = {k: data[k] for k in info_keys}
    return inputs, outputs, info

def get_tf_dataset(list_dataset, 
                  input_shape, 
                  output_shape,
                  batch_size, 
                  transforms, 
                  #max_box_num=22,
                  is_train=True,
                  max_box_num=20,
                  max_pair_num=40):
    print("start building dataset")
    #default_grid = coords_grid(list_dataset[0]["img_height"], list_dataset[0]["img_width"])[tf.newaxis,...]
    default_grid = coords_grid(input_shape[0], input_shape[1])[tf.newaxis,...]
    
    dataset = build_tf_dataset(list_dataset)
    
    dataset = dataset.map(resize_flow_to_rgb, num_parallel_calls=AUTO)
    dataset = dataset.map(transforms, num_parallel_calls=AUTO)
    #dataset = dataset.filter(lambda x: x["rectangle_num"] > 3)
    
    #dataset = dataset.map(lambda x: rgbreconst_by_flow(x, default_grid, input_shape), num_parallel_calls=AUTO)
    
    dataset = dataset.map(normalize_inputs_outputs, num_parallel_calls=AUTO)
    # dataset = dataset.map(lambda x: resize_flow_to_warp_w_target_scale(x, output_shape), num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: box_xycoords_to_tlbr(x, max_box_num=max_box_num, max_pair_num=max_pair_num), num_parallel_calls=AUTO)
    dataset = dataset.filter(lambda x: x["num_player"] > 1)
    
    dataset = dataset.map(lambda x: pad_box_and_labels_if_necessary(x, max_box_num=max_box_num, max_pair_num=max_pair_num), num_parallel_calls=AUTO)
    dataset = dataset.map(assign_input_output, num_parallel_calls=AUTO)
    
    if is_train:
        dataset = dataset.shuffle(128)
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
    else:# cache sometimes causes memory error
        #dataset = dataset.cache()
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset

def get_tf_dataset_inference(list_dataset, 
                  transforms, 
                  batch_size=1, 
                  max_box_num=25, # 
                  max_pair_num=100, # enough large
                  padding=True):
    """
    batch_size must be 1, if the model is not separated.
    """
    print("start building dataset")
    dataset = build_tf_dataset(list_dataset)
    
    dataset = dataset.map(transforms, num_parallel_calls=AUTO)
    #dataset = dataset.filter(lambda x: x["rectangle_num"] > 3)
    dataset = dataset.map(normalize_inputs_outputs, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: box_xycoords_to_tlbr(x, max_box_num=max_box_num, max_pair_num=max_pair_num, shuffle_order=False), num_parallel_calls=AUTO)
    dataset = dataset.filter(lambda x: x["num_player"] >= 1)
    if padding:
        dataset = dataset.map(lambda x: pad_box_and_labels_if_necessary(x, max_box_num=max_box_num, max_pair_num=max_pair_num), num_parallel_calls=AUTO)
    dataset = dataset.map(assign_input_output_w_info, num_parallel_calls=AUTO)
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset


def get_tf_dataset_inference_auto_resize(list_dataset, 
                  batch_size=1, 
                  max_box_num=25, # 
                  max_pair_num=100, # enough large
                  padding=True):
    """
    batch_size must be 1
    """
    if batch_size!=1:
        raise Exception("batch size must be 1")
    print("start building dataset")
    dataset = build_tf_dataset(list_dataset)
    
    #dataset = dataset.filter(lambda x: x["rectangle_num"] > 3)
    dataset = dataset.map(lambda x: resize_by_box_shape(x, target_box_length=20, stride_step=64), num_parallel_calls=AUTO)
    
    dataset = dataset.map(normalize_inputs_outputs, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: box_xycoords_to_tlbr(x, max_box_num=max_box_num, max_pair_num=max_pair_num), num_parallel_calls=AUTO)
    #dataset = dataset.filter(lambda x: x["num_player"] > 1)
    if padding:
        dataset = dataset.map(lambda x: pad_box_and_labels_if_necessary(x, max_box_num=max_box_num, max_pair_num=max_pair_num), num_parallel_calls=AUTO)
    dataset = dataset.map(assign_input_output_w_info, num_parallel_calls=AUTO)
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset

def resize_by_box_shape(data, target_box_length=20, stride_step=64):
    
    img = data["rgb"]
    box = data["rectangles"]
    img_height, img_width, _ = tf.unstack(tf.shape(img))
    box = tf.reshape(box, [-1,4,2])
    left_tops = tf.reduce_min(box, axis=1)
    right_bottoms = tf.reduce_max(box, axis=1)
    box_tlbr = tf.concat([left_tops[:,::-1], right_bottoms[:,::-1]], axis=-1)
    box_size = tf.math.sqrt((box_tlbr[:,2] - box_tlbr[:,0]) * (box_tlbr[:,3] - box_tlbr[:,1]))
    resize_ratio = target_box_length / tf.reduce_mean(box_size)
    resize_ratio = tf.clip_by_value(resize_ratio, 1., 4.,) # not smaller than original size(1) and not too large(4)
    
    resize_height = stride_step * (((tf.cast(img_height, tf.float32) * resize_ratio) + stride_step * 0.5) // stride_step)
    resize_width = stride_step * (((tf.cast(img_width, tf.float32) * resize_ratio) + stride_step * 0.5) // stride_step)
    h_rate = resize_height / tf.cast(img_height, tf.float32)
    w_rate = resize_width / tf.cast(img_width, tf.float32)
    
    resize_height = tf.cast(resize_height, tf.int32)
    resize_width = tf.cast(resize_width, tf.int32)    
    img = tf.image.resize(img[tf.newaxis, ...], (resize_height, resize_width), method="bilinear")[0,:,:,:]
    box = box * tf.cast(tf.stack([[[w_rate,h_rate]]]), tf.float32)
    
    data["rgb"] = img
    data["rectangles"] = box

    return data
    

def inference_preprocess(data, 
                  transforms, # maybe crop
                  max_box_num=25, # 
                  max_pair_num=100, # enough large
                  padding=True):
    """
    data:
        dict
        "rgb": 0-255 tf.float32 rgb[h,w,3],
        "rectangles": tf.float32 [num_player, 4(tlbr), 2(xy)], 
        "player_id": tf.int32 [num_player],
        "contact_pairlabels": tf.int32  [num_sample, 3(player_1, player_2, binary_label)],
        "num_labels": tf.int32 (=num_sample),
        "num_player": tf.int32 (=num_player), 
        "img_height": tf.int32,
        "img_width": tf.int32,
    
    returns:
        inputs, targets, other_info                                                                                              
    """
    data = transforms(data)
    data = normalize_inputs_outputs(data)
    data = box_xycoords_to_tlbr(data, max_box_num=max_box_num, max_pair_num=max_pair_num, shuffle_order=False)
    if padding:
        data = pad_box_and_labels_if_necessary(data, max_box_num=max_box_num, max_pair_num=max_pair_num)
    inputs, targets, other_info = assign_input_output_w_info(data)
    return inputs, targets, other_info


def inference_preprocess_auto_resize(data, 
                  max_box_num=25, # 
                  max_pair_num=100, # enough large
                  padding=True):
    """
    data:
        dict
        "rgb": 0-255 tf.float32 rgb[h,w,3],
        "rectangles": tf.float32 [num_player, 4(tlbr), 2(xy)], 
        "player_id": tf.int32 [num_player],
        "contact_pairlabels": tf.int32  [num_sample, 3(player_1, player_2, binary_label)],
        "num_labels": tf.int32 (=num_sample),
        "num_player": tf.int32 (=num_player), 
        "img_height": tf.int32,
        "img_width": tf.int32,
    
    returns:
        inputs, targets, other_info                                                                                              
    """
    data = resize_by_box_shape(data, target_box_length=20, stride_step=64)
    data = normalize_inputs_outputs(data)
    data = box_xycoords_to_tlbr(data, max_box_num=max_box_num, max_pair_num=max_pair_num, shuffle_order=False)
    if padding:
        data = pad_box_and_labels_if_necessary(data, max_box_num=max_box_num, max_pair_num=max_pair_num)
    inputs, targets, other_info = assign_input_output_w_info(data)
    return inputs, targets, other_info


def inference_preprocess_batch(data, 
                  transforms, # maybe crop or resize?
                  max_box_num=25, # 
                  max_pair_num=100, # enough large
                  padding=True):
    raise Exception("to be implemented")
    """
    data:
        dict
        "rgb": 0-255 tf.float32 rgb[h,w,3],
        "rectangles": tf.float32 [num_player, 4(tlbr), 2(xy)], 
        "player_id": tf.int32 [num_player],
        "contact_pairlabels": tf.int32  [num_sample, 3(player_1, player_2, binary_label)],
        "num_labels": tf.int32 (=num_sample),
        "num_player": tf.int32 (=num_player), 
        "img_height": tf.int32,
        "img_width": tf.int32,
    
    returns:
        inputs, targets, other_info                                                                                              
    """
    data = transforms(data)
    data = normalize_inputs_outputs(data)
    data = box_xycoords_to_tlbr(data, max_box_num=max_box_num, max_pair_num=max_pair_num, shuffle_order=False)
    if padding:
        data = pad_box_and_labels_if_necessary(data, max_box_num=max_box_num, max_pair_num=max_pair_num)
    inputs, targets, other_info = assign_input_output_w_info(data)
    return inputs, targets, other_info




####
#### --- GCN MODEL ---
####
def load_data_gcn(path,
              ):
    
    name_and_dtype = {"label_g_contact": np.int32, # [num_step, players] player ground contact. (training target)
                    "label_p_contact": np.int32, # [num_step, players, players] player pairs contact. (training target)
                    "p2p_adj_dist_matrix": np.float32, # [num_frame, players, players] yard distance btw players
                    "p2p_adj_team_matrix": np.float32, # [players, players] to show same team (-> diff team)
                    #"player_2d_num_matrix": tf.float32, # [players, numerical_features] tbd
                    #"player_2d_cat_matrix": tf.int32, # [players, categorical_features] position 
                    "player_3d_num_matrix": np.float32, # [num_frame, players, numerical_features] speed, 
                    #"num_player": tf.string,
                    #"frame_range": tf.int32, # [num_frame] frame number of features
                    #"num_features": tf.int32,
                    "unique_players":np.int32,
                    "step_range": np.int32,
                    }
    
    #files = [os.path.join(path, f"{name}.npy") for name in name_and_dtype.keys()]
    #gameplay_data = [np.load(f) for f in files]
    #data = {key: gameplay_data[i].astype(val) for i,[key,val] in enumerate(name_and_dtype.items())}
    data = {key: np.load(os.path.join(path, f"{key}.npy")).astype(val) for i,[key,val] in enumerate(name_and_dtype.items())}
    n_steps, n_players, n_features = data["player_3d_num_matrix"].shape[:3]
    data["num_players"] = n_players
    data["num_steps"] = n_steps # [num_frame] frame number of features
    data["num_features"] = n_features
    #each_files = [sorted(glob.glob(os.path.join(path, f"{name}.npy"))) for name in name_and_dtype.keys()]

    #num_files = len(each_files[0])
    #dataset = []
    #for i, files in enumerate(zip(each_files)):
    #    print("\r----- loading dataset {}/{} -----".format(i+1, num_files), end="")
    #    gameplay_data = [np.load(f) for f in files]
    #    dataset += [{key: gameplay_data[i].astype(val) for i,[key,val] in enumerate(name_and_dtype.items())}]
    return [data]
    
def cast_and_reshape_dataset_gcn(data):
    num_players = data["num_players"]
    num_steps = data["num_steps"] # [num_frame] frame number of features
    num_features = data["num_features"]
    data["label_g_contact"] = tf.reshape(data["label_g_contact"], [num_steps, num_players]) # player ground contact. (training target)
    data["label_p_contact"] = tf.reshape(data["label_p_contact"], [num_steps, num_players, num_players]) # player pairs contact. (training target)
    data["p2p_adj_dist_matrix"] = tf.reshape(data["p2p_adj_dist_matrix"], [num_steps, num_players, num_players])                                                       
    data["p2p_adj_team_matrix"] = tf.reshape(data["p2p_adj_team_matrix"], [num_players, num_players])                                                       
    data["player_3d_num_matrix"] = tf.reshape(data["player_3d_num_matrix"], [num_steps, num_players, num_features])                                                       
    data["step_range"] = tf.reshape(data["step_range"], [num_steps])                                                       
    data["step_range_norm"] = tf.cast(data["step_range"], tf.float32)
    data["step_range_norm"] = (data["step_range_norm"] - tf.reduce_min(data["step_range_norm"])) / (tf.reduce_max(data["step_range_norm"]) - tf.reduce_min(data["step_range_norm"]))
    return data


    
    
def build_tf_dataset_gcn(original_dataset):
    def gen_wrapper(dataset, data_keys=None):
        def generator():
            for data in dataset:
                yield data
        return generator
    dataset = tf.data.Dataset.from_generator(gen_wrapper(original_dataset), output_types={"label_g_contact": tf.int32, # [num_step, players] player ground contact. (training target)
                                                                                          "label_p_contact": tf.int32, # [num_step, players, players] player pairs contact. (training target)
                                                                                          "p2p_adj_dist_matrix": tf.float32, # [num_frame, players, players] yard distance btw players
                                                                                          "p2p_adj_team_matrix": tf.float32, # [players, players] to show same team (-> diff team)
                                                                                          #"player_2d_num_matrix": tf.float32, # [players, numerical_features] tbd
                                                                                          #"player_2d_cat_matrix": tf.int32, # [players, categorical_features] position 
                                                                                          "player_3d_num_matrix": tf.float32, # [num_frame, players, numerical_features] speed, 
                                                                                          "num_players": tf.int32,
                                                                                          "num_steps": tf.int32, # [num_frame] frame number of features
                                                                                          "num_features": tf.int32,
                                                                                          "step_range": tf.int32, # [num_step] frame number of targets
                                                                                          "unique_players":tf.int32,
                                                                                          })
    
    #dataset = dataset.map(decode_image, num_parallel_calls=AUTO)
    dataset = dataset.map(cast_and_reshape_dataset_gcn, num_parallel_calls=AUTO)
    return dataset

def crop_sequence(data, sequence_length=7):
    start = tf.random.uniform((), 0, sequence_length, tf.int32)
    num_steps, num_players, num_features = tf.unstack(tf.shape(data["player_3d_num_matrix"]))
    num_sequence = (num_steps - start) // sequence_length
    end = start + num_sequence * sequence_length
    data["player_3d_num_matrix"] = tf.reshape(data["player_3d_num_matrix"][start:end], [num_sequence, sequence_length, num_players, num_features])
    data["step_range"] = tf.reshape(data["step_range"][start:end], [num_sequence, sequence_length])
    data["step_range_norm"] = tf.reshape(data["step_range_norm"][start:end], [num_sequence, sequence_length])
    data["p2p_adj_dist_matrix"] = tf.reshape(data["p2p_adj_dist_matrix"][start:end], [num_sequence, sequence_length, num_players, num_players])
    data["p2p_adj_team_matrix"] = tf.tile(tf.reshape(data["p2p_adj_team_matrix"], [1, 1, num_players, num_players]), [num_sequence, sequence_length, 1, 1])
    data["label_p_contact"] = tf.reshape(data["label_p_contact"][start:end], [num_sequence, sequence_length, num_players, num_players])
    data["label_g_contact"] = tf.reshape(data["label_g_contact"][start:end], [num_sequence, sequence_length, num_players])
    data["num_steps"] = num_steps
    data["num_players"] = num_players
    data["sequence_length"] = sequence_length
    
    return data

def preprop_inference(data, sequence_length=7, num_default_player=22):
    """
    numpy?
    """
    #num_total_steps = len(data["step_range"])
    data["step_range_norm"] = tf.cast(data["step_range"], tf.float32)
    data["step_range_norm"] = (data["step_range_norm"] - tf.reduce_min(data["step_range_norm"])) / (tf.reduce_max(data["step_range_norm"]) - tf.reduce_min(data["step_range_norm"]))
    
    
    # num_total_steps, num_players, num_features = data["player_3d_num_matrix"].shape
    num_total_steps, num_players, num_features = tf.unstack(tf.shape(data["player_3d_num_matrix"]))
    #starts = np.minimum(np.arange(0, num_total_steps, step), num_total_steps-step) # final sequence can be overlapped
    starts = tf.minimum(tf.range(0, num_total_steps, sequence_length), num_total_steps - sequence_length) # final sequence can be overlapped
    ends = starts + sequence_length
    num_sequence = tf.shape(starts)[0] # ==batch
    
    data["player_3d_num_matrix"] = tf.stack([data["player_3d_num_matrix"][start:end] for start, end in zip(starts, ends)], axis=0)
    data["step_range"] = tf.stack([data["step_range"][start:end] for start, end in zip(starts, ends)], axis=0)
    data["step_range_norm"] = tf.stack([data["step_range_norm"][start:end] for start, end in zip(starts, ends)], axis=0)
    data["p2p_adj_dist_matrix"] = tf.stack([data["p2p_adj_dist_matrix"][start:end] for start, end in zip(starts, ends)], axis=0)
    data["p2p_adj_team_matrix"] = tf.tile(tf.reshape(data["p2p_adj_team_matrix"], [1, 1, num_players, num_players]), [num_sequence, sequence_length, 1, 1])

    data["label_p_contact"] = tf.stack([data["label_p_contact"][start:end] for start, end in zip(starts, ends)], axis=0)
    data["label_g_contact"] = tf.stack([data["label_g_contact"][start:end] for start, end in zip(starts, ends)], axis=0)
    
    data["num_players"] = num_players
    
    data = pad_player_if_necessary(data, num_default_player, no_label=False)
    data = normalize_inputs_outputs_gcn(data)
    """
    num_players = tf.minimum(data["num_players"], num_default_player)
    num_pad = num_default_player - num_players
    data["player_3d_num_matrix"] = data["player_3d_num_matrix"][:,:,:num_players,:]
    data["p2p_adj_dist_matrix"] = data["p2p_adj_dist_matrix"][:,:,:num_players,:num_players]
    data["p2p_adj_team_matrix"] = data["p2p_adj_team_matrix"][:,:,:num_players,:num_players]
    data["num_players"] = num_players
    
    data["player_3d_num_matrix"] = tf.pad(data["player_3d_num_matrix"], [[0,0],[0,0],[0,num_pad],[0,0]], constant_values=0)
    data["p2p_adj_dist_matrix"] = tf.pad(data["p2p_adj_dist_matrix"], [[0,0],[0,0],[0,num_pad],[0,num_pad]], constant_values=0)
    data["p2p_adj_team_matrix"] = tf.pad(data["p2p_adj_team_matrix"], [[0,0],[0,0],[0,num_pad],[0,num_pad]], constant_values=0)
    
    """
    
    #start = tf.random.uniform((), 0, sequence_length, tf.int32)
    #num_steps, num_players, num_features = tf.unstack(tf.shape(data["player_3d_num_matrix"]))
    #num_sequence = (num_steps - start) // sequence_length
    #end = start + num_sequence * sequence_length
    #data["player_3d_num_matrix"] = tf.reshape(data["player_3d_num_matrix"][start:end], [num_sequence, sequence_length, num_players, num_features])
    #data["step_range"] = tf.reshape(data["step_range"][start:end], [num_sequence, sequence_length])
    #data["step_range_norm"] = tf.reshape(data["step_range_norm"][start:end], [num_sequence, sequence_length])
    #data["p2p_adj_dist_matrix"] = tf.reshape(data["p2p_adj_dist_matrix"][start:end], [num_sequence, sequence_length, num_players, num_players])
    #data["p2p_adj_team_matrix"] = tf.tile(tf.reshape(data["p2p_adj_team_matrix"], [1, 1, num_players, num_players]), [num_sequence, sequence_length, 1, 1])
    #data["label_p_contact"] = tf.reshape(data["label_p_contact"][start:end], [num_sequence, sequence_length, num_players, num_players])
    #data["label_g_contact"] = tf.reshape(data["label_g_contact"][start:end], [num_sequence, sequence_length, num_players])
    #data["num_steps"] = num_steps
    #data["num_players"] = num_players
    #data["sequence_length"] = sequence_length
    
    return data

def pad_player_if_necessary(data, num_default=22, no_label=False):
    num_players = tf.minimum(data["num_players"], num_default)
    num_pad = num_default - num_players
    data["player_3d_num_matrix"] = data["player_3d_num_matrix"][:,:,:num_players,:]
    data["p2p_adj_dist_matrix"] = data["p2p_adj_dist_matrix"][:,:,:num_players,:num_players]
    data["p2p_adj_team_matrix"] = data["p2p_adj_team_matrix"][:,:,:num_players,:num_players]
    if not no_label:
        data["label_p_contact"] = data["label_p_contact"][:,:,:num_players,:num_players]
        data["label_g_contact"] = data["label_g_contact"][:,:,:num_players]
    data["num_players"] = num_players
    
    data["player_3d_num_matrix"] = tf.pad(data["player_3d_num_matrix"], [[0,0],[0,0],[0,num_pad],[0,0]], constant_values=0)
    data["p2p_adj_dist_matrix"] = tf.pad(data["p2p_adj_dist_matrix"], [[0,0],[0,0],[0,num_pad],[0,num_pad]], constant_values=0)
    data["p2p_adj_team_matrix"] = tf.pad(data["p2p_adj_team_matrix"], [[0,0],[0,0],[0,num_pad],[0,num_pad]], constant_values=0)
    if not no_label:
        data["label_p_contact"] = tf.pad(data["label_p_contact"], [[0,0],[0,0],[0,num_pad],[0,num_pad]], constant_values=-1)
        data["label_g_contact"] = tf.pad(data["label_g_contact"], [[0,0],[0,0],[0,num_pad]], constant_values=-1)
    return data

def mask_easy_samples(data):
    data["label_p_contact"] = data["label_p_contact"] - 2 * tf.cast(data["p2p_adj_dist_matrix"] >= 3, tf.int32)
    return data

def normalize_inputs_outputs_gcn(data):
    #dist_mat_1 = tf.cast(data["p2p_adj_dist_matrix"] < 1, tf.float32)
    #dist_mat_3 = tf.cast(data["p2p_adj_dist_matrix"] < 3, tf.float32)
    #dist_mat_1 = tf.maximum(1 - data["p2p_adj_dist_matrix"], 0)
    #dist_mat = tf.maximum(3 - data["p2p_adj_dist_matrix"], 0)
    dist_mat = tf.minimum(5., data["p2p_adj_dist_matrix"]) / 5.
    
    team_mat_same = tf.cast(data["p2p_adj_team_matrix"], tf.float32)
    
    relatives = data["player_3d_num_matrix"][:,:,tf.newaxis,:,:] - data["player_3d_num_matrix"][:,:,:,tf.newaxis,:]
    relatives = tf.transpose(relatives, [0, 4, 1, 2, 3])
    #team_mat_same = tf.cast(1-data["p2p_adj_team_matrix"], tf.float32)
    adj_mat = tf.stack([dist_mat, team_mat_same], axis=1)
    #data["adj_matrix"] = adj_mat#dist_mat[:,tf.newaxis]
    data["adj_matrix"] = tf.concat([adj_mat, relatives], axis=1)#dist_mat[:,tf.newaxis]
    return data

def select_unbatch_data(data):
    return {"player_3d_num_matrix": data["player_3d_num_matrix"],
            "adj_matrix": data["adj_matrix"],
                    "label_g_contact": data["label_g_contact"],
                            "label_p_contact": data["label_p_contact"],
                                    "step_range_norm" : data["step_range_norm"][...,tf.newaxis]}

def add_flip_pair(data):
    data["label_p_contact"] = tf.maximum(data["label_p_contact"], tf.transpose(data["label_p_contact"],[0,1,3,2]))
    return data

def assign_input_output_gcn(data, sequence_length=7):
    
    if sequence_length==1:
        inputs = {"input_features": data["player_3d_num_matrix"][0],
                  "input_adjacency_matrix": data["adj_matrix"][:,0],
                  "step_range": data["step_range_norm"][0],
                      }
        outputs = {
                  "g_contact": data["label_g_contact"][0],
                  "p_contact": data["label_p_contact"][0], # dummy
                  }        
    else:
        inputs = {"input_features": data["player_3d_num_matrix"],#tf.concat([data["player_3d_num_matrix"][...,:2],data["player_3d_num_matrix"][...,8:]], axis=-1),
                  "input_adjacency_matrix": data["adj_matrix"],
                  "step_range": data["step_range_norm"],
                  }
        outputs = {
                  "g_contact": data["label_g_contact"],
                  "p_contact": data["label_p_contact"], # dummy
                  }

    
    return inputs, outputs


def get_tf_dataset_gcn(list_dataset, 
                  #input_shape, 
                  #output_shape,
                  batch_size, 
                  #transforms, 
                  sequence_length=1,
                  num_players=22,
                  is_train=True,
                  ):
    print("start building dataset")
    
    num_data = len(list_dataset) * 50 / sequence_length # 500step~
    dataset = build_tf_dataset_gcn(list_dataset)
    
    dataset = dataset.map(lambda x: crop_sequence(x, sequence_length=sequence_length), num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: pad_player_if_necessary(x, num_default=22), num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: mask_easy_samples(x), num_parallel_calls=AUTO)
    
    dataset = dataset.map(normalize_inputs_outputs_gcn, num_parallel_calls=AUTO)
    dataset = dataset.map(add_flip_pair, num_parallel_calls=AUTO)
    
    dataset = dataset.map(select_unbatch_data, num_parallel_calls=AUTO)  
    
    
    
    dataset = dataset.unbatch().shuffle(int(num_data//4))
    
    dataset = dataset.map(lambda x: assign_input_output_gcn(x, sequence_length=sequence_length), num_parallel_calls=AUTO)
    
    if is_train:
        #dataset = dataset.shuffle(128)
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
    else:# cache sometimes causes memory error
        #dataset = dataset.cache()
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset


####
#### --- RAFT MODEL ---
####

def load_data_raft(path,
              num_max=-1, 
              frame_interval=1, 
              ):
    rgb_files = sorted(glob.glob(os.path.join(path, "*.jpg")))[::frame_interval]
    #annotation_files = sorted(glob.glob(os.path.join(path, "*_label.json")))[::frame_interval]
    num_files = len(rgb_files)
    dataset = []
    for i, [rgb_file_1, rgb_file_2] in enumerate(zip(rgb_files[:-1], rgb_files[1:])):
        if i%(num_files//4)==0:
            print("\r----- loading dataset {}/{} -----".format(i+1, num_files), end="")

        data = {"rgb_file_1": rgb_file_1,
                "rgb_file_2": rgb_file_2, 
                "rgb_height": 720,
                "rgb_width": 1280,
                }
        dataset.append(data)
    return dataset

def build_tf_dataset_raft(original_dataset):
    def gen_wrapper(dataset, data_keys=None):
        def generator():
            for data in dataset:
                yield data
        return generator
    dataset = tf.data.Dataset.from_generator(gen_wrapper(original_dataset), output_types={"rgb_file_1": tf.string,
                                                                                          "rgb_file_2": tf.string,
                                                                                          "rgb_height": tf.int32,
                                                                                          "rgb_width": tf.int32,
                                                                                          })    
    return dataset

def decode_image_raft(data):
    def read_jpg(img_file):
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img, channels=3)
        return img
    data["rgb_1"] = read_jpg(data["rgb_file_1"])
    data["rgb_2"] = read_jpg(data["rgb_file_2"])
    return data

def cast_and_reshape_dataset_raft(data):
    height = data["rgb_height"]
    width = data["rgb_width"]
    data["rgb_1"] = tf.reshape(tf.cast(data["rgb_1"], tf.float32),[height, width, 3])
    data["rgb_2"] = tf.reshape(tf.cast(data["rgb_2"], tf.float32),[height, width, 3])
    return data

def read_and_decode_raft(dataset):
    dataset = dataset.map(decode_image_raft, num_parallel_calls=AUTO)
    dataset = dataset.map(cast_and_reshape_dataset_raft, num_parallel_calls=AUTO)
    return dataset

def normalize_inputs_outputs_raft(data):
    data["rgb_1"] = data["rgb_1"]/255
    data["rgb_2"] = data["rgb_2"]/255
    return data

def crop_rgb(data, min_w=0.9, min_h=0.9):
    #妙な落としかたするとカメラパラメータ変わるので注意
    original_h = tf.cast(data["rgb_height"], tf.float32)
    crop_height = tf.random.uniform((), min_h*original_h, original_h)
    crop_height = tf.cast(crop_height, tf.int32)
    top = tf.random.uniform((), 0, data["rgb_height"]-crop_height, tf.int32)
    bottom = top + crop_height
    
    original_w = tf.cast(data["rgb_width"], tf.float32)
    crop_width = tf.random.uniform((), min_w*original_w, original_w)
    crop_width = tf.cast(crop_width, tf.int32)
    left_r = tf.random.uniform((), 0, data["rgb_width"]-crop_width, tf.int32)
    right_r = left_r + crop_width
    left_l = tf.random.uniform((), 0, data["rgb_width"]-crop_width, tf.int32)
    right_l = left_l + crop_width
    data["rgb_1"] = data["rgb_1"][top:bottom, left_r:right_r, :]
    data["rgb_2"] = data["rgb_2"][top:bottom, left_l:right_l, :]
    data["rgb_width"] = crop_width
    data["rgb_height"] = crop_height
    return data

def resize_rgb(data, input_height=200, input_width=480):
    #disparity to depthも変更する必要あり 
    #height = data["rgb_height"]
    #width = data["rgb_width"]
    
    data["rgb_1"]= tf.image.resize(data["rgb_1"][tf.newaxis, ...], (input_height, input_width), method="bilinear")[0,:,:,:]
    data["rgb_2"]= tf.image.resize(data["rgb_2"][tf.newaxis, ...], (input_height, input_width), method="bilinear")[0,:,:,:]
    #rate_h = tf.cast(input_height, tf.float32)/tf.cast(height, tf.float32)
    #rate_w = tf.cast(input_width, tf.float32)/tf.cast(width, tf.float32)
    
    data["rgb_height"] = input_height
    data["rgb_width"] = input_width
    return data

def assign_input_output_raft(data):
    """
    inputs is 2 rgbs [(batch,) height, width, 4]
    targets is distance btween cars [(batch,) 1]
    """
    inputs = {"frame_1": data["rgb_1"], 
              "frame_2": data["rgb_2"], 
              }
    return inputs#, targets

def assign_input_output_raft_inference(data):
    """
    inputs is 2 rgbs [(batch,) height, width, 4]
    targets is distance btween cars [(batch,) 1]
    """
    inputs = {"frame_1": data["rgb_1"], 
              "frame_2": data["rgb_2"], 
              }
    filenames = {"rgb_file_1": data["rgb_file_1"], 
              "rgb_file_2": data["rgb_file_2"], 
              }
    return inputs, filenames

def get_dataset_raft(list_dataset, 
                       batch_size, 
                       transforms, 
                       input_shape,
                       is_train=True,
                       ):
    print("start building dataset")
    
    # final output's shape is [batch, sequence_length, h, w, ch]
    # so keep data length in advance.
    """
    for i, dataset_single in enumerate(list_dataset):
        dataset_single = build_tf_dataset_depth(dataset_single)
        dataset_single = keep_sequential_length(dataset_single, seq_length)
        if i==0:
            dataset = dataset_single
        else:
            dataset = dataset.concatenate(dataset_single)        
    if is_train:
        dataset = dataset.shuffle(32)
    dataset = dataset.unbatch()
    """
    dataset = build_tf_dataset_raft(list_dataset)
    dataset = read_and_decode_raft(dataset)

    #color系
    dataset = dataset.map(transforms, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: normalize_inputs_outputs_raft(x), num_parallel_calls=AUTO)
    dataset = dataset.map(crop_rgb, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: resize_rgb(x, input_shape[0], input_shape[1]), num_parallel_calls=AUTO)
    
    #dataset = dataset.batch(seq_length, drop_remainder=True)
    #loss設計上意味はなしdataset = dataset.map(random_flip_sequence, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: assign_input_output_raft(x), num_parallel_calls=AUTO)   
    if is_train:
        dataset = dataset.shuffle(64)
        dataset = dataset.repeat()    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_dataset_raft_inference(list_dataset, 
                               batch_size, 
                               input_shape,
                               ):
    print("start building dataset")
    
    dataset = build_tf_dataset_raft(list_dataset)
    dataset = read_and_decode_raft(dataset)

    dataset = dataset.map(lambda x: normalize_inputs_outputs_raft(x), num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: resize_rgb(x, input_shape[0], input_shape[1]), num_parallel_calls=AUTO)
    
    dataset = dataset.map(lambda x: assign_input_output_raft_inference(x), num_parallel_calls=AUTO)   
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)
    
    return dataset



"""
###################
###################
###################
###################
###################
###################
###################
###################
###################

"""











def __load_dataset(files, frame_interval=1, rate=0.8, 
                 load_all=False, 
                 #player_dataset=False, 
                 detection_dataset=False, 
                 #player_classifier_dataset=False, 
                 #load_motion=False, load_depth=False, 
                 use_pseudo_data=False, 
                 #load_jersey_instead_team=False, 
                 skip_few_box=True):
    train_dataset = []
    val_dataset = []
    #player_labels = {"id_counter": 0}
    for i, file in enumerate(files):
        if i<rate*len(files):
            #if player_dataset:#ひっつけない。ここのプレイで分けておく。
            #    data_single_play, player_labels = load_data_player(path=file, frame_interval=frame_interval, load_all=load_all, player_labels=player_labels)
            #    train_dataset += [data_single_play]
            if detection_dataset:
                train_dataset += load_data_detector(path=file, frame_interval=frame_interval)
            #elif player_classifier_dataset:
            #    data_single_play, player_labels = load_data_player(path=file, frame_interval=frame_interval, load_all=load_all, player_labels=player_labels)
            #    train_dataset += data_single_play
            else:
                train_dataset += load_data(path=file, frame_interval=frame_interval, load_all=load_all, 
                                           #load_motion=load_motion, load_depth=load_depth, 
                                           use_pseudo_data=use_pseudo_data, 
                                           #load_jersey_instead_team=load_jersey_instead_team, 
                                           skip_few_box=skip_few_box)
        else:
            #if player_dataset:
            #    data_single_play, player_labels = load_data_player(path=file, frame_interval=frame_interval, load_all=load_all, player_labels=player_labels)
            #    val_dataset += [data_single_play]
            #    #val_dataset += [load_data_player(path=file, frame_interval=frame_interval, load_all=load_all)]
            if detection_dataset:
                val_dataset += load_data_detector(path=file, frame_interval=frame_interval)
            #elif player_classifier_dataset:
            #    data_single_play, player_labels = load_data_player(path=file, frame_interval=frame_interval, load_all=load_all, player_labels=player_labels)
            #    val_dataset += data_single_play
            else:
                val_dataset += load_data(path=file, frame_interval=frame_interval, load_all=load_all, 
                                         #load_motion=load_motion, load_depth=load_depth, 
                                         use_pseudo_data=use_pseudo_data, 
                                         #load_jersey_instead_team=load_jersey_instead_team, 
                                         skip_few_box=skip_few_box)
    #if player_classifier_dataset:
    #    print("num_player is", player_labels["id_counter"]+1)
    #    return train_dataset, val_dataset, player_labels
    return train_dataset, val_dataset



"""
def load_dataset_team_pair(files, frame_interval=1, num_shuffle=20, rate=0.8, load_all=False, player_dataset=False, detection_dataset=False, player_classifier_dataset=False, load_motion=False, load_depth=False):
    train_dataset = []
    for i, file in enumerate(files):
        if i<rate*len(files):
            train_dataset += [load_data(path=file, frame_interval=frame_interval, load_all=load_all, load_motion=load_motion, load_depth=load_depth)]
    shuffle_pair_dataset = []
    for single_play_dataset in train_dataset:
        for _ in range(num_shuffle):
            shuffle_pair_dataset += make_shuffled_pair(single_play_dataset)
    #np.random.shuffle(shuffle_pair_dataset)
    return shuffle_pair_dataset
NUM_PAD = 100

def make_shuffled_pair(single_dataset):
    dataset_1 = single_dataset.copy()
    dataset_2 = single_dataset.copy()
    np.random.shuffle(dataset_1)
    np.random.shuffle(dataset_2)
    pair_dataset = []
    offset = np.array([[[0.0,NUM_PAD+720.0]]])#apply y_offset on second image
    for d_1, d_2 in zip(dataset_1, dataset_2):
        data = {"file_1": d_1["file"],
                "file_2": d_2["file"],
                "rectangles": np.concatenate([d_1["rectangles"],d_2["rectangles"]+offset], axis=0), 
                "locations": np.concatenate([d_1["locations"],d_2["locations"]], axis=0), 
                #"motions": motions,
                "rectangle_num": d_1["rectangle_num"] + d_2["rectangle_num"],
                "team_labels": np.concatenate([d_1["team_labels"],d_2["team_labels"]], axis=0),
                "jersey_labels": np.concatenate([d_1["jersey_labels"],d_2["jersey_labels"]], axis=0),
                "img_height": 720,
                "img_width": 1280,
                }
        #print(data["rectangles"])
        pair_dataset.append(data)
    return pair_dataset
"""
        

def jersey_num_to_label(value):
    #if value>=100:
    #    raise Exception("e")
    label = np.zeros((10))
    ones_place = value%10
    tens_place = value//10
    label[ones_place] = 1
    if tens_place!=0:
        label[tens_place] = 1
    return label
    

def __load_data(num_max=-1, frame_interval=1, path="", 
              load_all=False, load_motion=False, load_depth=False, use_pseudo_data=False,
              load_jersey_instead_team=False, skip_few_box=True):
    rgb_files = sorted(glob.glob(os.path.join(path, "*.jpg")))[::frame_interval]
    annotation_files = sorted(glob.glob(os.path.join(path, "*_train.json")))[::frame_interval]
    annotation_files_test = sorted(glob.glob(os.path.join(path, "*_test.json")))[::frame_interval]
    num_files = len(rgb_files)
    dataset = []
    for i, [rgb_file, ann_file] in enumerate(zip(rgb_files, annotation_files)):
        if i%(num_files//4)==0:
            print("\r----- loading dataset {}/{} -----".format(i+1, num_files), end="")

        ann = json.load(open(annotation_files[i], 'r')) 
        rectangles = np.array(ann["rectangles"], np.float32)
        locations = np.array(ann["location"], np.float32)
        team_labels = np.array(["H" in p for p in ann["player"]]).astype(np.int32)
        t_neg = np.array(["V" in p for p in ann["player"]]).astype(np.int32)
        
        jersey_labels = np.array([jersey_num_to_label(int(p[1:])) for p in ann["player"]]).astype(np.int32)
        rectangle_num = len(rectangles)

        if np.sum(t_neg+team_labels)!=rectangle_num:
            raise Exception("H {} and V {}".format(np.sum(team_labels), np.sum(t_neg)))
        if load_jersey_instead_team:
           team_labels = np.array([t*100 + int(p[1:]) for t,p in zip(team_labels, ann["player"])])
        if skip_few_box:
            if rectangle_num<=3: continue
        data = {"file": rgb_files[i],
                "rectangles": rectangles, 
                "locations": locations, 
                "rectangle_num": rectangle_num,
                "team_labels": team_labels,
                "jersey_labels": jersey_labels,
                "img_height": 720,
                "img_width": 1280,
                }
        if use_pseudo_data:
            data.update({"is_pseudo_label": 0,
                         "team_matrix": tf.ones((rectangle_num**2,), tf.float32),#dummy
                         })
        if load_motion:
            data.update({"motions": np.array(ann["motions"], np.float32)})
            if i!=(len(rgb_files)-1):
                data.update({"file_next": rgb_files[i+1]})
            else:
                data.update({"file_next": rgb_files[i]})
    
            if i!=0:
                data.update({"file_prev": rgb_files[i-1]})
            else:
                data.update({"file_prev": rgb_files[0]})
                
        if load_all:
            ann_t = json.load(open(annotation_files_test[i], 'r')) 
            labeled_player = np.array(ann["player"])
            all_locations = np.array(ann_t["location"], np.float32)
            all_player = np.array(ann_t["player"])

            data.update({"players": labeled_player,
                         "all_players": all_player,
                         "all_locations": all_locations,
                         })
            
            if load_motion:
                data.update({"all_motions": np.array(ann_t["motions"], np.float32)})
        if load_depth:
            data.update({"file_depth": data["file"].replace("train_img", "train_img_depth").replace("jpg","png")})
        dataset.append(data)
    return dataset


def load_data_detector(num_max=-1, frame_interval=1, path=""):
    rgb_files = sorted(glob.glob(os.path.join(path, "*.jpg")))[::frame_interval]
    annotation_files = sorted(glob.glob(os.path.join(path, "*_train.json")))[::frame_interval]
    #hash_file = os.path.join(path, "hash.json")
    #if os.path.exists(hash_file):
    #    hash_survives = json.load(open(hash_file, 'r')) 
    #else:
    #    hash_survives = None
        
    num_files = len(rgb_files)
    dataset = []
    for i, [rgb_file, ann_file] in enumerate(zip(rgb_files, annotation_files)):
        if i%(num_files//4)==0:
            print("\r----- loading dataset {}/{} -----".format(i+1, num_files), end="")
        #if hash_survives is not None:
        #    if not i in hash_survives:
        #        continue
        ann = json.load(open(annotation_files[i], 'r')) 
        rectangles = np.array(ann["rectangles"], np.float32)        
        rectangle_num = len(rectangles)

        data = {"file": rgb_files[i],
                "rectangles": rectangles, 
                "rectangle_num": rectangle_num,
                "img_height": 720,
                "img_width": 1280,
                }

        
        dataset.append(data)
    return dataset

def make_rectangle(df):
    top = df.top.values.reshape(-1)
    left = df.left.values.reshape(-1)
    width = df.width.values.reshape(-1)
    height = df.height.values.reshape(-1)
    bottom = top + height
    right = left + width
    top = top.tolist() 
    left = left.tolist()
    bottom = bottom.tolist()
    right = right.tolist()
    rectangles=[]
    for i in range(len(top)):
        rectangle = [[left[i], top[i]],[right[i], top[i]],
                     [right[i], bottom[i]],[left[i], bottom[i]]]
        rectangles.append(rectangle)
    return rectangles

def load_dataset_helmet_imgs(load_pseudo_data=False,
                             img_path = "data/images/",
                             annotation_path = "data/image_labels.csv",
                             pseudo_path = "remake_data/images/"):
    dataset = []
    df = pd.read_csv(annotation_path)
    #df = df[df["label"]!="Helmet-Sideline"]
    max_box = []
    for img_name, _df in df.groupby("image"):        
        #if np.all(_df["label"]=="Helmet"):
        rectangles = np.array(make_rectangle(_df), np.float32)
        data = {"file": img_path+img_name,
                "rectangles": rectangles, 
                "rectangle_num": len(rectangles),
                "img_height": 720,
                "img_width": 1280,
                }
        
        if load_pseudo_data:
            team_mat_flatten = np.load((pseudo_path+img_name).replace("jpg", "npy"))
            data["team_matrix"] = team_mat_flatten
            #if load_dummy_data:
            loc_pseudo = (pseudo_path+img_name).replace(".jpg", "_loc.npy")
            if os.path.exists(loc_pseudo):
                locations = np.load(loc_pseudo)
            else:
                locations = np.zeros((len(rectangles), 1, 2), float)
            data.update({"locations": locations.reshape(len(rectangles), 1, 2),#dummy 
                        "team_labels": np.ones((len(rectangles), 1), np.int32),#dummy 
                        "jersey_labels": np.ones((len(rectangles), 10), np.int32),#dummy 
                        "is_pseudo_label": 1})
                        
        dataset.append(data)
        max_box.append(len(rectangles))
    print("MAX NUMBER OF BOX ", np.max(max_box))
    return dataset

"""

def load_data_player(num_max=-1, frame_interval=1, path="", load_all=False, player_labels={}):
    game_id = os.path.basename(path).split("_", 1)[0]
    if game_id in player_labels.keys():
        this_game_players = player_labels[game_id]
    else:
        this_game_players = {}
    rgb_files = sorted(glob.glob(os.path.join(path, "*.jpg")))[::frame_interval]
    #annotation_files = sorted(glob.glob(os.path.join(path, "*_train.json")))[::frame_interval]
    #annotation_files_test = sorted(glob.glob(os.path.join(path, "*_test.json")))[::frame_interval]
    num_files = len(rgb_files)
    dataset = []
    for i, rgb_file in enumerate(rgb_files):
        if i%(num_files//4)==0:
            print("\r----- loading dataset {}/{} -----".format(i+1, num_files), end="")
        img_name = os.path.basename(rgb_file)
        label = 1 if img_name[0]=="H" else 0
        player = img_name.split("_", 1)[0]
        jersey_labels = jersey_num_to_label(int(player[1:])).astype(np.int32)
        
        if player in this_game_players:
            p_label = this_game_players[player]
        else:
            p_label = player_labels["id_counter"]
            player_labels["id_counter"] += 1
            this_game_players[player] = p_label

        data = {"file": rgb_file,
                "file_mask": rgb_file.replace(".jpg", "_mask.png"),
                "team_labels": label, 
                "player_labels": p_label, 
                "jersey_labels": jersey_labels,
                "img_height": 96,
                "img_width": 64,
                }
        dataset.append(data)
    player_labels[game_id] = this_game_players
    return dataset, player_labels
"""


def cast_and_reshape_dataset_detection(data):
    height = data["img_height"]
    width = data["img_width"]
    num_box = data["rectangle_num"]
    data["rgb"] = tf.reshape(tf.cast(data["rgb"], tf.float32),[height, width, 3])
    data["rectangles"] = tf.reshape(tf.cast(data["rectangles"], tf.float32),[num_box, 4, 2])
    return data




def box_xycoords_to_tlbr_detection(data, input_shape):
    MAX_NUM_BOX = 74
    
    left_tops = tf.reduce_min(data["rectangles"], axis=1)
    right_bottoms = tf.reduce_max(data["rectangles"], axis=1)
    box_tlbr = tf.concat([left_tops[:,::-1], right_bottoms[:,::-1]], axis=-1)
    box_tlbr = tf.clip_by_value(box_tlbr, tf.zeros((1,4),tf.float32), tf.constant([[input_shape[0],input_shape[1],input_shape[0],input_shape[1]]],tf.float32))
    box_size = (box_tlbr[...,2] - box_tlbr[...,0]) * (box_tlbr[...,3] - box_tlbr[...,1])
    # neglect too small box
    mask = (box_size > 1.0)#input scale 1pix
    box_tlbr = tf.boolean_mask(box_tlbr, mask)

    num_survive = tf.minimum(tf.reduce_sum(tf.cast(mask, tf.int32)), MAX_NUM_BOX)
    data["rectangle_num"] = num_survive
    box_tlbr = box_tlbr[:num_survive]
    box_tlbr = tf.reshape(box_tlbr, [-1,4])
    box_tlbr = tf.pad(box_tlbr, [[0,MAX_NUM_BOX-num_survive],[0,0]], "CONSTANT")
    data["rectangles"] = tf.reshape(box_tlbr, [MAX_NUM_BOX,4])
    return data

def tile_or_pad_boxes(data, max_box_num):
    num_boxes = data["rectangle_num"]
    box_tlbr = data["rectangles"]
    box_location = data["locations"]
    team_labels = data["team_labels"]
    jersey_labels = data["jersey_labels"]
    #削るより増やす方が実装が簡単(batchnormの都合)なので同じものを増やしてしまう。
    # shuffle_gather後にtile
    
    shuffled_idx = tf.random.shuffle(tf.range(num_boxes))
    box_tlbr = tf.gather(box_tlbr, shuffled_idx)
    box_location = tf.gather(box_location, shuffled_idx)
    team_labels = tf.gather(team_labels, shuffled_idx)
    jersey_labels = tf.gather(jersey_labels, shuffled_idx)
    
    num_boxes = tf.minimum(max_box_num, num_boxes)
    shuffled_idx = shuffled_idx[:num_boxes]
    box_tlbr = box_tlbr[:num_boxes]
    box_location = box_location[:num_boxes]
    team_labels = team_labels[:num_boxes]
    jersey_labels = jersey_labels[:num_boxes]
    
    num_tile = tf.math.ceil(tf.cast(max_box_num,tf.float32)/tf.cast(num_boxes,tf.float32))
    box_tlbr = tf.tile(box_tlbr, [num_tile,1])[:max_box_num, :]
    box_location = tf.tile(box_location, [num_tile,1])[:max_box_num, :]
    team_labels = tf.tile(team_labels, [num_tile,1])[:max_box_num, :]
    tiled_mask = tf.cast(tf.range(max_box_num)<num_boxes, tf.float32)
    jersey_labels = tf.tile(jersey_labels, [num_tile,1])[:max_box_num, :]
    
    data["shuffled_idx"] = shuffled_idx
    data["rectangles"] = tf.reshape(box_tlbr, [max_box_num,4])
    data["locations"] = tf.reshape(box_location, [max_box_num,2])
    data["team_labels"] = tf.reshape(team_labels, [max_box_num,1])
    data["tiled_mask"] = tf.reshape(tiled_mask, [max_box_num,1])
    data["jersey_labels"] = tf.cast(tf.reshape(jersey_labels, [max_box_num,10]), tf.float32)
    return data

    
def make_team_matrix(data, conf_thresh=0.05, max_box_num=22):
    if data["is_pseudo_label"]==0:
        team_labels = data["team_labels"]
        tiled_mask = data["tiled_mask"]
        team_label_1 = team_labels[:,tf.newaxis,:]
        team_label_2 = team_labels[tf.newaxis,:,:]
        tiled_mask_1 = tiled_mask[:,tf.newaxis,:]
        tiled_mask_2 = tiled_mask[tf.newaxis,:,:]
        
        team_mat = tf.cast(team_label_1 == team_label_2, tf.float32)
        mask_mat = tf.cast(tiled_mask_1 * tiled_mask_2, tf.float32)
        data["team_similarity"] = tf.concat([team_mat, mask_mat], axis=-1)
    else:
        tiled_mask = data["tiled_mask"]
        team_mat = data["team_matrix"]
        team_mat = tf.gather(tf.gather(team_mat, data["shuffled_idx"]), data["shuffled_idx"], axis=1)    
        num_pad = max_box_num - tf.shape(team_mat)[0]
        team_mat = tf.pad(team_mat, [[0,num_pad],[0,num_pad],[0,0]], "CONSTANT")
        
        tiled_mask_1 = tiled_mask[:,tf.newaxis,:]
        tiled_mask_2 = tiled_mask[tf.newaxis,:,:]
        
        mask_mat = tf.cast(tiled_mask_1 * tiled_mask_2, tf.float32)
        high_conf_mask = tf.cast(tf.logical_or(team_mat>(1.0-conf_thresh), team_mat<conf_thresh), tf.float32)
        mask_mat = mask_mat * high_conf_mask
        
        team_mat_binary = tf.cast(team_mat>0.5, tf.float32)
        data["team_similarity"] = tf.concat([team_mat_binary, mask_mat], axis=-1)
    return data
        

def _assign_input_output(data):

    inputs = {"input_rgb": data["rgb"],
              "input_boxes": data["rectangles"],
              "met_sizes": data["box_size"],
              "gt_points": data["locations"],
              }
    # outputs are dummy
    loss_weight = 1.0 - 0.75 * tf.cast([data["is_pseudo_label"]],tf.float32)
    outputs = {
              "z_error": loss_weight,#tf.ones((1,1), tf.float32),
              "xy_error": loss_weight,#tf.ones((1,1), tf.float32),
              "zoom_dev_abs": loss_weight,#tf.ones((1,1), tf.float32),
              }

    return inputs, outputs

def assign_input_output_team(data):

    inputs = {"input_rgb": data["rgb"],
              "input_boxes": data["rectangles"],
              "met_sizes": data["box_size"],
              "label": data["team_similarity"][:,:,:1],
              }
    outputs = {
              "team_similarity": data["team_similarity"],
              }

    return inputs, outputs

def assign_input_output_detection(data):

    inputs = {"input_rgb": data["rgb"],#batch, 2, h, w, ch
              "inputs_box_tlbr_input_scale": data["rectangles"],
              }
    outputs = {
              "out_offsets": tf.ones((1,1), tf.float32),#dummy
              "out_centerness": tf.ones((1,1), tf.float32),
              }

    return inputs, outputs



def __build_tf_dataset(original_dataset):
    def gen_wrapper(dataset, data_keys=None):
        def generator():
            for data in dataset:
                yield data
        return generator
    dataset = tf.data.Dataset.from_generator(gen_wrapper(original_dataset), output_types={"file": tf.string,
                                                                                          "rectangles": tf.float32, 
                                                                                          "locations": tf.float32,
                                                                                          "rectangle_num": tf.int32,
                                                                                          "img_height": tf.int32,
                                                                                          "img_width": tf.int32,
                                                                                          "team_labels": tf.int32,
                                                                                          "jersey_labels": tf.int32, #not use
                                                                                          "is_pseudo_label": tf.int32,
                                                                                          "team_matrix": tf.float32})
    
    dataset = dataset.map(decode_image, num_parallel_calls=AUTO)
    dataset = dataset.map(cast_and_reshape_dataset, num_parallel_calls=AUTO)
    return dataset



def build_tf_dataset_detection(original_dataset):
    def gen_wrapper(dataset, data_keys=None):
        def generator():
            for data in dataset:
                yield data
        return generator
    dataset = tf.data.Dataset.from_generator(gen_wrapper(original_dataset), output_types={"file": tf.string,
                                                                                          "rectangles": tf.float32, 
                                                                                          #"is_impact": tf.int32,
                                                                                          "rectangle_num": tf.int32,
                                                                                          "img_height": tf.int32,
                                                                                          "img_width": tf.int32,
                                                                                          })
    
    dataset = dataset.map(decode_image, num_parallel_calls=AUTO)
    dataset = dataset.map(cast_and_reshape_dataset_detection, num_parallel_calls=AUTO)
    return dataset


def normalize_inputs_outputs_pseudo(data, input_shape):
    h, w = tf.unstack(tf.shape(data["rgb"]))[:2]
    data["rgb"] = data["rgb"]/255
    if data["is_pseudo_label"]==0:
        data["locations"] = data["locations"]/20#) - 1.0
    data["rectangles"] = data["rectangles"]/tf.cast(tf.stack([[[w,h]]]),tf.float32)
    return data

def normalize_inputs_outputs_detection(data, input_shape):
    h, w = input_shape[:2]
    data["rgb"] = data["rgb"]/255
    return data

def get_met_size(data):
    boxes = data["rectangles"]
    box_heights = boxes[:,2] - boxes[:,0]
    box_widths = boxes[:,3] - boxes[:,1]
    box_size = tf.reduce_mean(tf.math.sqrt(box_heights * box_widths), axis=0, keepdims=True)
    data["box_size"] = box_size
    return data



def cut_mix(data_1, data_2):
    img_height, img_width, _ = tf.unstack(tf.shape(data_1["rgb"]))
    crop_h = tf.random.uniform((), 0, img_height, tf.int32)
    crop_w = tf.random.uniform((), 0, img_width, tf.int32)
    horizontal_split = tf.random.uniform(())>0.5
    def mix_img(img_1, img_2):
        if horizontal_split:
            mixed_img = tf.concat([img_1[:crop_h,:,:], img_2[crop_h:,:,:]], axis=0)
        else:
            mixed_img = tf.concat([img_1[:,:crop_w,:], img_2[:,crop_w:,:]], axis=1)
        return mixed_img
    def mix_labels(boxes_1, boxes_2):
        if horizontal_split:
            boxes_1 = tf.minimum(boxes_1, tf.stack([[[1.0e7, tf.cast(crop_h, tf.float32)]]]))
            boxes_2 = tf.maximum(boxes_2, tf.stack([[[-1.0e7, tf.cast(crop_h, tf.float32)]]]))
            #mixed_boxes = tf.concat([boxes_1, boxes_2], axis=0)
        else:
            boxes_1 = tf.minimum(boxes_1, tf.stack([[[tf.cast(crop_w, tf.float32), 1.0e7]]]))
            boxes_2 = tf.maximum(boxes_2, tf.stack([[[tf.cast(crop_w, tf.float32), -1.0e7]]]))
        mixed_boxes = tf.concat([boxes_1, boxes_2], axis=0)
        return mixed_boxes
    data_1["rectangles"] = mix_labels(data_1["rectangles"], data_2["rectangles"])
    data_1["rgb"] = mix_img(data_1["rgb"], data_2["rgb"])
    return data_1


def get_dataset_team_map(list_dataset, 
                  input_shape, 
                  output_shape,
                  batch_size, 
                  transforms, 
                  max_box_num=22,
                  is_train=True,
                  only_team=False,
                  ):
    print("start building dataset")
    dataset = build_tf_dataset(list_dataset)
        
    dataset = dataset.map(transforms, num_parallel_calls=AUTO)
    dataset = dataset.filter(lambda x: x["rectangle_num"] > 3)
    
    dataset = dataset.map(lambda x: normalize_inputs_outputs_pseudo(x, input_shape), num_parallel_calls=AUTO)
    
    dataset = dataset.map(box_xycoords_to_tlbr, num_parallel_calls=AUTO)
    dataset = dataset.filter(lambda x: x["rectangle_num"] > 3)
    dataset = dataset.map(lambda x: tile_or_pad_boxes(x, max_box_num), num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x: make_team_matrix(x, max_box_num=max_box_num), num_parallel_calls=AUTO)
    
    dataset = dataset.map(get_met_size, num_parallel_calls=AUTO)
    
    if only_team:
        dataset = dataset.map(assign_input_output_team, num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(assign_input_output, num_parallel_calls=AUTO)
    
    if is_train:
        dataset = dataset.shuffle(128)
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
    else:# cache sometimes causes memory error
        #dataset = dataset.cache()
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    #"""
    
    return dataset


def get_dataset_detection(list_dataset, 
                  input_shape, 
                  output_shape,
                  batch_size, 
                  transforms, 
                  max_box_num=20,
                  is_train=True,
                  use_cut_mix=False):
    print("start building dataset")
    dataset = build_tf_dataset_detection(list_dataset)
    dataset = dataset.map(transforms, num_parallel_calls=AUTO)
    #dataset = dataset.filter(lambda x: x["rectangle_num"] > 3)
    dataset = dataset.map(lambda x: normalize_inputs_outputs_detection(x, input_shape), num_parallel_calls=AUTO)
    
    
    if is_train and use_cut_mix:
        mix_dataset = list_dataset.copy()
        np.random.shuffle(mix_dataset)
        mix_dataset = build_tf_dataset_detection(mix_dataset)
        mix_dataset = mix_dataset.map(transforms, num_parallel_calls=AUTO)
        mix_dataset = mix_dataset.map(lambda x: normalize_inputs_outputs_detection(x, input_shape), num_parallel_calls=AUTO)
        dataset = tf.data.Dataset.zip((dataset, mix_dataset))
        dataset = dataset.map(cut_mix, num_parallel_calls=AUTO)
   
    dataset = dataset.map(lambda x: box_xycoords_to_tlbr_detection(x, input_shape), num_parallel_calls=AUTO)
    dataset = dataset.map(assign_input_output_detection, num_parallel_calls=AUTO)
    
    if is_train:
        dataset = dataset.shuffle(128)
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
    else:# cache sometimes causes memory error
        dataset = dataset.cache()           
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    
    return dataset


    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    """
    images = tf.ones([1,100,100,1], tf.float32)
    images = tf.pad(images,[[0,0],[0,100],[0,100],[0,0]])
    boxes = tf.constant([[0.2,0.2,0.6,0.6]], tf.float32)
    h_scale = 1. / (boxes[:,2:3] - boxes[:,0:1])
    w_scale = 1. / (boxes[:,3:4] - boxes[:,1:2])
    inv_tops = -boxes[:,:1] * h_scale
    inv_lefts = -boxes[:,1:2] * w_scale
    inv_bottoms = 1. + (1. - boxes[:,2:3]) * h_scale
    inv_rights = 1. + (1. - boxes[:,3:4]) * w_scale
    inv_boxes = tf.concat([inv_tops, inv_lefts, inv_bottoms, inv_rights], axis=1)
    box_indices = [0]
    crop_size = [150,150]
    crop_images = tf.image.crop_and_resize(images, boxes, box_indices, crop_size, method='bilinear')
    plt.imshow(images[0])
    plt.show()
    plt.imshow(crop_images[0])
    plt.show()
    reconst_size = [200,200]
    reconst_images = tf.image.crop_and_resize(crop_images, inv_boxes, box_indices, reconst_size, method='bilinear')
    plt.imshow(reconst_images[0])
    plt.show()
    
    
    
    raise Exception()
    
    player_ids = tf.constant([1,3,5,7,9,11])
    pair_labels = tf.constant([[1,3,1],
                               [3,1,1],
                               [2,1,0],
                               [1,2,0],
                               [1,11,0],
                               [11,1,0],
                               [4,6,0],
                               [6,4,0],
                               [4,7,1],
                               [7,4,1],
                               [9,11,1],
                               [11,9,1],
                               ])
    print(pair_labels_to_matrix(player_ids, pair_labels))
    print(tf.where(pair_labels_to_matrix(player_ids, pair_labels)==0))
    raise Exception()
    """
    path = "../input_preprocess/train_img/58168_003392_Endzone/"
    dataset = load_data(path)
    
    import sys
    import os
    sys.path.append("../")
    from train_utils.tf_Augmentations_detection import Compose, Oneof, HorizontalFlip, VerticalFlip, Crop, Center_Crop, Resize, BrightnessContrast, CoarseDropout, HueShift, ToGlay, Blur, PertialBrightnessContrast, Shadow, GaussianNoise, Rotation
    from train_utils.tf_Augmentations_detection import Center_Crop_by_box_shape, Crop_by_box_shape


    transforms_train = [
                  HorizontalFlip(p=0.5),# not active for jersey classifier
                  Crop(p=1, min_height=550, min_width=1000),
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
    

    train_transforms = Compose(transforms_train)

    
    
    tf_dataset = get_tf_dataset(dataset[::5],
                                transforms=train_transforms)
    for d in tf_dataset.take(20):
        print("d")
        print(d["player_id"], d["contact_pairlabels"])
        print(d["contact_pairlabels_indices"])
        print(d["num_labels"])
        print(d["num_player"])
        print(d["rectangles"])
        plt.imshow(d["rgb"])
        plt.title(d["contact_pairlabels"].numpy())
        plt.show()


