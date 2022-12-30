# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 22:01:23 2022

@author: kmat
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
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv1D, Conv2D, Conv2DTranspose, BatchNormalization, Activation, GlobalAveragePooling2D, Lambda, Input, Concatenate, Add, UpSampling2D, LeakyReLU, ZeroPadding2D,Multiply, DepthwiseConv2D, MaxPooling2D, LayerNormalization
from tensorflow.keras.layers import GRU, Bidirectional
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class GraphConv(Layer):
    """
    Graph convolution layer.
    """
    def __init__(self, 
                 input_ch, 
                 output_ch, 
                 num_adj=1,
                 #placeholders, dropout=0.,
                 #activation="relu", 
                 **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.num_adj = num_adj
        #self.activation = Activation(activation, name="act")

    def build(self, input_shape):
        self.kernels = []
        for i in range(self.num_adj):
            self.kernels.append(self.add_weight(f"kernel_{i}",
                                          shape=[self.input_ch,
                                                 self.output_ch]))
            
    def call(self, inputs):
        """
        x [batch, num_node, features]
        adjacency_mats [batch, num_mat, num_node, num_node])
        """
        x, adjacency_mats = inputs
        num_node, num_features = tf.unstack(tf.shape(x)[-2:])
        outputs = []
        for i in range(self.num_adj):
            adjacency_mat = adjacency_mats[:,i]
            kernel = self.kernels[i]
            
            eye = tf.eye(num_node, dtype=adjacency_mat.dtype)
            diag_mat = tf.reduce_sum(adjacency_mat, axis=1, keepdims=True) * eye[tf.newaxis,:,:]
            inv_diag_mat = 1./(diag_mat+1e-7) * tf.cast(diag_mat>0, tf.float32)
            
            #inv_diag_mat = tf.linalg.diag(1./(1e-7+tf.reduce_sum(adjacency_mat, axis=1, keepdims=False)))
            
            out = tf.matmul(inv_diag_mat, tf.matmul(adjacency_mat, #eyeは足されてる前提と思う方がよさそう。 
                                                       tf.matmul(x, kernel)))
            #out = tf.matmul(x, kernel)
            outputs.append(out)
        outputs = tf.add_n(outputs)
        return outputs #self.activation(outputs)

class GRBBlock(Layer):
    """
    graph conv -> batch norm -> relu
    
    inputs is list(features, list(adj_matrix))
    """
    def __init__(self, 
                 input_ch, 
                 output_ch, #placeholders, dropout=0.,
                 num_adj=1,
                 activation="relu", 
                 use_batchnorm=True,
                 **kwargs):
        super(GRBBlock, self).__init__(**kwargs)
        self.g = GraphConv(input_ch, output_ch, num_adj)
        if use_batchnorm:
            self.b = BatchNormalization()
        self.use_batchnorm = use_batchnorm
        self.r = Activation("relu")
        
    def call(self, inputs):
        x = self.g(inputs)
        if self.use_batchnorm:
            x = self.b(x)
        x = self.r(x)
        return x


class TransposedConv1D(Conv1D):
        
    def call(self, inputs):
        inputs_trans = tf.transpose(inputs, [0,2,1,3]) # batch, player, step, features
        outputs = super(TransposedConv1D, self).call(inputs_trans)
        outputs_trans = tf.transpose(outputs, [0,2,1,3]) # batch, step, player, features
        return outputs_trans

class TransposedConv1D_t2(Conv1D):
    def call(self, inputs):
        inputs_trans = tf.transpose(inputs, [0,2,3,1,4]) # batch, player, step, features
        outputs = super(TransposedConv1D_t2, self).call(inputs_trans)
        outputs_trans = tf.transpose(outputs, [0,3,1,2,4]) # batch, step, player, features
        return outputs_trans

def trans1d_cbr(x, out_layer, kernel, stride, name, bias=False, dilation=1, num_trans=1):
    if num_trans==1:
        TConv = TransposedConv1D
    elif num_trans==2:
        TConv = TransposedConv1D_t2
    else:
        raise Exception()
    x = TConv(out_layer, kernel_size=kernel, strides=stride, use_bias=bias, padding="same", dilation_rate=dilation, name=name+"_conv")(x)
    x = BatchNormalization(name=name+"_bw")(x)
    x = Activation("relu",name=name+"_activation")(x)
    return x

    
def pairwise_feature_extractor(inputs):
    """
    
    features:
        [batch, num_player, num_features]
    pairs:
        [batch, num_pairs, 2]
        values are index of player. 0 is NOT ground.
    concat_feature:
        player contact [batch, num_pairs, 2x num_features]
        (ground contact [batch, num_player, num_features] (same as the inputs))
    """
    features, pairs = inputs
    player_feature_1 = tf.gather(features, pairs[:,:,0], axis=1, batch_dims=1)
    player_feature_2 = tf.gather(features, pairs[:,:,1], axis=1, batch_dims=1)
    concat_feature = tf.concat([player_feature_1, player_feature_2], axis=-1)
    return concat_feature

def grid_pair_concatenate(inputs):
    """
    inputs:
        [batch, num_player, num_features]
    outputs:
        [batch, num_player, num_player, num_features]
    """
    num_player = tf.shape(inputs)[-2]
    inputs_rank = tf.rank(inputs)
    player_feature_1 = tf.tile(inputs[...,:,tf.newaxis,:], 
                               tf.concat([tf.tile([1], [inputs_rank-3]), tf.stack([1,1,num_player,1])], axis=0))
    player_feature_2 = tf.tile(inputs[...,tf.newaxis,:,:],
                               tf.concat([tf.tile([1], [inputs_rank-3]), tf.stack([1,num_player,1,1])], axis=0))
    concat_feature = tf.concat([player_feature_1, player_feature_2], axis=-1)
    return concat_feature

def grid_pair_multiply(inputs):
    """
    inputs:
        [batch, num_player, num_features]
    outputs:
        [batch, num_player, num_player, num_features]
    """
    num_player = tf.shape(inputs)[-2]
    inputs_rank = tf.rank(inputs)
    player_feature_1 = tf.tile(inputs[...,:,tf.newaxis,:], 
                               tf.concat([tf.tile([1], [inputs_rank-3]), tf.stack([1,1,num_player,1])], axis=0))
    player_feature_2 = tf.tile(inputs[...,tf.newaxis,:,:],
                               tf.concat([tf.tile([1], [inputs_rank-3]), tf.stack([1,num_player,1,1])], axis=0))
    concat_feature = player_feature_1 * player_feature_2
    return concat_feature

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

def masked_bce_loss(y_true, y_pred):
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
    
def build_gcn(num_player, num_input_feature, num_adj):
    input_features = Input(shape=(num_player, num_input_feature),name="input_features")
    adj_mats = Input(shape=(num_adj, num_player, num_player),name="input_adjacency_matrix")
    #input_adjacency_dist = Input(shape=(num_player, num_player),name="input_adjacency_dist")
    #input_adjacency_team = Input(shape=(num_player, num_player),name="input_adjacency_sameteam")
    ### input_pairs = Input(shape=(None, 2),name="input_pairs", dtype=tf.int32)
    #adj_mats = [input_adjacency_dist, input_adjacency_team]
    #adj_mats = Lambda(lambda x: tf.stack(x, axis=-1))(adjs)
    #num_adj = len(adj_mats)
    # つくれる。input_adjacency_team = Input(shape=(num_player, num_player),name="input_adjacency_diffteam")
    
    x = input_features
    x = GRBBlock(input_ch=num_input_feature, output_ch=32, num_adj=num_adj, activation="relu", name="grb_0")([x, adj_mats])
    x = GRBBlock(input_ch=32, output_ch=32, num_adj=num_adj, activation="relu", name="grb_1")([x, adj_mats])
    x = GRBBlock(input_ch=32, output_ch=32, num_adj=num_adj, activation="relu", name="grb_2")([x, adj_mats])
    
    g_contact_feature = x # [batch, num_pairs, 2x num_features]
    ### p_contact_feature = pairwise_feature_extractor([x, input_pairs]) # [batch, num_pairs, num_features]
    p_contact_feature = Lambda(lambda x: grid_pair_concatenate(x))(x)
    
    g_contact_feature = Dropout(0.2, name='g_drop')(g_contact_feature)
    p_contact_feature = Dropout(0.2, name='p_drop')(p_contact_feature)
    
    g_contact = Dense(1, activation="sigmoid", name="g_contact")(g_contact_feature)
    p_contact = Dense(1, activation="sigmoid", name="p_contact")(p_contact_feature)

    inputs = [input_features, adj_mats]#input_adjacency_dist, input_adjacency_team]#, input_pairs]
    outputs = [g_contact, p_contact]
    losses = {"g_contact": masked_bce_loss,
              "p_contact": masked_bce_loss,
              }
    loss_weights = {"g_contact": 1.,
                    "p_contact": 1.,
                    }
    metrics = {"g_contact": [matthews_correlation_best],
               "p_contact": [matthews_correlation_best],
               }
    
    
    
    model = Model(inputs, outputs)
    # 時間軸作る場合は時間でたたむ、グラフで統合を繰り返してはどうか？
    return model, losses, loss_weights, metrics


def build_gcn_1dcnn(num_player, num_input_feature, num_adj):
    input_features = Input(shape=(None, num_player, num_input_feature),name="input_features")
    input_adj_mats = Input(shape=(num_adj, None, num_player, num_player),name="input_adjacency_matrix")
    input_step_range = Input(shape=(None, 1),name="step_range")

    #input_adjacency_dist = Input(shape=(None, num_player, num_player),name="input_adjacency_dist")
    #input_adjacency_team = Input(shape=(None, num_player, num_player),name="input_adjacency_sameteam")
    ### input_pairs = Input(shape=(None, 2),name="input_pairs", dtype=tf.int32)
    #adj_mats = [input_adjacency_dist, input_adjacency_team]
    #adj_mats = Lambda(lambda x: tf.stack(x, axis=-1))(adjs)
    #num_adj = len(adj_mats)
    # つくれる。input_adjacency_team = Input(shape=(num_player, num_player),name="input_adjacency_diffteam")
    
    #x = input_features
    x = Lambda(lambda x: tf.concat([x[0], tf.broadcast_to(x[1][...,tf.newaxis], tf.shape(x[0]))], axis=-1))([input_features, input_step_range])
    
    
    k = 9#9#5#9
    ks = None#5
    filters = 64*2
    x = trans1d_cbr(x, filters, kernel=k, stride=1, name="cbr_1")
    if ks is not None:
        xs = trans1d_cbr(x, filters, kernel=ks, stride=1, dilation=3, name="cbr_s1")
        x = Lambda(lambda x: tf.concat(x, axis=-1))([x, xs])
    #x = GRBBlock(input_ch=32, output_ch=32, num_adj=num_adj, activation="relu", name="grb_0")([x, input_adj_mats])
    #x = trans1d_cbr(x, 64, kernel=3, stride=1, name="cbr_2")
    #x = GRBBlock(input_ch=64, output_ch=64, num_adj=num_adj, activation="relu", name="grb_1")([x, input_adj_mats])
    x = trans1d_cbr(x, filters, kernel=k, stride=1, name="cbr_3")
    if ks is not None:
        xs = trans1d_cbr(x, filters, kernel=ks, stride=1, dilation=3, name="cbr_s3")
        x = Lambda(lambda x: tf.concat(x, axis=-1))([x, xs])
    #x = GRBBlock(input_ch=64, output_ch=64, num_adj=num_adj, activation="relu", name="grb_2")([x, input_adj_mats])
    x = trans1d_cbr(x, filters, kernel=k, stride=1, name="cbr_4")
    if ks is not None:
        xs = trans1d_cbr(x, filters, kernel=ks, stride=1, dilation=3, name="cbr_s4")
        x = Lambda(lambda x: tf.concat(x, axis=-1))([x, xs])
    
    g_contact_feature = x # [batch, num_pairs, 2x num_features]
    ### p_contact_feature = pairwise_feature_extractor([x, input_pairs]) # [batch, num_pairs, num_features]
    ## p_contact_feature = Lambda(lambda x: grid_pair_concatenate(x))(x)
    p_contact_feature = Lambda(lambda x: grid_pair_multiply(x))(x)
    
    
    
    filters_adj=64#*2
    kadj = 5
    adj_features = Lambda(lambda x: tf.transpose(x, [0,2,3,4,1]))(input_adj_mats)
    adj_features_p = Dense(filters_adj, activation="relu", name="adj_featuresD1")(adj_features)
    adj_features_p = Dense(filters_adj, activation="relu", name="adj_featuresD2")(adj_features_p)
    #adj_features_g = Dense(filters_adj, activation="relu", name="adj_featuresD3")(adj_features)
    #adj_features_g = Dense(filters_adj, activation="relu", name="adj_featuresD4")(adj_features_g)
    
    #"""
    adj_features = trans1d_cbr(adj_features, filters_adj, kernel=kadj, stride=1, name="cbr_adj1", num_trans=2)
    #adj_features = trans1d_cbr(adj_features, filters_adj, kernel=k, stride=1, name="cbr_adj2", num_trans=2)
    #adj_features_p = trans1d_cbr(adj_features, filters_adj, kernel=kadj, stride=1, name="cbr_adj3p", num_trans=2)
    adj_features_g = trans1d_cbr(adj_features, filters_adj, kernel=kadj, stride=1, name="cbr_adj3g", num_trans=2)
    #"""
    g_contact_feature = Lambda(lambda x: tf.concat([x[0], tf.reduce_mean(x[1], axis=2)], axis=-1))([g_contact_feature, adj_features_g])
    p_contact_feature = Lambda(lambda x: tf.concat(x, axis=-1))([p_contact_feature, adj_features_p])
    
    g_contact_feature = Dropout(0.25, name='g_drop')(g_contact_feature)
    p_contact_feature = Dropout(0.25, name='p_drop')(p_contact_feature)
    
    g_contact = Dense(1, activation="sigmoid", name="g_contact")(g_contact_feature)
    p_contact = Dense(1, activation="sigmoid", name="p_contact")(p_contact_feature)

    inputs = [input_features, input_adj_mats, input_step_range]#input_adjacency_dist, input_adjacency_team]#, input_pairs]
    outputs = [g_contact, p_contact]
    losses = {"g_contact": masked_bce_loss,
              "p_contact": masked_bce_loss,
              }
    loss_weights = {"g_contact": 1.,
                    "p_contact": 1.,
                    }
    metrics = {"g_contact": [matthews_correlation_best],
               "p_contact": [matthews_correlation_best],
               }
    
    
    model = Model(inputs, outputs)
    # 時間軸作る場合は時間でたたむ、グラフで統合を繰り返してはどうか？
    return model, losses, loss_weights, metrics

def build_dense(num_player, num_input_feature, num_adj):
    input_features = Input(shape=(None, num_player, num_input_feature),name="input_features")
    input_adj_mats = Input(shape=(num_adj, None, num_player, num_player),name="input_adjacency_matrix")

    #input_adjacency_dist = Input(shape=(None, num_player, num_player),name="input_adjacency_dist")
    #input_adjacency_team = Input(shape=(None, num_player, num_player),name="input_adjacency_sameteam")
    ### input_pairs = Input(shape=(None, 2),name="input_pairs", dtype=tf.int32)
    #adj_mats = [input_adjacency_dist, input_adjacency_team]
    #adj_mats = Lambda(lambda x: tf.stack(x, axis=-1))(adjs)
    #num_adj = len(adj_mats)
    # つくれる。input_adjacency_team = Input(shape=(num_player, num_player),name="input_adjacency_diffteam")
    
    x = input_features
    
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    
    g_contact_feature = x
    #p_contact_feature = Lambda(lambda x: grid_pair_concatenate(x))(x)
    p_contact_feature = Lambda(lambda x: grid_pair_multiply(x))(x)
    
    g_contact_feature = Dropout(0.2, name='g_drop')(g_contact_feature)
    p_contact_feature = Dropout(0.2, name='p_drop')(p_contact_feature)
    
    g_contact = Dense(1, activation="sigmoid", name="g_contact")(g_contact_feature)
    p_contact = Dense(1, activation="sigmoid", name="p_contact")(p_contact_feature)

    inputs = [input_features, input_adj_mats]#input_adjacency_dist, input_adjacency_team]#, input_pairs]
    outputs = [g_contact, p_contact]
    losses = {"g_contact": masked_bce_loss,
              "p_contact": masked_bce_loss,
              }
    loss_weights = {"g_contact": 1.,
                    "p_contact": 1.,
                    }
    metrics = {"g_contact": [matthews_correlation_best],
               "p_contact": [matthews_correlation_best],
               }
    
    
    model = Model(inputs, outputs)
    # 時間軸作る場合は時間でたたむ、グラフで統合を繰り返してはどうか？
    return model, losses, loss_weights, metrics    

class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
    
    

"""

player_time_features = ["x_position",
                   "y_position",
                   "speed",
                   "direction",
                   "orientation",# distance???
                   "acceleration"]
player_features = ["team",
                   ]
for gp in labels["game_play"].unique():
    gp_track = tr_tracking[tr_tracking["game_play"]==gp]
    gp_labels = labels[labels["game_play"]==gp]
    
    min_step_track, max_step_track = gp_track["step"].min(), gp_track["step"].max()
    min_step_label, max_step_label = gp_labels["step"].min(), gp_labels["step"].max()
    min_step = max(min_step_track, min_step_label)
    max_step = min(max_step_track, max_step_label)
    
    gp_track = gp_track[np.logical_and(gp_track["step"]>=min_step, gp_track["step"]<=max_step)]
    gp_labels = gp_labels[np.logical_and(gp_labels["step"]>=min_step, gp_labels["step"]<=max_step)]
    gp_track["nfl_player_id"] = gp_track["nfl_player_id"].astype(str)
    gp_labels["nfl_player_id_1"] = gp_labels["nfl_player_id_1"].astype(str)
    gp_labels["nfl_player_id_2"] = gp_labels["nfl_player_id_2"].astype(str)
    
    unique_players = list(gp_track["nfl_player_id"].unique())#
    player_id2idx = {str(pid): num for num,pid in enumerate(unique_players + ["G"])}
    gp_track["nfl_player_idx"] = gp_track["nfl_player_id"].astype(str).map(player_id2idx).astype(int)
    gp_labels["nfl_player_idx_1"] = gp_labels["nfl_player_id_1"].astype(str).map(player_id2idx).astype(int)
    gp_labels["nfl_player_idx_2"] = gp_labels["nfl_player_id_2"].astype(str).map(player_id2idx).astype(int)
    #player_team2idx = {"away":1, "home":0}
    
    #player_and_team = gp_track.groupby("nfl_player_idx")["team"].head(1)
    

    #gp_labels_ground = gp_labels[gp_labels["nfl_player_idx_2"]==(len(unique_players)-1)]
    num_step = (max_step - min_step) + 1
    num_players = len(unique_players)
    gp_data_for_gcn =  {#"label_g_contact": - np.ones((num_step, num_players), np.int32), # [num_step, players] player ground contact. (training target)
      #"label_p_contact": - np.ones((num_step, num_players, num_players), np.int32), # [num_step, players, players] player pairs contact. (training target)
      "label_contact": - np.ones((num_step, num_players+1, num_players+1), np.int32),
      "p2p_adj_dist_matrix": - np.ones((num_step, num_players, num_players), np.float32), # [num_frame, players, players] yard distance btw players
      "p2p_adj_team_matrix": - np.ones((num_players, num_players), np.int32), # [players, players] to show same team (-> diff team)
      ##"player_2d_num_matrix": - np.ones((num_players, num_features_numeric), np.float32), # [players, numerical_features] tbd
      ##"player_2d_cat_matrix": - np.ones((num_players, num_features_categorical), np.int32), # [players, categorical_features] position 
      "player_3d_num_matrix": - np.ones((num_step, num_players, len(player_time_features)), np.int32), # [num_frame, players, numerical_features] speed, 
      "step_range": np.arange(min_step, max_step+1, dtype=np.int32),
      }# [num_step] frame number of targets
    player_and_team = dict(gp_track.groupby("nfl_player_id")["nfl_player_id","team"].head(1).values)#set_index("nfl_player_id").to_dict()["team"]
    player_teams = [player_and_team[str(pid)] for pid in unique_players]
    team_matrix = (np.array(player_teams)[:,np.newaxis] == np.array(player_teams)[np.newaxis, :]).astype(int)
    gp_data_for_gcn["p2p_adj_team_matrix"] = team_matrix
    
    gp_data_for_gcn["label_contact"][gp_labels["step"].values,
                                     gp_labels["nfl_player_idx_1"].values,
                                     gp_labels["nfl_player_idx_2"].values,
                                     ] = gp_labels["contact"].values
    gp_data_for_gcn["label_p_contact"] = gp_data_for_gcn["label_contact"][:,:-1,:-1]
    gp_data_for_gcn["label_g_contact"] = gp_data_for_gcn["label_contact"][:,:-1,-1]
    
    gp_data_for_gcn["player_3d_num_matrix"][gp_track["step"].values,
                                            gp_track["nfl_player_idx"].values,
                                            ] = gp_track[player_time_features].values
    
    
    xy = gp_data_for_gcn["player_3d_num_matrix"][:,:,:2]
    gp_data_for_gcn["p2p_adj_dist_matrix"] = np.sqrt(np.sum((xy[:,np.newaxis,:,:] - xy[:,:,np.newaxis,:])**2, axis=-1))
    
    del gp_data_for_gcn["label_contact"]
    path = f"gp_data_for_gcn/{gp}"
    for key, val in gp_data_for_gcn.items():
        print(key, val.shape)
        #save_path = os.path.join(path, f"{key}.npy")
        #os.makedirs(save_path, exist_ok=True)
        #np.save(save_path, val)
        
        
        
        



    
    
    
    
    
    
player_time_features = ["x_position",
                   "y_position",
                   "speed",
                   "direction",
                   "orientation",# distance???
                   "acceleration"]
player_features = ["team",
                   ]
for gp in labels["game_play"].unique():
    gp_track = tr_tracking[tr_tracking["game_play"]==gp]
    gp_labels = labels[labels["game_play"]==gp]
    
    min_step_track, max_step_track = gp_track["step"].min(), gp_track["step"].max()
    min_step_label, max_step_label = gp_labels["step"].min(), gp_labels["step"].max()
    min_step = max(min_step_track, min_step_label)
    max_step = min(max_step_track, max_step_label)
    
    gp_track = gp_track[np.logical_and(gp_track["step"]>=min_step, gp_track["step"]<=max_step)]
    gp_labels = gp_labels[np.logical_and(gp_labels["step"]>=min_step, gp_labels["step"]<=max_step)]
    
    unique_players = list(gp_track["nfl_player_id"].unique())#
    player_id2idx = {pid: num for num,pid in enumerate(unique_players + ["G"])}
    gp_track["nfl_player_idx"] = gp_track["nfl_player_id"].map(player_id2idx).astype(int)
    gp_labels["nfl_player_idx_1"] = gp_labels["nfl_player_id_1"].map(player_id2idx).astype(int)
    gp_labels["nfl_player_idx_2"] = gp_labels["nfl_player_id_2"].map(player_id2idx).astype(int)
    #player_team2idx = {"away":1, "home":0}
    
    #player_and_team = gp_track.groupby("nfl_player_idx")["team"].head(1)
    

    #gp_labels_ground = gp_labels[gp_labels["nfl_player_idx_2"]==(len(unique_players)-1)]
    num_step = (max_step - min_step) + 1
    num_players = len(unique_players)
    gp_data_for_gcn =  {#"label_g_contact": - np.ones((num_step, num_players), np.int32), # [num_step, players] player ground contact. (training target)
      #"label_p_contact": - np.ones((num_step, num_players, num_players), np.int32), # [num_step, players, players] player pairs contact. (training target)
      "label_contact": - np.ones((num_step, num_players+1, num_players+1), np.int32),
      "p2p_adj_dist_matrix": - np.ones((num_step, num_players, num_players), np.float32), # [num_frame, players, players] yard distance btw players
      "p2p_adj_team_matrix": - np.ones((num_players, num_players), np.int32), # [players, players] to show same team (-> diff team)
      ##"player_2d_num_matrix": - np.ones((num_players, num_features_numeric), np.float32), # [players, numerical_features] tbd
      ##"player_2d_cat_matrix": - np.ones((num_players, num_features_categorical), np.int32), # [players, categorical_features] position 
      "player_3d_num_matrix": - np.ones((num_step, num_players, len(player_time_features)), np.int32), # [num_frame, players, numerical_features] speed, 
      "step_range": np.arange(min_step, max_step+1, dtype=np.int32),
      }# [num_step] frame number of targets
    print("making")
    player_and_team = gp_track.groupby("nfl_player_idx")["team"].head(1).to_dict()
    player_teams = [player_and_team[id] for id in unique_players]
    team_matrix = (np.array(player_teams)[:,np.newaxis] == np.array(player_teams)[np.newaxis, :]).astype(int)
    gp_data_for_gcn["p2p_adj_team_matrix"] = team_matrix
    
    gp_data_for_gcn["label_contact"][gp_labels["step"].values,
                                     gp_labels["nfl_player_idx_1"].values,
                                     gp_labels["nfl_player_idx_2"].values,
                                     ] = gp_labels["contact"].values
    gp_data_for_gcn["label_p_contact"] = gp_data_for_gcn["label_contact"][:-1,:-1]
    gp_data_for_gcn["label_p_contact"] = gp_data_for_gcn["label_contact"][:-1,-1:]
    
    gp_data_for_gcn["player_3d_num_matrix"][gp_track["step"].values,
                                            gp_track["nfl_player_idx"].values,
                                            ] = gp_track[player_time_features].values
    
    
    xy = gp_data_for_gcn["player_3d_num_matrix"][:,:,:2]
    gp_data_for_gcn["p2p_adj_dist_matrix"] = np.sqrt(np.sum((xy[:,np.newaxis,:,:] - xy[:,:,np.newaxis,:])**2, axis=-1))
    
    
    path = f"gp_data_for_gcn/{game_play}"
    for key, val in gp_data_for_gcn.items():
        print(key, val.shape)
        #save_path = os.path.join(path, f"{key}.npy")
        #os.makedirs(save_path, exist_ok=True)
        #np.save(save_path, val)
        



player_time_features = ["x_position",
                   "y_position",
                   "speed",
                   "direction",
                   "orientation",# distance???
                   "acceleration"]
player_features = ["team",
                   ]
for gp in labels["game_play"].unique():
    gp_track = tr_tracking[tr_tracking["game_play"]==gp]
    gp_labels = labels[labels["game_play"]==gp]
    
    min_step_track, max_step_track = gp_track["step"].min(), gp_track["step"].max()
    min_step_label, max_step_label = gp_labels["step"].min(), gp_labels["step"].max()
    min_step = max(min_step_track, min_step_label)
    max_step = min(max_step_track, max_step_label)
    
    gp_track = gp_track[np.logical_and(gp_track["step"]>=min_step, gp_track["step"]<=max_step)]
    gp_labels = gp_labels[np.logical_and(gp_labels["step"]>=min_step, gp_labels["step"]<=max_step)]
    
    unique_players = list(g_track["nfl_palyer_id"].unique())#
    player_id2idx = {pid: num for i,pid in enumerate(unique_players + ["G"])}
    gp_track["nfl_player_idx"] = gp_track["nfl_player_id"].map(player_id2idx).astype(int)
    gp_labels["nfl_player_idx_1"] = gp_labels["nfl_player_id_1"].map(player_id2idx).astype(int)
    gp_labels["nfl_player_idx_2"] = gp_labels["nfl_player_id_2"].map(player_id2idx).astype(int)
    player_team2idx = {"away":1, "home":0}


    #gp_labels_ground = gp_labels[gp_labels["nfl_player_idx_2"]==(len(unique_players)-1)]
    num_step = (max_step - min_step) + 1
    num_players = len(unique_players)
    gp_data_for_gcn =  {#"label_g_contact": - np.ones((num_step, num_players), np.int32), # [num_step, players] player ground contact. (training target)
      #"label_p_contact": - np.ones((num_step, num_players, num_players), np.int32), # [num_step, players, players] player pairs contact. (training target)
      "label_contact": - np.ones((num_step, num_players+1, num_players+1), np.int32),
      "p2p_adj_dist_matrix": - np.ones((num_step, num_players, num_players), np.float32), # [num_frame, players, players] yard distance btw players
      "p2p_adj_team_matrix": - np.ones((num_players, num_players), np.int32), # [players, players] to show same team (-> diff team)
      ##"player_2d_num_matrix": - np.ones((num_players, num_features_numeric), np.float32), # [players, numerical_features] tbd
      ##"player_2d_cat_matrix": - np.ones((num_players, num_features_categorical), np.int32), # [players, categorical_features] position 
      "player_3d_num_matrix": - np.ones((num_step, num_players, len(player_time_features)), np.int32), # [num_frame, players, numerical_features] speed, 
      "step_range": np.arange(min_step, max_step+1, dtype=np.int32),
      }# [num_step] frame number of targets
    
    gp_data_for_gcn["label_contact"][gp_labels["step"].values,
                                     gp_labels["nfl_player_idx_1"].values,
                                     gp_labels["nfl_player_idx_2"].values,
                                     ] = gp_labels["contact"].values
    gp_data_for_gcn["label_p_contact"] = gp_data_for_gcn["label_contact"][:-1,:-1]
    gp_data_for_gcn["label_p_contact"] = gp_data_for_gcn["label_contact"][:-1,-1:]
    
    gp_data_for_gcn["player_3d_num_matrix"][gp_track["step"].values,
                                            gp_track["nfl_player_idx"].values,
                                            ] = gp_track[player_time_features].values
    
    
    xy = gp_data_for_gcn["player_3d_num_matrix"][:,:,:2]
    gp_data_for_gcn["p2p_adj_dist_matrix"] = np.sqrt(np.sum((xy[:,np.newaxis,:,:] - xy[:,:,np.newaxis,:])**2, axis=-1))
    
    
    path = f"gp_data_for_gcn/{game_play}"
    for key, val in gp_data_for_gcn.items():
        save_path = os.path.join(path, f"{key}.npy")
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path, val)
    
"""    
if __name__ == "__main__":

    #model, _, _, _ = build_gcn(num_player=22, num_input_feature=8, num_adj=3)
    model, _, _, _ = build_gcn_1dcnn(num_player=22, num_input_feature=8, num_adj=3)
    
    
    print(model.summary())    