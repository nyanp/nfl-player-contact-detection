import argparse
import gc
import os
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from feature_engineering.kalmen_filter import expand_helmet_smooth
from feature_engineering.merge_for_seq_model import make_features_for_seq_model
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import PredefinedSplit
from feature_engineering.point_set_matching import match_p2p_with_cache
from utils.debugger import set_debugger
from utils.general import (LabelEncoders, LGBMSerializer, make_oof,
                           plot_importance, timer, reduce_mem_usage)
from utils.metrics import binarize_pred, metrics, search_best_threshold_pair_optuna, summarize_per_play_mcc
from utils.nfl import (
    Config,
    ModelSize,
    add_contact_id,
    expand_contact_id,
    expand_helmet,
    read_csv_with_cache,
    TRAIN_COLS,
    TRACK_COLS,
    NON_FEATURE_COLS)

pd.options.mode.chained_assignment = None  # default='warn'


def train(cfg: Config):
    save_dir = f'output/{cfg.EXP_NAME}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cfg.CACHE, exist_ok=True)

    with timer("load file"):
        tr_tracking = read_csv_with_cache("train_player_tracking.csv", cfg.INPUT, cfg.CACHE, usecols=TRACK_COLS)
        train_df = read_csv_with_cache("train_labels.csv", cfg.INPUT, cfg.CACHE, usecols=TRAIN_COLS)
        tr_helmets = read_csv_with_cache("train_baseline_helmets.csv", cfg.HELMET_DIR, cfg.CACHE)
        tr_meta = pd.read_csv(os.path.join(cfg.INPUT, "train_video_metadata.csv"),
                              parse_dates=["start_time", "end_time", "snap_time"])
        tr_regist = pd.read_csv('../input/mfl2cnnkmat0219/output/p2p_registration_residuals.csv')

    with timer("assign helmet metadata"):
        train_df = expand_helmet(cfg, train_df, "train")
        train_df = expand_helmet_smooth(cfg, train_df, "train")
        del tr_helmets, tr_meta
        gc.collect()

    train_feature_df = make_features_for_seq_model(train_df, tr_tracking, tr_regist)
    del tr_tracking, tr_regist
    gc.collect()

    train_feature_df.to_feather(f'{save_dir}/train_feature_df_for_seq_model.f')


def inference(cfg: Config):
    with timer("load file"):
        test_tracking = read_csv_with_cache(
            "test_player_tracking.csv", cfg.INPUT, cfg.CACHE, usecols=TRACK_COLS)

        sub = read_csv_with_cache("sample_submission.csv", cfg.INPUT, cfg.CACHE)
        test_df = expand_contact_id(sub)
        test_df = pd.merge(test_df,
                           test_tracking[["step", "game_play", "datetime"]].drop_duplicates(),
                           on=["game_play", "step"], how="left")
        test_df = expand_helmet(cfg, test_df, "test")
        test_df = expand_helmet_smooth(cfg, test_df, "test")

        te_helmets = read_csv_with_cache("test_baseline_helmets.csv", cfg.HELMET_DIR, cfg.CACHE)
        te_meta = pd.read_csv(os.path.join(cfg.INPUT, "test_video_metadata.csv"),
                              parse_dates=["start_time", "end_time", "snap_time"])
        test_regist = match_p2p_with_cache(os.path.join(cfg.CACHE, "test_registration.f"), tracking=test_tracking, helmets=te_helmets, meta=te_meta)

    def _make_features_per_game(game_play, game_test_df, game_test_tracking, game_test_regist, cnn_df_dict):
        with timer("make features(test)"):
            game_test_feature_df = make_features_for_seq_model(game_test_df, game_test_tracking, game_test_regist)
            game_test_feature_df.to_feather(f'{game_play}_test_feature_df_for_seq_model.f')

    game_plays = test_df['game_play'].unique()
    game_test_gb = test_df.groupby(['game_play'])
    game_test_tracking_gb = test_tracking.groupby(['game_play'])
    game_test_regist_gb = test_regist.groupby(['game_play'])
    for game_play in game_plays:
        game_test_df = game_test_gb.get_group(game_play)
        game_test_tracking = game_test_tracking_gb.get_group(game_play)
        game_test_regist = game_test_regist_gb.get_group(game_play)
        _make_features_per_game(game_play, game_test_df, game_test_tracking, game_test_regist, cnn_df_dict={})


def main(args):
    cfg = Config(
        EXP_NAME='features_for_seq_model',
        PRETRAINED_MODEL_PATH=args.lgbm_path,
        CAMARO_DF1_PATH=args.camaro1_path,
        CAMARO_DF1_ANY_PATH=args.camaro1_any_path,
        CAMARO_DF2_PATH=args.camaro2_path,
        CAMARO_DF2_ANY_PATH=args.camaro2_any_path,
        KMAT_END_DF_PATH=args.kmat_end_path,
        KMAT_SIDE_DF_PATH=args.kmat_side_path,
        KMAT_END_MAP_DF_PATH=args.kmat_end_map_path,
        KMAT_SIDE_MAP_DF_PATH=args.kmat_side_map_path,
        MODEL_SIZE=ModelSize.HUGE,
        ENABLE_MULTIPROCESS=args.enable_multiprocess,
        DEBUG=args.debug,
        RAW_OUTPUT=args.raw_output,
    )
    if args.debug:
        set_debugger()
        cfg.MODEL_SIZE = ModelSize.SMALL
    if not args.inference_only:
        train(cfg)
    # inference(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--inference_only", "-i", action="store_true")
    parser.add_argument("--validate_only", "-v", action="store_true")
    parser.add_argument("--lgbm_path", "-l", default="", type=str)
    parser.add_argument("--camaro1_path", "-c1", default="", type=str)
    parser.add_argument("--camaro1_any_path", "-c1a", default="", type=str)
    parser.add_argument("--camaro2_path", "-c2", default="", type=str)
    parser.add_argument("--camaro2_any_path", "-c2a", default="", type=str)
    parser.add_argument("--kmat_end_path", "-e", default="", type=str)
    parser.add_argument("--kmat_side_path", "-s", default="", type=str)
    parser.add_argument("--kmat_end_map_path", "-em", default="", type=str)
    parser.add_argument("--kmat_side_map_path", "-sm", default="", type=str)
    parser.add_argument("--enable_multiprocess", "-m", action='store_true')
    parser.add_argument("--raw_output", "-r", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
