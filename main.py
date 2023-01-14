import argparse
import gc
import os
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from feature_engineering.merge import make_features
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import PredefinedSplit
from feature_engineering.point_set_matching import match_p2p_with_cache
from utils.debugger import set_debugger
from utils.general import (LabelEncoders, LGBMSerializer, make_oof,
                           plot_importance, timer)
from utils.metrics import binarize_pred, metrics, search_best_threshold_pair, summarize_per_play_mcc
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


def get_lgb_params(cfg):
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "max_depth": -1,
        "num_leaves": 48,
        "verbose": -1,
    }
    if cfg.MODEL_SIZE == ModelSize.SMALL:
        lgb_params["learning_rate"] = 0.1
        lgb_params["boosting"] = "goss"
    elif cfg.MODEL_SIZE == ModelSize.MEDIUM:
        lgb_params["learning_rate"] = 0.05
        lgb_params["boosting"] = "goss"
    else:
        lgb_params["learning_rate"] = 0.03
        lgb_params["reg_lambda"] = 1
    return lgb_params


def train_cv(
        cfg,
        train_df,
        split_defs,
        original_df,
        selected_index,
        calc_oof=True,
        search_threshold=True):

    lgb_params = get_lgb_params(cfg)

    is_ground = train_df["nfl_player_id_2"] == -1
    y_train = train_df["contact"]

    split_df = train_df[["game_play"]].copy()
    split_df["game"] = split_df["game_play"].str[:5].astype(int)
    split_df = pd.merge(split_df, split_defs, how="left")
    split = list(PredefinedSplit(split_df["fold"]).split())

    encoder = LabelEncoders()

    with timer("make dataset"):
        # train_df.values.astype(np.float32)とかやるとOOMで死ぬので、箱を先に用意して値を入れていく
        feature_names = [
            c for c in train_df.columns if c not in NON_FEATURE_COLS]

        X_train = np.empty(
            (len(train_df), len(feature_names)), dtype=np.float32)
        print(X_train.shape)

        for i, c in enumerate(feature_names):
            if train_df[c].dtype.name == "object":
                X_train[:, i] = encoder.fit_transform_one(train_df[c])
            else:
                X_train[:, i] = train_df[c]

        gc.collect()
        # print(f"features: {feature_names}")
        print(f"category: {list(encoder.encoders.keys())}")

        ds_train = lgb.Dataset(X_train, y_train, feature_name=feature_names)
        gc.collect()

    with timer("lgb.cv"):
        ret = lgb.cv(lgb_params, ds_train,
                     num_boost_round=1000,
                     folds=split,
                     return_cvbooster=True,
                     callbacks=[
                         lgb.early_stopping(stopping_rounds=50, verbose=True),
                         lgb.log_evaluation(25),
                         #  wandb_callback()
                     ])

        for booster in ret["cvbooster"].boosters:
            booster.best_iteration = ret["cvbooster"].best_iteration

        del ds_train
        gc.collect()

    plot_importance(ret["cvbooster"])

    if calc_oof:
        oof = make_oof(ret["cvbooster"], X_train, y_train, split)

        np.save("X_train.npy", X_train)
        np.save("oof.npy", oof)

        if search_threshold:
            with timer("find best threshold"):
                threshold_1, threshold_2 = search_best_threshold_pair(
                    y_train, oof, is_ground)

            y_train_all = original_df["contact"]
            oof_pred_all = np.zeros(len(y_train_all))
            oof_pred_all[selected_index] = oof
            is_ground_all = original_df["nfl_player_id_2"] == -1
            mcc = metrics(
                y_train_all,
                oof_pred_all,
                threshold_1,
                threshold_2,
                is_ground_all)
            auc = roc_auc_score(y_train_all, oof_pred_all)
            print(
                f"threshold: {threshold_1:.5f}, {threshold_2:.5f}, mcc: {mcc:.5f}, auc: {auc:.5f}")

            mcc_ground = metrics(
                y_train_all[is_ground_all],
                oof_pred_all[is_ground_all],
                threshold_2)
            mcc_non_ground = metrics(
                y_train_all[~is_ground_all], oof_pred_all[~is_ground_all], threshold_1)

            print(
                f"mcc(ground): {mcc_ground:.5f}, mcc(non-ground): {mcc_non_ground:.5f}")

            y_pred_all = binarize_pred(oof_pred_all, threshold_1, threshold_2, is_ground_all)
            original_df["oof"] = oof_pred_all
            original_df["y_pred"] = y_pred_all
            per_play_mcc_df = summarize_per_play_mcc(original_df)
            per_play_mcc_df.to_csv('per_play_mcc_df.csv', index=False)

            wandb.log(dict(
                threshold_1=threshold_1,
                threshold_2=threshold_2,
                mcc=mcc,
                mcc_ground=mcc_ground,
                mcc_non_ground=mcc_non_ground,
                auc=auc,
                per_play_mcc=wandb.Table(dataframe=per_play_mcc_df)
            ))

            return ret["cvbooster"], encoder, threshold_1, threshold_2

    return ret["cvbooster"], encoder, None, None


def train(cfg: Config):
    mode = 'disabled' if cfg.DEBUG else None
    wandb.init(
        project=cfg.PROJECT,
        name=f'{cfg.EXP_NAME}',
        config=cfg,
        reinit=True,
        mode=mode)

    with timer("load file"):
        tr_tracking = read_csv_with_cache("train_player_tracking.csv", cfg.INPUT, cfg.CACHE, usecols=TRACK_COLS)
        train_df = read_csv_with_cache("train_labels.csv", cfg.INPUT, cfg.CACHE, usecols=TRAIN_COLS)
        split_defs = pd.read_csv(cfg.SPLIT_FILE_PATH)
        tr_helmets = read_csv_with_cache("train_baseline_helmets.csv", cfg.HELMET_DIR, cfg.CACHE)
        tr_meta = pd.read_csv(os.path.join(cfg.INPUT, "train_video_metadata.csv"),
                              parse_dates=["start_time", "end_time", "snap_time"])
        tr_regist = match_p2p_with_cache("train_registration.f", tracking=tr_tracking, helmets=tr_helmets, meta=tr_meta)

    with timer("assign helmet metadata"):
        train_df = expand_helmet(cfg, train_df, "train")
        gc.collect()

    if cfg.DEBUG:
        print('sample small subset for debug.')
        train_df = train_df.sample(10000)
    train_feature_df, train_selected_index = make_features(train_df, tr_tracking, tr_regist)
    gc.collect()

    asdict(cfg)

    cvbooster, encoder, threshold_1, threshold_2 = train_cv(
        cfg, train_feature_df, split_defs, train_df, train_selected_index)
    gc.collect()

    del train_feature_df
    gc.collect()

    serializer = LGBMSerializer(cvbooster, encoder, threshold_1, threshold_2)
    serializer.to_file("lgb")


def inference(cfg: Config):
    serializer = LGBMSerializer.from_file(
        os.path.join(cfg.PRETRAINED_MODEL_PATH, "lgb"))
    cvbooster = serializer.booster
    encoder = serializer.encoders
    threshold_1 = serializer.threshold_1
    threshold_2 = serializer.threshold_2

    with timer("load file"):
        te_tracking = read_csv_with_cache(
            "test_player_tracking.csv", cfg.INPUT, cfg.CACHE, usecols=TRACK_COLS)

        sub = read_csv_with_cache("sample_submission.csv", cfg.INPUT, cfg.CACHE)
        test_df = expand_contact_id(sub)
        test_df = pd.merge(test_df,
                           te_tracking[["step", "game_play", "datetime"]].drop_duplicates(),
                           on=["game_play", "step"], how="left")
        test_df = expand_helmet(cfg, test_df, "test")

        te_helmets = read_csv_with_cache("test_baseline_helmets.csv", cfg.HELMET_DIR, cfg.CACHE)
        te_meta = pd.read_csv(os.path.join(cfg.INPUT, "test_video_metadata.csv"),
                              parse_dates=["start_time", "end_time", "snap_time"])
        te_regist = match_p2p_with_cache("test_registration.f", tracking=te_tracking, helmets=te_helmets, meta=te_meta)

        df_args = []
        if cfg.CAMARO_DF_PATH:
            df_args.append(pd.read_csv(cfg.CAMARO_DF_PATH))
        if cfg.KMAT_END_DF_PATH:
            df_args.append(pd.read_csv(cfg.KMAT_END_DF_PATH))
        if cfg.KMAT_SIDE_DF_PATH:
            df_args.append(pd.read_csv(cfg.KMAT_SIDE_DF_PATH))

    feature_cols = cvbooster.feature_name()[0]

    with timer("make features(test)"):
        test_feature_df, test_selected_index = make_features(test_df, te_tracking, te_regist, df_args)

    X_test = encoder.transform(test_feature_df[feature_cols])
    predicted = cvbooster.predict(X_test)

    avg_predicted = np.array(predicted).mean(axis=0)

    is_ground = test_feature_df["nfl_player_id_2"] == -1
    pred_binalized = binarize_pred(
        avg_predicted, threshold_1, threshold_2, is_ground)

    test_df = add_contact_id(test_df)
    test_df['contact'] = 0
    test_df.loc[test_selected_index, 'contact'] = pred_binalized.astype(int).values
    test_df[['contact_id', 'contact']].to_csv('submission.csv', index=False)


def main(args):
    cfg = Config(
        EXP_NAME='exp019_exp048_both_org_and_agg_re',
        PRETRAINED_MODEL_PATH=args.lgbm_path,
        CAMARO_DF_PATH=args.camaro_path,
        KMAT_END_DF_PATH=args.kmat_end_path,
        KMAT_SIDE_DF_PATH=args.kmat_side_path,
        MODEL_SIZE=ModelSize.LARGE,
        DEBUG=args.debug)

    if args.debug:
        set_debugger()
        cfg.MODEL_SIZE = ModelSize.SMALL

    if not args.inference_only:
        train(cfg)
    inference(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--inference_only", "-i", action="store_true")
    parser.add_argument("--lgbm_path", "-l", default="", type=str)
    parser.add_argument("--camaro_path", "-c", default="", type=str)
    parser.add_argument("--kmat_end_path", "-e", default="", type=str)
    parser.add_argument("--kmat_side_path", "-s", default="", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
