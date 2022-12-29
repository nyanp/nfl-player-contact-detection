import argparse
import gc
import os
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from feature_engineering import make_features
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import PredefinedSplit
from utils.debugger import set_debugger
from utils.general import (LabelEncoders, LGBMSerializer, make_oof,
                           plot_importance, timer)
from utils.metrics import binarize_pred, metrics, search_best_threshold_pair
from utils.nfl import (
    Config,
    ModelSize,
    add_contact_id,
    expand_contact_id,
    expand_helmet,
    read_csv_with_cache,
    TRAIN_COLS,
    TRACK_COLS)


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
    non_feature_cols = [
        "contacgt_id",
        "game_play",
        "datetime",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "team_1",
        "team_2",
        "contact_id",
        # "position_1",
        # "position_2"
        # "direction_1",
        # "direction_2",
        "x_position_1",
        "x_position_2",
        "y_position_1",
        "y_position_2",
        "x_position_start_1",
        "x_position_start_2",
        "y_position_start_1",
        "y_position_start_2",

        "x_position_future5_1",
        "x_position_future5_2",
        "y_position_future5_1",
        "y_position_future5_2",
        "x_position_past5_1",
        "x_position_past5_2",
        "y_position_past5_1",
        "y_position_past5_2",

        # "orientation_past5_1",
        # "direction_past5_1",
        # "orientation_past5_2",
        # "direction_past5_2",
    ]

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
            c for c in train_df.columns if c not in non_feature_cols]

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

            wandb.log(dict(
                threshold_1=threshold_1,
                threshold_2=threshold_2,
                mcc=mcc,
                mcc_ground=mcc_ground,
                mcc_non_ground=mcc_non_ground,
                auc=auc,
            ))

            return ret["cvbooster"], encoder, threshold_1, threshold_2

    return ret["cvbooster"], encoder, None, None


def train(cfg):
    with timer("load file"):
        tr_tracking = read_csv_with_cache(
            "train_player_tracking.csv", cfg, usecols=TRACK_COLS)
        train = read_csv_with_cache(
            "train_labels.csv", cfg, usecols=TRAIN_COLS)
        split_defs = pd.read_csv(cfg.SPLIT_FILE_PATH)

    with timer("assign helmet metadata"):
        train = expand_helmet(cfg, train, "train")
        gc.collect()

    if cfg.DEBUG:
        print('sample small subset for debug.')
        train = train.sample(10000)
    train_df, train_selected_index = make_features(train, tr_tracking)
    gc.collect()

    asdict(cfg)

    cvbooster, encoder, threshold_1, threshold_2 = train_cv(
        cfg, train_df, split_defs, train, train_selected_index)
    gc.collect()

    del train_df
    gc.collect()

    serializer = LGBMSerializer(cvbooster, encoder, threshold_1, threshold_2)
    serializer.to_file("lgb")


def predict(cfg: Config):
    serializer = LGBMSerializer.from_file(
        os.path.join(cfg.PRETRAINED_MODEL_PATH, "lgb"))
    cvbooster = serializer.booster
    encoder = serializer.encoders
    threshold_1 = serializer.threshold_1
    threshold_2 = serializer.threshold_2

    te_tracking = read_csv_with_cache(
        "test_player_tracking.csv", cfg, usecols=TRACK_COLS)

    sub = read_csv_with_cache("sample_submission.csv", cfg)
    test = expand_contact_id(sub)
    test = pd.merge(test,
                    te_tracking[["step",
                                 "game_play",
                                 "datetime"]].drop_duplicates(),
                    on=["game_play",
                        "step"],
                    how="left")
    test = expand_helmet(cfg, test, "test")

    feature_cols = cvbooster.feature_name()[0]

    with timer("make features(test)"):
        test_df, test_selected_index = make_features(test, te_tracking)

    X_test = encoder.transform(test_df[feature_cols])
    predicted = cvbooster.predict(X_test)

    avg_predicted = np.array(predicted).mean(axis=0)

    is_ground = test_df["nfl_player_id_2"] == -1
    pred_binalized = binarize_pred(
        avg_predicted, threshold_1, threshold_2, is_ground)

    test = add_contact_id(test)
    test['contact'] = 0
    test.loc[test_selected_index, 'contact'] = pred_binalized.astype(int)
    test[['contact_id', 'contact']].to_csv('submission.csv', index=False)


def main(args):
    if args.debug:
        set_debugger()

    cfg = Config(
        EXP_NAME='exp002_remove_hard_example_large',
        PRETRAINED_MODEL_PATH='./',
        MODEL_SIZE=ModelSize.LARGE,
        DEBUG=args.debug)

    mode = 'disabled' if cfg.DEBUG else None
    wandb.init(
        project=cfg.PROJECT,
        name=f'{cfg.EXP_NAME}',
        config=cfg,
        reinit=True,
        mode=mode)

    if not cfg.USE_PRETRAINED_MODEL:
        train(cfg)
    predict(cfg)

# LARGE
# threshold: 0.31710, 0.22987, mcc: 0.72922, auc: 0.99593
# mcc(ground): 0.62898, mcc(non-ground): 0.76083

# SMALL
# threshold: 0.19394, 0.09807, mcc: 0.68636, auc: 0.99447
# mcc(ground): 0.56401, mcc(non-ground): 0.72784

# SMALL with removing hard example
# threshold: 0.29897, 0.24560, mcc: 0.72684, auc: 0.99544
# mcc(ground): 0.62286, mcc(non-ground): 0.75924


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-db", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
