import argparse
import gc
import os
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from feature_engineering.merge import make_features
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

IGNORE_PAIRS = [
    ('58577_002486', 45573, 47863),  # strange
    ('58311_002159', 43475, 44892),  # strange
    ('58577_002486', 45033, 45573),  # strange
    # ('58403_001076', 45005, 52410), # missing
    # ('58279_002309', 41947, 46238), # missing
    # ('58384_000142', 45203, 48099), # missing
    ('58270_002527', 43362, 52435),  # strange
    # ('58247_003522', 45305, 52462), # miss
    # ('58399_000570', 37077, 46778), # miss
    # ('58180_000986', 46113, 46657), # miss
    # ('58406_000188', 46243, 47802), # lost
    # ('58551_003569', 53462, 53475), # lost
    ('58301_002369', 42883, 46204),  # strange
    # ('58415_003155', 43700, 48988), miss
    # ('58220_002149', 41292, 52459), miss
    # ('58537_000757', 46184, 53640), # lost
    # ('58245_002594', 43361, 47871), # lost
    ('58180_000986', 43697, 46164),  # ambiguous,
    # ('58551_003569', 42375, 53475), # lost
    # ('58240_000086', 42392, 47857), # lost
    # ('58247_003522', 46085, 52462), # lost
    ('58525_003852', 43534, 53079),  # strange
    # ('58204_002864', 42388, 43334), # lost
    ('58187_002329', 41947, 43426),  # strange
    ('58202_002335', 43399, 46165),  # strange
    ('58401_001720', 43341, 43454),  # strange
    ('58555_002563', 45532, 46527),  # strange
    # ('58552_002943', 39998, 46082), # lost
    ('58537_000757', 43474, 53640),  # strange?
    # ('58308_004092', 41275, 44927), # lost
    # ('58516_003538', 47800, 53464), # miss
    # ('58407_001855', 43319, 52499), # lost
    ('58224_002486', 42404, 46188),  # strange
    ('58188_001757', 42830, 52449),  # strange
    ('58216_001891', 42358, 44892),  # strange
    ('58266_000095', 42390, 45003),  # strange
    ('58422_001959', 43648, 46203),  # strange
    ('58291_001043', 39957, 48220),  # strange
    ('58546_003306', 42924, 43436),  # strange
    ('58527_000757', 38542, 42830),  # strange
    # ('58308_004092', 37104, 44927), # lost
    # ('58302_004013', 46618, 52619), # miss
    # ('58187_002329', 40116, 41947), # lost
    # ('58281_000692', 40011, 42771), # miss
    # ('58403_001076', 44876, 46082), # miss
    # ('58550_001554', 43293, 44826), # miss
    # ('58227_000943', 46191, 47844), # lost
    ('58330_000759', 43484, 45012),  # ambiguous
    ('58174_001792', 44827, 52450),  # strange
]


def get_lgb_params(cfg):
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "max_depth": -1,
        "num_leaves": 40,
        "verbose": -1,
    }
    if cfg.MODEL_SIZE == ModelSize.SMALL:
        lgb_params["learning_rate"] = 0.1
        lgb_params["boosting"] = "goss"
    elif cfg.MODEL_SIZE == ModelSize.MEDIUM:
        lgb_params["learning_rate"] = 0.05
        lgb_params["boosting"] = "goss"
    elif cfg.MODEL_SIZE == ModelSize.LARGE:
        lgb_params["learning_rate"] = 0.03
        lgb_params["reg_lambda"] = 0.5
        lgb_params["reg_alpha"] = 1
    elif cfg.MODEL_SIZE == ModelSize.HUGE:
        lgb_params["feature_fraction"] = 0.3
        lgb_params["learning_rate"] = 0.02
        lgb_params["min_child_samples"] = 10
        lgb_params["min_child_weight"] = 5
        lgb_params["num_leaves"] = 128
        lgb_params["subsample_for_bin"] = 50000
    else:
        raise NotImplementedError()
    return lgb_params


def get_pseudo_ds_train(X_train, y_train, is_ground, feature_names):
    # pseudo labeling from exp046
    pseudo_y_train = np.load('output/exp046_holdout_fold3_only_camaro_cnn/oof.npy')
    serializer = LGBMSerializer.from_file("lgb", 'output/exp046_holdout_fold3_only_camaro_cnn')
    threshold_1 = serializer.threshold_1
    threshold_2 = serializer.threshold_2
    pseudo_y_pred = binarize_pred(pseudo_y_train, threshold_1, threshold_2, is_ground)
    ds_train = lgb.Dataset(X_train, pseudo_y_pred, feature_name=feature_names)
    return ds_train


def get_ignore_strange_ds_train(X_train, y_train, is_ground, feature_names):
    # ignore strange examples by exp046 oof
    THRESHOLD = 0.99
    oof = np.load('output/exp046_holdout_fold3_only_camaro_cnn/oof.npy')
    weights = (np.abs(y_train - oof) < THRESHOLD)
    num_samples = len(weights)
    ignore_samples = num_samples - weights.sum()
    print(f'ignore {ignore_samples} samples out of {num_samples} samples')
    ds_train = lgb.Dataset(X_train, y_train, feature_name=feature_names, weight=weights.astype(np.float32))
    return ds_train


def get_manuall_ignore_strange_ds_train(X_train, y_train, is_ground, feature_names, train_df):
    # ignore strange labels manually
    weights = []
    for key, df in train_df.groupby(['game_play', 'nfl_player_id_1', 'nfl_player_id_2']):
        if key in IGNORE_PAIRS:
            weights.append(np.zeros(len(df)))
        else:
            weights.append(np.ones(len(df)))
    weights = np.concatenate(weights)
    print(f'ignore {(weights==0).sum()} samples out of {len(weights)} samples')
    ds_train = lgb.Dataset(X_train, y_train, feature_name=feature_names, weight=weights.astype(np.float32))
    return ds_train


def get_normal_ds_train(X_train, y_train, is_ground, feature_names):
    return lgb.Dataset(X_train, y_train, feature_name=feature_names)


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

        # ds_train = get_pseudo_ds_train(X_train, y_train, is_ground, feature_names)
        # ds_train = get_ignore_strange_ds_train(X_train, y_train, is_ground, feature_names)
        # ds_train = get_manuall_ignore_strange_ds_train(X_train, y_train, is_ground, feature_names, train_df)
        ds_train = get_normal_ds_train(X_train, y_train, is_ground, feature_names)

        gc.collect()

    with timer("lgb.cv"):
        split.append((np.arange(len(train_df)), np.arange(10)))
        ret = lgb.cv(lgb_params, ds_train,
                     num_boost_round=4000,
                     folds=split,
                     return_cvbooster=True,
                     callbacks=[
                         #  lgb.early_stopping(stopping_rounds=100, verbose=True),
                         lgb.log_evaluation(25),
                         #  wandb_callback()
                     ])

        for booster in ret["cvbooster"].boosters:
            booster.best_iteration = ret["cvbooster"].best_iteration

        del ds_train
        gc.collect()

    plot_importance(ret["cvbooster"])

    if calc_oof:
        oof = make_oof(ret["cvbooster"].boosters[:-1], X_train, y_train, split)

        save_dir = f'output/{cfg.EXP_NAME}'
        np.save(f"{save_dir}/X_train.npy", X_train)
        np.save(f"{save_dir}/oof.npy", oof)

        if search_threshold:
            with timer("find best threshold"):
                threshold_1, threshold_2 = search_best_threshold_pair_optuna(
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

            # holdout_mcc = holdout_validate(cfg, ret["cvbooster"], encoder, threshold_1, threshold_2)

            wandb.log(dict(
                threshold_1=threshold_1,
                threshold_2=threshold_2,
                mcc=mcc,
                mcc_ground=mcc_ground,
                mcc_non_ground=mcc_non_ground,
                # holdout_mcc=holdout_mcc,
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
        entity='nfl-osaka-tigers',
        config=cfg,
        reinit=True,
        mode=mode)

    save_dir = f'output/{cfg.EXP_NAME}'
    os.makedirs(save_dir, exist_ok=True)

    with timer("load file"):
        tr_tracking = read_csv_with_cache("train_player_tracking.csv", cfg.INPUT, cfg.CACHE, usecols=TRACK_COLS)
        train_df = read_csv_with_cache("train_labels.csv", cfg.INPUT, cfg.CACHE, usecols=TRAIN_COLS)
        split_defs = pd.read_csv(cfg.SPLIT_FILE_PATH)
        tr_helmets = read_csv_with_cache("train_baseline_helmets.csv", cfg.HELMET_DIR, cfg.CACHE)
        tr_meta = pd.read_csv(os.path.join(cfg.INPUT, "train_video_metadata.csv"),
                              parse_dates=["start_time", "end_time", "snap_time"])
        # tr_regist = match_p2p_with_cache(os.path.join(cfg.CACHE, "train_registration.f"), tracking=tr_tracking, helmets=tr_helmets, meta=tr_meta)
        tr_regist = pd.read_csv('../input/mfl2cnnkmat0121/output/p2p_registration_residuals.csv')

    with timer("assign helmet metadata"):
        train_df = expand_helmet(cfg, train_df, "train")
        del tr_helmets, tr_meta
        gc.collect()

    if cfg.DEBUG:
        print('sample small subset for debug.')
        train_df = train_df.sample(10000).reset_index(drop=True)

    # # hold out fold 3
    # split_df = train_df[["game_play"]].copy()
    # split_df["game"] = split_df["game_play"].str[:5].astype(int)
    # split_df = pd.merge(split_df, split_defs, how="left")
    # train_df = train_df.loc[split_df['fold'] != 3].reset_index(drop=True)
    # print('hold out fold 3.', train_df.shape)

    train_feature_df, train_selected_index = make_features(train_df, tr_tracking, tr_regist)
    del tr_tracking, tr_regist
    gc.collect()

    asdict(cfg)

    train_feature_df.to_feather(f'{save_dir}/train_feature_df.f')

    cvbooster, encoder, threshold_1, threshold_2 = train_cv(
        cfg, train_feature_df, split_defs, train_df, train_selected_index)
    gc.collect()

    del train_feature_df
    gc.collect()

    serializer = LGBMSerializer(cvbooster, encoder, threshold_1, threshold_2)
    serializer.to_file("lgb", save_dir)


def holdout_validate(cfg: Config, cvbooster, encoder, threshold_1, threshold_2):
    with timer("load file"):
        tr_tracking = read_csv_with_cache("train_player_tracking.csv", cfg.INPUT, cfg.CACHE, usecols=TRACK_COLS)
        train_df = read_csv_with_cache("train_labels.csv", cfg.INPUT, cfg.CACHE, usecols=TRAIN_COLS)
        split_defs = pd.read_csv(cfg.SPLIT_FILE_PATH)
        tr_helmets = read_csv_with_cache("train_baseline_helmets.csv", cfg.HELMET_DIR, cfg.CACHE)
        tr_meta = pd.read_csv(os.path.join(cfg.INPUT, "train_video_metadata.csv"),
                              parse_dates=["start_time", "end_time", "snap_time"])
        tr_regist = match_p2p_with_cache(os.path.join(cfg.CACHE, "train_registration.f"), tracking=tr_tracking, helmets=tr_helmets, meta=tr_meta)

    with timer("assign helmet metadata"):
        train_df = expand_helmet(cfg, train_df, "train")
        gc.collect()

    if cfg.DEBUG:
        print('sample small subset for debug.')
        train_df = train_df.sample(10000)

    # hold out fold 3
    split_df = train_df[["game_play"]].copy()
    split_df["game"] = split_df["game_play"].str[:5].astype(int)
    split_df = pd.merge(split_df, split_defs, how="left")
    test_df = train_df.loc[split_df['fold'] == 3].reset_index(drop=True)
    print('hold out fold 3.', test_df.shape)

    test_tracking = tr_tracking
    test_regist = tr_regist

    df_args = [None, None, None, pd.read_csv('../pipeline/exp096_holdout3_preds.csv')]

    feature_cols = cvbooster.feature_name()[0]

    def _predict_per_game(game_test_df, game_test_tracking, game_test_regist, df_args):
        with timer("make features(test)"):
            game_test_feature_df, test_selected_index = make_features(
                game_test_df, game_test_tracking, game_test_regist, df_args, enable_multiprocess=False)

        X_test = encoder.transform(game_test_feature_df[feature_cols])
        predicted = cvbooster.predict(X_test)

        del X_test
        gc.collect()

        avg_predicted = np.array(predicted).mean(axis=0)

        is_ground = game_test_feature_df["nfl_player_id_2"] == -1
        pred_binalized = binarize_pred(
            avg_predicted, threshold_1, threshold_2, is_ground)

        game_test_df = add_contact_id(game_test_df)
        game_test_df['contact'] = 0
        game_test_df.loc[test_selected_index, 'contact'] = pred_binalized.astype(int).values
        return game_test_df

    game_test_dfs = []
    game_plays = test_df['game_play'].unique()
    game_test_gb = test_df.groupby(['game_play'])
    game_test_tracking_gb = test_tracking.groupby(['game_play'])
    game_test_regist_gb = test_regist.groupby(['game_play'])
    for game_play in game_plays:
        game_test_df = game_test_gb.get_group(game_play)
        game_test_tracking = game_test_tracking_gb.get_group(game_play)
        game_test_regist = game_test_regist_gb.get_group(game_play)
        game_test_df = _predict_per_game(game_test_df, game_test_tracking, game_test_regist, df_args)
        game_test_dfs.append(game_test_df)
        gc.collect()
    sub = pd.concat(game_test_dfs).reset_index(drop=True)

    save_dir = cfg.PRETRAINED_MODEL_PATH or f'output/{cfg.EXP_NAME}'
    sub[['contact_id', 'contact']].to_csv(f'{save_dir}/holdout_preds.csv', index=False)
    mcc = matthews_corrcoef(test_df.contact, sub.contact)
    print(f'hold out set score = {mcc}')
    return mcc


def inference(cfg: Config):
    save_dir = cfg.PRETRAINED_MODEL_PATH or f'output/{cfg.EXP_NAME}'
    serializer = LGBMSerializer.from_file("lgb", save_dir)
    cvbooster = serializer.booster
    encoder = serializer.encoders
    threshold_1 = serializer.threshold_1
    threshold_2 = serializer.threshold_2

    with timer("load file"):
        test_tracking = read_csv_with_cache(
            "test_player_tracking.csv", cfg.INPUT, cfg.CACHE, usecols=TRACK_COLS)

        sub = read_csv_with_cache("sample_submission.csv", cfg.INPUT, cfg.CACHE)
        test_df = expand_contact_id(sub)
        test_df = pd.merge(test_df,
                           test_tracking[["step", "game_play", "datetime"]].drop_duplicates(),
                           on=["game_play", "step"], how="left")
        test_df = expand_helmet(cfg, test_df, "test")

        te_helmets = read_csv_with_cache("test_baseline_helmets.csv", cfg.HELMET_DIR, cfg.CACHE)
        te_meta = pd.read_csv(os.path.join(cfg.INPUT, "test_video_metadata.csv"),
                              parse_dates=["start_time", "end_time", "snap_time"])
        test_regist = match_p2p_with_cache(os.path.join(cfg.CACHE, "test_registration.f"), tracking=test_tracking, helmets=te_helmets, meta=te_meta)

        df_args = []
        if cfg.CAMARO_DF_PATH:
            df_args.append(pd.read_csv(cfg.CAMARO_DF_PATH))
        else:
            df_args.append(None)
        if cfg.KMAT_END_DF_PATH:
            df_args.append(pd.read_csv(cfg.KMAT_END_DF_PATH))
        else:
            df_args.append(None)
        if cfg.KMAT_SIDE_DF_PATH:
            df_args.append(pd.read_csv(cfg.KMAT_SIDE_DF_PATH))
        else:
            df_args.append(None)
        if cfg.CAMARO_DF2_PATH:
            df_args.append(pd.read_csv(cfg.CAMARO_DF2_PATH))
        else:
            df_args.append(None)
        if cfg.CAMARO_DF3_PATH:
            df_args.append(pd.read_csv(cfg.CAMARO_DF3_PATH))
        else:
            df_args.append(None)
        if cfg.CAMARO_DF4_PATH:
            df_args.append(pd.read_csv(cfg.CAMARO_DF4_PATH))
        else:
            df_args.append(None)

    feature_cols = cvbooster.feature_name()[0]

    def _predict_per_game(game_test_df, game_test_tracking, game_test_regist, df_args):
        with timer("make features(test)"):
            game_test_feature_df, test_selected_index = make_features(
                game_test_df, game_test_tracking, game_test_regist, df_args, False)

        X_test = encoder.transform(game_test_feature_df[feature_cols])
        predicted = cvbooster.predict(X_test)

        del X_test
        gc.collect()

        avg_predicted = np.array(predicted).mean(axis=0)

        is_ground = game_test_feature_df["nfl_player_id_2"] == -1
        pred_binalized = binarize_pred(
            avg_predicted, threshold_1, threshold_2, is_ground)

        game_test_df = add_contact_id(game_test_df)
        game_test_df['contact'] = 0
        game_test_df.loc[test_selected_index, 'contact'] = pred_binalized.astype(int).values
        # game_test_df[['contact_id', 'contact']].to_csv('submission.csv', index=False)
        return game_test_df

    game_test_dfs = []
    game_plays = test_df['game_play'].unique()
    game_test_gb = test_df.groupby(['game_play'])
    game_test_tracking_gb = test_tracking.groupby(['game_play'])
    game_test_regist_gb = test_regist.groupby(['game_play'])
    for game_play in game_plays:
        game_test_df = game_test_gb.get_group(game_play)
        game_test_tracking = game_test_tracking_gb.get_group(game_play)
        game_test_regist = game_test_regist_gb.get_group(game_play)
        game_test_df = _predict_per_game(game_test_df, game_test_tracking, game_test_regist, df_args)
        game_test_dfs.append(game_test_df)
        gc.collect()
    test_df = pd.concat(game_test_dfs).reset_index(drop=True)
    test_df[['contact_id', 'contact']].to_csv('submission.csv', index=False)


def main(args):
    cfg = Config(
        EXP_NAME='exp075_camaro128_agg',
        PRETRAINED_MODEL_PATH=args.lgbm_path,
        CAMARO_DF_PATH=args.camaro_path,
        CAMARO_DF2_PATH=args.camaro2_path,
        CAMARO_DF3_PATH=args.camaro3_path,
        CAMARO_DF4_PATH=args.camaro4_path,
        KMAT_END_DF_PATH=args.kmat_end_path,
        KMAT_SIDE_DF_PATH=args.kmat_side_path,
        MODEL_SIZE=ModelSize.HUGE,
        ENABLE_MULTIPROCESS=args.enable_multiprocess,
        DEBUG=args.debug)
    if args.debug:
        set_debugger()
        cfg.MODEL_SIZE = ModelSize.SMALL

    if args.validate_only:
        save_dir = cfg.PRETRAINED_MODEL_PATH or f'output/{cfg.EXP_NAME}'
        serializer = LGBMSerializer.from_file("lgb", save_dir)
        cvbooster = serializer.booster
        encoder = serializer.encoders
        threshold_1 = serializer.threshold_1
        threshold_2 = serializer.threshold_2
        holdout_validate(cfg, cvbooster, encoder, threshold_1, threshold_2)
        return
    if not args.inference_only:
        train(cfg)
    inference(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--inference_only", "-i", action="store_true")
    parser.add_argument("--validate_only", "-v", action="store_true")
    parser.add_argument("--lgbm_path", "-l", default="", type=str)
    parser.add_argument("--camaro_path", "-c", default="", type=str)
    parser.add_argument("--camaro2_path", "-c2", default="", type=str)
    parser.add_argument("--camaro3_path", "-c3", default="", type=str)
    parser.add_argument("--camaro4_path", "-c4", default="", type=str)
    parser.add_argument("--kmat_end_path", "-e", default="", type=str)
    parser.add_argument("--kmat_side_path", "-s", default="", type=str)
    parser.add_argument("--enable_multiprocess", "-m", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
