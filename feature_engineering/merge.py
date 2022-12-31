from feature_engineering.cnn_feats import add_cnn_features
from feature_engineering.interpolate_features import interpolate_features
from feature_engineering.point_set_matching import add_p2p_matching_features
from feature_engineering.table import (
    add_aspect_ratio_feature, add_basic_features, add_bbox_features,
    add_distance_around_player,
    add_misc_features_after_agg, add_shift_of_player, add_step_feature,
    add_t0_feature, add_tracking_agg_features, select_close_example,
    tracking_prep)
from utils.general import timer
from utils.nfl import merge_tracking


def make_features(df, tracking, regist, df_args=None):
    if df_args is None:
        df_args = []
    with timer("merge"):
        tracking = tracking_prep(tracking)
        feature_df = merge_tracking(
            df,
            tracking,
            [
                "team", "position", "x_position", "y_position",
                "speed", "distance", "direction", "orientation", "acceleration",
                "sa",
                # "direction_p1_diff", "direction_m1_diff",
                # "orientation_p1_diff", "orientation_m1_diff",
                # "distance_p1", "distance_m1"
            ]
        )

    print(feature_df.shape)

    with timer("tracking_agg_features"):
        feature_df = add_basic_features(feature_df)
        feature_df = add_cnn_features(feature_df, *df_args)
        feature_df = add_p2p_matching_features(feature_df, regist)
        feature_df = interpolate_features(
            feature_df,
            window_size=11,
            columns_to_roll=[
                'camaro_pred',
                'cnn_pred_Sideline',
                'cnn_pred_Endzone',
                'x_rel_position_offset_on_img_End',
                'y_rel_position_offset_on_img_Side'])

        feature_df, close_sample_index = select_close_example(feature_df)

        feature_df = add_bbox_features(feature_df)
        feature_df = add_step_feature(feature_df, tracking)
        feature_df = add_tracking_agg_features(feature_df, tracking)
        feature_df = add_t0_feature(feature_df, tracking)
        feature_df = add_distance_around_player(feature_df)
        feature_df = add_aspect_ratio_feature(feature_df, False)
        feature_df = add_misc_features_after_agg(feature_df)
        feature_df = add_shift_of_player(feature_df, tracking, [-5, 5, 10], add_diff=True, player_id="1")
        feature_df = add_shift_of_player(feature_df, tracking, [-5, 5], add_diff=True, player_id="2")
    print(feature_df.shape)
    # print(feature_df.columns.tolist())
    return feature_df, close_sample_index
