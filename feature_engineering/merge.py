from feature_engineering.cnn_feats import add_cnn_agg_features, add_cnn_features, add_cnn_shift_diff_features, agg_cnn_feature
from feature_engineering.interpolate_features import interpolate_features
from feature_engineering.point_set_matching import add_p2p_matching_features
from feature_engineering.table import (
    add_aspect_ratio_feature, add_basic_features, add_bbox_features, add_bbox_std_features, add_bbox_std_overlap_feature, add_distance_agg_features,
    add_distance_around_player, add_image_coords_features, add_interceptor_feature,
    add_misc_features_after_agg, add_second_nearest_distance, add_shift_of_player, add_step_feature,
    add_t0_feature, add_tracking_agg_features, select_close_example,
    tracking_prep)
from utils.general import reduce_dtype, timer
from utils.nfl import merge_tracking


def make_features(df, tracking, regist, df_args=None, enable_multiprocess=True):
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
        feature_df = add_distance_agg_features(feature_df)
        feature_df = add_distance_around_player(feature_df, True)
        feature_df = add_second_nearest_distance(feature_df, "1")
        feature_df = add_second_nearest_distance(feature_df, "2")

        feature_df = add_cnn_features(feature_df, *df_args)
        feature_df = add_p2p_matching_features(feature_df, regist)

        base_feature_cols = [
            'cnn_pred_Sideline',
            'cnn_pred_Endzone',
            'camaro_pred',
            # 'camaro_pred2',
        ]
        offset_cols = [
            'x_rel_position_offset_on_img_End',
            'y_rel_position_offset_on_img_Side'
        ]
        feature_df = interpolate_features(
            feature_df,
            window_size=5,
            columns_to_roll=base_feature_cols + offset_cols,
            enable_multiprocess=enable_multiprocess)

        feature_df = interpolate_features(
            feature_df,
            window_size=11,
            columns_to_roll=base_feature_cols + offset_cols,
            enable_multiprocess=enable_multiprocess)

        feature_df = interpolate_features(
            feature_df,
            window_size=21,
            columns_to_roll=base_feature_cols + offset_cols,
            enable_multiprocess=enable_multiprocess)

        feature_df = add_cnn_agg_features(feature_df)

        feature_df = add_cnn_shift_diff_features(feature_df, columns=base_feature_cols)
        feature_df = agg_cnn_feature(feature_df, columns=base_feature_cols)
        feature_df, close_sample_index = select_close_example(feature_df)

        feature_df = add_bbox_features(feature_df)
        feature_df = add_step_feature(feature_df, tracking)
        feature_df = add_tracking_agg_features(feature_df, tracking)
        feature_df = add_t0_feature(feature_df, tracking)
        feature_df = add_distance_around_player(feature_df, False)
        feature_df = add_aspect_ratio_feature(feature_df, False)
        feature_df = add_misc_features_after_agg(feature_df)
        feature_df = add_shift_of_player(feature_df, tracking, [-5, 5, 10], add_diff=True, player_id="1")
        feature_df = add_shift_of_player(feature_df, tracking, [-5, 5], add_diff=True, player_id="2")
        feature_df = add_bbox_std_overlap_feature(feature_df)
        feature_df = add_interceptor_feature(feature_df)
        # feature_df = add_image_coords_features(feature_df)
        # feature_df = add_bbox_std_features(feature_df)
    print(feature_df.shape)
    # print(feature_df.columns.tolist())
    return feature_df, close_sample_index
