import numpy as np
import pandas as pd


class PairRollHolder:
    def __init__(self, window_size, all_players):
        self.window_size = window_size
        self.window_dev = int(window_size // 2)
        self.pair_vals = {str(p1) + str(p2): [0] * window_size for p1 in all_players for p2 in all_players}
        self.pair_counts = {str(p1) + str(p2): [0] * window_size for p1 in all_players for p2 in all_players}
        self.roll_map = {}

    def ready_next(self):
        self.pair_vals = {key: val[1:] + [0] for key, val in self.pair_vals.items()}
        self.pair_counts = {key: val[1:] + [0] for key, val in self.pair_counts.items()}

    def add_if_not_nan(self, p1, p2, val):
        if not np.isnan(val):
            self.pair_vals[str(p1) + str(p2)][-1] = val
            self.pair_counts[str(p1) + str(p2)][-1] = 1
            self.pair_vals[str(p2) + str(p1)][-1] = val
            self.pair_counts[str(p2) + str(p1)][-1] = 1

    def end_of_step(self, step):
        self.roll_map.update({f"{step-self.window_dev}_" + key: sum(val) /
                             (sum(self.pair_counts[key]) + 1e-7) for key, val in self.pair_vals.items()})


def interpolate_features(train_df, window_size=11, columns_to_roll=['cnn_pred_Sideline', 'cnn_pred_Endzone']):
    """
    game_play, pair(player1, player2), でstep方向にroll (現状average。ガウス重みがベター？)とる。
    画像系特徴の 予測欠損補間が主目的。特にGround接触時はヘルメットが隠れやすく、画像からの予測が存在しないところがあるはず。
    """
    df_roll_features = []
    c = 0
    num_gp = len(train_df["game_play"].unique())
    feature_names = [f"{column}_roll{window_size}" for column in columns_to_roll]
    for gp, df_gp in train_df.groupby("game_play"):
        print(f"\r{c} / {num_gp} game_play", end="")
        c += 1
        p1_uniq = list(df_gp["nfl_player_id_1"].unique())
        p2_uniq = list(df_gp["nfl_player_id_2"].unique())
        step_uniq = list(df_gp["step"].unique())
        df_gp["key"] = df_gp["step"].astype(str) + "_" + df_gp["nfl_player_id_1"].astype(str) + df_gp["nfl_player_id_2"].astype(str)
        min_step = np.min(step_uniq)
        max_step = np.max(step_uniq)
        all_players = list(set(p1_uniq + p2_uniq))
        dev = window_size // 2
        roll_feature_holders = [PairRollHolder(window_size, all_players) for _ in columns_to_roll]

        for step in range(min_step, max_step + 1 + dev):
            df_gp_step = df_gp[df_gp["step"] == step]
            p1s = df_gp_step["nfl_player_id_1"].values
            p2s = df_gp_step["nfl_player_id_2"].values
            values = [df_gp_step[column].values for column in columns_to_roll]
            for holder in roll_feature_holders:
                holder.ready_next()
            for p1p2values in zip(p1s, p2s, *values):
                p1 = p1p2values[0]
                p2 = p1p2values[1]
                each_val = p1p2values[2:]
                for holder, v in zip(roll_feature_holders, each_val):
                    holder.add_if_not_nan(p1, p2, v)
            for holder in roll_feature_holders:
                holder.end_of_step(step)
        for column, holder in zip(columns_to_roll, roll_feature_holders):
            df_gp[f"{column}_roll{window_size}"] = df_gp["key"].map(holder.roll_map)
        df_roll_features.append(df_gp[["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"] + feature_names])

    df_roll_features = pd.concat(df_roll_features, axis=0)
    # merge遅いし重いし良くないかも…
    train_df = pd.merge(train_df, df_roll_features, how="left", on=["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"])
    return train_df
