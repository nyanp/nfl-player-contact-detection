from typing import Optional

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, matthews_corrcoef

matplotlib.rcParams['animation.embed_limit'] = 2**128


def find_fold(game_play: str) -> int:
    game_fold = pd.read_csv("../input/game_fold.csv")
    return game_fold[game_fold["game"] == int(game_play[:5])]["fold"].iloc[0]


def find_model(serializer, game_play: str):
    return serializer.booster.boosters[find_fold(game_play)]


class EDAConfig:
    def __init__(self,
                 serializer,
                 tracking: pd.DataFrame,
                 labels: pd.DataFrame,
                 game_play: str,
                 x_train: np.ndarray,
                 is_hard_sample: pd.Series,
                 player_id_1: Optional[int] = None,
                 player_id_2: Optional[int] = None,
                 to_g: bool = True):
        self.game_play = game_play
        self.booster = find_model(serializer, game_play)

        # tracking df
        self.play = tracking[tracking["game_play"] == game_play].sort_values(by=["nfl_player_id", "step"]).reset_index(drop=True)
        self.labels = labels[(labels["game_play"] == game_play)]
        self.contacts = labels[(labels["contact"] == 1) & (labels["game_play"] == game_play)].reset_index(drop=True)

        if player_id_1 and player_id_2:
            self.player_id_1 = player_id_1
            self.player_id_2 = player_id_2
        else:
            if to_g:
                self.player_id_2 = -1
                self.player_id_1 = self.contacts[self.contacts["nfl_player_id_2"] == -1]["nfl_player_id_1"].sample(1).iloc[0]
            else:
                c = self.contacts.sample(1).iloc[0]
                self.player_id_1 = c["nfl_player_id_1"]
                self.player_id_2 = c["nfl_player_id_2"]

        game_play_mask = (
            (labels["game_play"][is_hard_sample] == game_play) & (
                labels["nfl_player_id_1"][is_hard_sample] == self.player_id_1) & (
                labels["nfl_player_id_2"][is_hard_sample] == self.player_id_2)).values
        self.x_train_for_play = x_train[game_play_mask]

        # patch
        self.booster.params["objective"] = "binary"
        explainer = shap.Explainer(self.booster, feature_names=self.booster.feature_name())
        self.shap_values = explainer(self.x_train_for_play)

        self.player_pair = labels[
            (labels["nfl_player_id_1"] == self.player_id_1) &
            (labels["nfl_player_id_2"] == self.player_id_2) &
            (labels["game_play"] == game_play)
        ]
        self.player_pair_contacts = self.player_pair[
            self.player_pair["contact"] == 1
        ]
        self.player_tracking = self.play[
            self.play["nfl_player_id"].isin([self.player_id_1, self.player_id_2])
        ]
        self.player1_tracking = self.play[
            self.play["nfl_player_id"] == self.player_id_1
        ]
        self.hard_sample_step = self.player_pair[self.player_pair["oof"].notnull()]["step"].values

        print(f"target: {self.game_play}")
        print(f"target players: {self.player_id_1} vs {self.player_id_2}")
        print(f"total {len(self.player_pair_contacts)} contacts ({self.player_pair_contacts.step.tolist()})")

        mcc_pair = matthews_corrcoef(self.player_pair["contact"], self.player_pair["y_pred"].fillna(False))
        tn, fp, fn, tp = confusion_matrix(self.player_pair["contact"], self.player_pair["y_pred"].fillna(False)).ravel()
        print(f"mcc (player {self.player_id_1} vs {self.player_id_2}): {mcc_pair:.4f} (tp:{tp},fp:{fp},tn:{tn},fn:{fn})")

        mcc_play = matthews_corrcoef(self.labels["contact"], self.labels["y_pred"].fillna(False))
        tn, fp, fn, tp = confusion_matrix(self.labels["contact"], self.labels["y_pred"].fillna(False)).ravel()
        print(f"mcc (whole game-play): {mcc_play:.4f} (tp:{tp},fp:{fp},tn:{tn},fn:{fn})")

    def contacted_steps(self):
        return list(sorted(set(self.player_pair_contacts["step"])))

    def shap_value_of_step(self, step):
        found = np.where(self.hard_sample_step == step)[0]
        if len(found) == 0:
            return None
        return self.shap_values[found[0], :, 1]

    def feature_contribution_of_step(self, step):
        found = np.where(self.hard_sample_step == step)[0]
        if len(found) == 0:
            return None

        shap = self.shap_values[found[0], :, 1].values
        feature_value = self.x_train_for_play[found[0], :]
        feature_name = self.booster.feature_name()

        return pd.DataFrame({
            "name": feature_name,
            "value": feature_value,
            "contribution": shap
        })


def plot_animation(config, min_step=-20):
    play = config.play

    if True:
        fig = plt.figure(tight_layout=True, figsize=(12, 12))
        axes = []
        gs = fig.add_gridspec(11, 2)
        axes.append(fig.add_subplot(gs[:4, :]))
        axes.append(fig.add_subplot(gs[4:6, :]))
        axes.append(fig.add_subplot(gs[6:8:, 0]))
        axes.append(fig.add_subplot(gs[6:8:, 1]))
        axes.append(fig.add_subplot(gs[8, :]))
        axes.append(fig.add_subplot(gs[9, :]))
        axes.append(fig.add_subplot(gs[10, :]))

    # fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12,12),
    #                         gridspec_kw={'height_ratios': [4, 2, 1, 1, 1]})
    plt.tight_layout()

    if min_step is not None:
        play = play[(play["step"] >= min_step)]

    steps = [s for s in sorted(play["step"].unique())]

    players = sorted(play["nfl_player_id"].unique())

    print(f"steps:{steps}")

    positions = play[["x_position", "y_position"]].values
    xmin = np.min(positions[:, 0])
    xmax = np.max(positions[:, 0])
    ymin = np.min(positions[:, 1])
    ymax = np.max(positions[:, 1])

    positions = positions.reshape((22, len(play) // 22, 2))

    oof = config.player_pair["oof"].values
    y_pred = config.player_pair["y_pred"].values
    y_true = config.player_pair["contact"].values

    player_team = {}
    for p in players:
        player_team[p] = play[play["nfl_player_id"] == p]["team"].iloc[0]

    home_mask = np.array([player_team[p] == "home" for p in players])
    player_mask = np.array([p in [config.player_id_1, config.player_id_2] for p in players])

    def plot_contribution_table(m, ax):
        ax.axis('off')
        ax.axis('tight')

        text = []
        for i, row in m.iterrows():
            text.append([row["name"], f"{row.value:.2f}", f"{row.contribution:.2f}"])
        fig = ax.table(cellText=text, colLabels=m.columns, loc='center')
        fig.auto_set_column_width((-1, 0, 1, 2))

    def animate(i):
        #step = play[(play["step"]<=steps[i])&(steps[i]-n_history<=play["step"])].set_index("nfl_player_id")

        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[3].clear()
        axes[4].clear()
        axes[5].clear()
        axes[6].clear()

        is_contacted = steps[i] in config.contacted_steps()

        x_home = positions[home_mask, i, 0]
        y_home = positions[home_mask, i, 1]

        axes[0].scatter(
            positions[home_mask, i, 0],
            positions[home_mask, i, 1],
            c="blue")

        axes[0].scatter(
            positions[~home_mask, i, 0],
            positions[~home_mask, i, 1],
            c="green")

        axes[0].scatter(
            positions[player_mask, i, 0],
            positions[player_mask, i, 1],
            c="orange",
            edgecolors="red" if is_contacted else "black",
            linewidth=2 if is_contacted else 1
        )

        if steps[i] >= 0 and steps[i] < len(oof):
            if pd.isnull(oof[steps[i]]):
                # easy sample.
                text = "easy sample"
            else:
                # hard sample
                text = f"y_true={is_contacted}, y_pred={y_pred[steps[i]]}, oof={oof[steps[i]]:.4f}"
        else:
            text = "out of play"

        axes[0].set_title(f"step={steps[i]}, {text}")
        axes[0].set_xlim(xmin, xmax)
        axes[0].set_ylim(ymin, ymax)

        track = config.player1_tracking[config.player1_tracking["step"] <= steps[i]]

        axes[1].set_title("pred/gt")
        if steps[i] >= 0 and steps[i] < len(oof):
            axes[1].plot(
                np.arange(steps[i] + 1),
                y_pred[:steps[i] + 1].astype(float)
            )
            axes[1].plot(
                np.arange(steps[i] + 1),
                oof[:steps[i] + 1]
            )
            axes[1].plot(
                np.arange(steps[i] + 1),
                y_true[:steps[i] + 1].astype(float)
            )
            axes[1].legend(["y_pred", "oof", "y_true"])
            axes[1].set_ylim(-0.1, 1.1)
        elif steps[i] <= 0:
            axes[1].text(0.4, 0.5, "(play not started)", size=15)
        else:
            axes[1].text(0.4, 0.5, "(play finished)", size=15)

        fc = config.feature_contribution_of_step(steps[i])
        if fc is not None:
            fc = fc.sort_values(by="contribution")
            plot_contribution_table(fc.iloc[:8], axes[2])
            plot_contribution_table(fc.iloc[-8:][::-1], axes[3])
        else:
            axes[2].axis('off')
            axes[3].axis('off')
        axes[2].set_title("shap contrib (negative top)")
        axes[3].set_title("shap contrib (positive top)")
        axes[4].set_title("orientation/direction")
        axes[4].plot(
            track["step"],
            track["orientation"]
        )
        # axes[2].set_title("direction")
        axes[4].plot(
            track["step"],
            track["direction"]
        )
        axes[4].legend(["orientation", "direction"])
        axes[5].set_title("speed")
        axes[5].plot(
            track["step"],
            track["speed"]
        )
        axes[6].set_title("acceleration/sa")
        axes[6].plot(
            track["step"],
            track["acceleration"]
        )
        axes[6].plot(
            track["step"],
            track["sa"]
        )
        axes[6].legend(["acceleration", "sa"])

    plt.close()

    anim = animation.FuncAnimation(fig, animate, frames=len(steps), interval=200, blit=False)
    return anim
