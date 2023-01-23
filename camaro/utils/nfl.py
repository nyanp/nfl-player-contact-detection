import numpy as np


def step_to_frame(step):
    step0_frame = 300
    fps_frame = 59.94
    fps_step = 10
    return int(round(step * fps_frame / fps_step + step0_frame))


def frame_to_step(frame):
    step0_frame = 300
    fps_frame = 59.94
    fps_step = 10
    return int(round((frame - step0_frame) * fps_step / fps_frame))


def cast_player_id(df):
    # RAM消費を減らしたいので、Gを-1に置換して整数で持つ。
    if "nfl_player_id_2" in df.columns:
        df.loc[df["nfl_player_id_2"] == "G", "nfl_player_id_2"] = "-1"

    for c in ["nfl_player_id", "nfl_player_id_1", "nfl_player_id_2"]:
        if c in df.columns:
            df[c] = df[c].astype(np.int32)

    return df


def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["game"] = df["game_play"].str[:5].astype(int)
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return cast_player_id(df)
