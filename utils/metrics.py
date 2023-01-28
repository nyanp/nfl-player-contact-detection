import pandas as pd
from tqdm.auto import tqdm
from scipy.optimize import minimize
from sklearn.metrics import matthews_corrcoef
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def search_best_threshold(y_true, y_pred):
    def func(x_list):
        score = matthews_corrcoef(y_true, y_pred > x_list[0])
        return -score

    x0 = [0.5]
    result = minimize(func, x0, method="nelder-mead")

    return result.x[0]


def binarize_pred(y_pred, threshold, threshold2, threshold2_mask):
    return ~threshold2_mask * (y_pred > threshold) + \
        threshold2_mask * (y_pred > threshold2)


def search_best_threshold_pair(y_true, y_pred, is_ground):
    def func(x_list):
        score = matthews_corrcoef(y_true, binarize_pred(
            y_pred, x_list[0], x_list[1], is_ground))
        return -score

    x0 = [0.5, 0.5]
    result = minimize(func, x0, method="nelder-mead")

    return result.x[0], result.x[1]


def search_best_threshold_pair_optuna(y_true, y_pred, is_ground, n_trials=100):
    def objective(trial):
        th1 = trial.suggest_float('th1', 0.1, 0.6)
        th2 = trial.suggest_float('th2', 0.1, 0.6)
        score = matthews_corrcoef(y_true, binarize_pred(y_pred, th1, th2, is_ground))
        return -score
    study = optuna.create_study()  # Create a new study.
    study.optimize(objective, n_trials=n_trials)  # Invoke optimization of the objective function.
    return study.best_params["th1"], study.best_params["th2"]


def metrics(
        y_true,
        y_pred,
        threshold=None,
        threshold2=None,
        threshold2_mask=None):
    if threshold is None:
        threshold = search_best_threshold(y_true, y_pred)

    if threshold2 is not None and threshold2_mask is not None:
        return matthews_corrcoef(
            y_true,
            binarize_pred(
                y_pred,
                threshold,
                threshold2,
                threshold2_mask))
    else:
        return matthews_corrcoef(y_true, y_pred > threshold)


def summarize_per_play_mcc(labels):
    plays = []
    mccs = []
    n_contacts = []

    l = labels.set_index("game_play")

    for play in tqdm(labels["game_play"].unique()):
        p = l.loc[play]
        mccs.append(matthews_corrcoef(p["contact"], p["y_pred"].fillna(False)))
        plays.append(play)
        n_contacts.append(p["contact"].sum())

    return pd.DataFrame({
        "play": plays,
        "mcc": mccs,
        "number_of_contacts": n_contacts
    })
