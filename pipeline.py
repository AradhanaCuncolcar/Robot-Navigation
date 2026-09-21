"""
pipeline.py - One function that runs the full exercise and returns every result.
Used by both main.py (command line) and app.py (Streamlit).
"""
import time
import config
from data_loader import load_dataset
from mdp_builder import MDP
from value_iteration import value_iteration
import evaluate


def train(variant=config.DEFAULT_VARIANT, gamma=config.GAMMA, tol=config.TOLERANCE,
          rewards=config.REWARDS, train_fraction=config.TRAIN_FRACTION, verbose=False):
    df = load_dataset(variant)
    sensors = config.VARIANTS[variant]["sensors"]
    split = int(len(df) * train_fraction)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    mdp = MDP(train_df, sensors=sensors, gamma=gamma, rewards=rewards)
    t0 = time.perf_counter()
    V, policy, Q, history = value_iteration(mdp.P, mdp.R, gamma, tol, verbose=verbose)
    vi_ms = (time.perf_counter() - t0) * 1000

    acc_train, _, _ = evaluate.policy_agreement(mdp, policy, train_df)
    acc_test, pred, true = evaluate.policy_agreement(mdp, policy, test_df)
    return {
        "variant": variant, "sensors": sensors, "gamma": gamma, "tol": tol,
        "df": df, "train_df": train_df, "test_df": test_df,
        "mdp": mdp, "V": V, "policy": policy, "Q": Q, "history": history, "vi_ms": vi_ms,
        "acc_train": acc_train, "acc_test": acc_test,
        "report": evaluate.per_class_report(true, pred, mdp.actions),
        "confusion": evaluate.confusion(true, pred, mdp.actions),
        "resolve": mdp.nearest_known(),
    }
