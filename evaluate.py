"""
evaluate.py - Compare the learned policy with the real robot's actions.
"""
import numpy as np
import pandas as pd


def policy_agreement(mdp, policy, df):
    s = mdp.space.encode_df(df)
    pred = np.array([mdp.actions[policy[i]] for i in s])
    true = df["Class"].to_numpy()
    return float((pred == true).mean()), pred, true


def confusion(true, pred, labels):
    m = pd.crosstab(pd.Series(true, name="True"), pd.Series(pred, name="Policy"))
    return m.reindex(index=labels, columns=labels, fill_value=0)


def per_class_report(true, pred, labels):
    rows = []
    for c in labels:
        tp = np.sum((pred == c) & (true == c))
        prec = tp / max(np.sum(pred == c), 1)
        rec = tp / max(np.sum(true == c), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        rows.append([c, round(prec, 3), round(rec, 3), round(f1, 3), int(np.sum(true == c))])
    return pd.DataFrame(rows, columns=["class", "precision", "recall", "f1", "support"])


def plot_convergence(history, tol, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.semilogy(range(1, len(history) + 1), history, lw=2)
    plt.axhline(tol, color="red", ls="--", label=f"tolerance = {tol}")
    plt.xlabel("Iteration"); plt.ylabel("max |V_k+1 - V_k|")
    plt.title("Value Iteration convergence"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
