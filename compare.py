"""
compare.py - Train all four algorithms, cache them on disk, and compare them
in the same mission room.

    python compare.py            # trains (about 1.5 minutes the first time) and prints the comparison
"""
import hashlib
import json
import os
import pickle
import numpy as np
import pandas as pd
import config
import pipeline
import world
import algorithms as A

CACHE_DIR = os.path.join(config.OUTPUT_DIR, "cache")


def _cached(name, key, fn):
    os.makedirs(CACHE_DIR, exist_ok=True)
    h = hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    path = os.path.join(CACHE_DIR, f"{name}_{h}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:            # e.g. cache written by a different NumPy version: retrain
            pass
    out = fn()
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def train_all(vi_result, rewards=config.REWARDS, progress=None):
    """Returns {key: {"controller": ..., "details": ...}} for vi, mc, hj, ga."""
    variant, gamma = vi_result["variant"], vi_result["gamma"]
    sensors = config.VARIANTS[variant]["sensors"]
    world_key = {"walls": world.WALLS, "pillars": world.PILLARS, "effects": config.ACTION_EFFECTS,
                 "cp": world.CHECKPOINTS}

    def p(stage, lo, hi):
        return (lambda frac, msg: progress(lo + (hi - lo) * frac, msg)) if progress else None

    mc = _cached("mc", {"v": variant, "r": rewards, **world_key},
                 lambda: A.monte_carlo_best(sensors, rewards, progress=p("mc", 0.0, 0.45)))
    hj = _cached("hj", world_key, lambda: A.hooke_jeeves(progress=p("hj", 0.45, 0.55)))
    ga = _cached("ga", {"v": variant, **world_key}, lambda: A.genetic(sensors, progress=p("ga", 0.55, 1.0)))
    if progress:
        progress(1.0, "All algorithms trained")
    return {
        "vi": {"controller": A.value_iteration_controller(vi_result), "details": vi_result},
        "mc": {"controller": mc["controller"], "details": mc},
        "hj": {"controller": hj["controller"], "details": hj},
        "ga": {"controller": ga["controller"], "details": ga},
    }


def missions(models, noise=False, seed=0):
    out = {}
    for k, m in models.items():
        metrics, traj = world.run_mission(m["controller"], noise=noise, seed=seed)
        out[k] = {"metrics": metrics, "traj": traj}
    return out


def robustness(models, runs=10):
    rows = []
    for k, m in models.items():
        res = [world.run_mission(m["controller"], noise=True, seed=100 + s, record=False)[0] for s in range(runs)]
        done = [r for r in res if r["completed"]]
        rows.append({"algorithm": A.NAMES[k], "completed": f"{len(done)}/{runs}",
                     "mean steps": round(np.mean([r["steps"] for r in done]), 1) if done else None,
                     "mean path (m)": round(np.mean([r["distance_m"] for r in done]), 1) if done else None,
                     "collisions": int(sum(r["collisions"] for r in res)),
                     "ideal distance (%)": round(np.mean([r["ideal_band_pct"] for r in res]), 1),
                     "mean fitness": round(np.mean([r["fitness"] for r in res]), 1)})
    return pd.DataFrame(rows)


def dataset_agreement(models, df):
    """How often each controller picks the real robot's action for the recorded readings."""
    out = {}
    cols = [c for c in config.ALL_SENSORS if c in df.columns]
    recs = df[cols].to_dict("records")
    for k, m in models.items():
        pred = np.array([m["controller"](r) for r in recs])
        out[k] = float((pred == df["Class"].to_numpy()).mean())
    return out


if __name__ == "__main__":
    r = pipeline.train()
    models = train_all(r, progress=lambda f, m: print(f"\r{m:<50}", end=""))
    print()
    ms = missions(models)
    print(pd.DataFrame({A.NAMES[k]: v["metrics"] for k, v in ms.items()}).T.to_string())
    print("\nRobustness (10 noisy runs):\n", robustness(models).to_string(index=False))
    print("\nAgreement with the real robot (test data):",
          {A.NAMES[k]: f"{v:.1%}" for k, v in dataset_agreement(models, r["test_df"]).items()})


def view_payload(runs, start_view="2d"):
    """JSON-ready data for components/mission_view.html."""
    algos = []
    for k, v in runs.items():
        tr, m = v["traj"], v["metrics"]
        algos.append({
            "key": k, "name": A.NAMES[k], "short": A.SHORT[k], "color": A.COLORS[k],
            "x": np.round(tr["x"], 3).tolist(), "y": np.round(tr["y"], 3).tolist(),
            "th": np.round(tr["th"], 4).tolist(),
            "a": [config.ACTIONS.index(a) for a in tr["a"]],
            "r": np.round(np.array(tr["r"]), 2).tolist(),
            "metrics": {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else float(vv)) for kk, vv in m.items()},
        })
    geo = json.loads(json.dumps(world.geometry(), default=float))
    return {"geo": geo, "algos": algos, "actions": config.ACTIONS, "arcDeg": config.SENSOR_ARC_DEG,
            "startView": start_view}


def view_html(runs, start_view="2d"):
    with open(os.path.join(config.BASE_DIR, "components", "mission_view.html")) as f:
        return f.read().replace("__DATA__", json.dumps(view_payload(runs, start_view)))
