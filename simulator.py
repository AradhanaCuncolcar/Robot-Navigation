"""
simulator.py - Run the entry-to-exit mission for every algorithm from the
command line (the Streamlit app has the full visual version).

  python simulator.py                  # table of results, clean sensors
  python simulator.py --noise --seed 3
  python simulator.py --plot           # saves outputs/mission_paths.png
"""
import argparse
import os
import pandas as pd
import config
import pipeline
import compare
import world
import algorithms as A


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 8))
    for x0, y0, x1, y1 in world.WALLS:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#26323F"))
    for p in world.PILLARS:
        ax.add_patch(plt.Polygon(world._pillar_poly(p), color="#8E979F"))
    for k, v in runs.items():
        ax.plot(v["traj"]["x"], v["traj"]["y"], color=A.COLORS[k], lw=2, label=A.NAMES[k])
    ax.annotate("Entry", (0.8, -2.4), color="#1E8A5A", weight="bold")
    ax.annotate("Exit", (3.9, -2.4), color="#C0392B", weight="bold")
    ax.set_aspect("equal"); ax.legend(loc="upper right"); ax.set_title("Entry-to-exit mission")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(config.VARIANTS), default=config.DEFAULT_VARIANT)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    models = compare.train_all(pipeline.train(a.variant),
                               progress=lambda f, m: print(f"\r{m:<48}", end="", flush=True))
    print()
    runs = compare.missions(models, noise=a.noise, seed=a.seed)
    print(pd.DataFrame({A.NAMES[k]: v["metrics"] for k, v in runs.items()}).T.to_string())
    if a.plot:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        out = os.path.join(config.OUTPUT_DIR, "mission_paths.png")
        plot(runs, out)
        print("saved", out)
