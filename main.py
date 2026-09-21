"""
main.py - Run the complete exercise from the command line.

  python main.py                      # 4-sensor dataset, gamma 0.9, tol 1e-6
  python main.py --variant 2          # front + left sensors only
  python main.py --gamma 0.95 --tolerance 1e-8
  streamlit run app.py                # the interactive dashboard
"""
import argparse
import os
import config
import pipeline
from value_iteration import save_results
from evaluate import plot_convergence


def main():
    ap = argparse.ArgumentParser(description="MDP + Value Iteration wall-following robot")
    ap.add_argument("--variant", choices=list(config.VARIANTS), default=config.DEFAULT_VARIANT)
    ap.add_argument("--gamma", type=float, default=config.GAMMA)
    ap.add_argument("--tolerance", type=float, default=config.TOLERANCE)
    a = ap.parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    bar = "=" * 64

    print(f"{bar}\nSTEP 1  Load dataset: {config.VARIANTS[a.variant]['file']}\n{bar}")
    print(f"STEP 2  Build MDP  |  STEP 3  Value Iteration (gamma={a.gamma}, tol={a.tolerance})\n{bar}")
    r = pipeline.train(a.variant, a.gamma, a.tolerance, verbose=True)
    m = r["mdp"]
    print(f"  {len(r['df'])} rows -> train {len(r['train_df'])} / test {len(r['test_df'])}")
    print(" ", m.summary())

    m.save()
    save_results(m, r["V"], r["policy"], r["Q"], r["history"])
    plot_convergence(r["history"], a.tolerance, f"{config.OUTPUT_DIR}/convergence_plot.png")

    print(f"{bar}\nSTEP 4  Evaluation\n{bar}")
    print(f"  Policy agrees with the real robot: train {r['acc_train']:.1%} | test {r['acc_test']:.1%}")
    print("\n", r["report"].to_string(index=False))
    print("\n", r["confusion"].to_string())
    with open(f"{config.OUTPUT_DIR}/evaluation_report.txt", "w") as f:
        f.write(f"variant={a.variant} gamma={a.gamma} tolerance={a.tolerance} "
                f"iterations={len(r['history'])}\n{m.summary()}\n")
        f.write(f"Agreement train={r['acc_train']:.4f} test={r['acc_test']:.4f}\n\n")
        f.write(r["report"].to_string(index=False) + "\n\n" + r["confusion"].to_string() + "\n")
    print(f"\nSaved to {config.OUTPUT_DIR}: optimal_value_function.csv, optimal_policy.csv, "
          "transition_probabilities.csv, rewards.csv, convergence_history.csv, "
          "convergence_plot.png, evaluation_report.txt")


if __name__ == "__main__":
    main()
