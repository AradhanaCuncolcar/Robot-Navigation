"""
value_iteration.py - Value Iteration (Bellman optimality updates).

    V_{k+1}(s) = max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) V_k(s') ]

Stops when  max_s |V_{k+1}(s) - V_k(s)| < tolerance.
"""
import numpy as np
import pandas as pd
import config


def value_iteration(P, R, gamma=config.GAMMA, tol=config.TOLERANCE,
                    max_iter=config.MAX_ITERATIONS, verbose=True):
    V = np.zeros(P.shape[0])
    history = []
    for it in range(1, max_iter + 1):
        V_new = (R + gamma * (P @ V)).max(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        history.append(delta)
        V = V_new
        if verbose and (it <= 5 or it % 25 == 0):
            print(f"  iteration {it:4d}   max|dV| = {delta:.8f}")
        if delta < tol:
            if verbose:
                print(f"  converged after {it} iterations (dV = {delta:.2e} < {tol})")
            break
    Q = R + gamma * (P @ V)
    return V, Q.argmax(axis=1), Q, history


def value_df(mdp, V):
    return pd.DataFrame({"state_id": range(mdp.nS), "state": [mdp.space.name(i) for i in range(mdp.nS)],
                         "optimal_value": np.round(V, 6)})


def policy_df(mdp, policy, Q):
    df = pd.DataFrame({"state_id": range(mdp.nS), "state": [mdp.space.name(i) for i in range(mdp.nS)],
                       "optimal_action": [mdp.actions[a] for a in policy], "visits_in_data": mdp.visits})
    for j, a in enumerate(mdp.actions):
        df[f"Q[{a}]"] = np.round(Q[:, j], 4)
    return df


def save_results(mdp, V, policy, Q, history, out_dir=config.OUTPUT_DIR):
    value_df(mdp, V).to_csv(f"{out_dir}/optimal_value_function.csv", index=False)
    policy_df(mdp, policy, Q).to_csv(f"{out_dir}/optimal_policy.csv", index=False)
    pd.DataFrame({"iteration": range(1, len(history) + 1), "max_delta": history}) \
        .to_csv(f"{out_dir}/convergence_history.csv", index=False)
