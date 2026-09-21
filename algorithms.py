"""
algorithms.py - Four ways to get a wall-following controller, all using the
same robot, sensors, actions and room (world.py):

  Value Iteration  model-based: MDP estimated from the Kaggle dataset,
                   solved with Bellman updates (value_iteration.py)
  Monte Carlo      model-free RL: on-policy first-visit Monte Carlo control
                   with an epsilon-greedy policy, learning Q(s,a) from whole
                   episodes driven in the simulator
  Hooke-Jeeves     derivative-free pattern search over the three distance
                   thresholds of a rule-based controller
  Genetic          evolves the full state -> action table (one gene per MDP
                   state) with tournament selection, crossover and mutation
"""
import numpy as np
import config
import world
from mdp_builder import StateSpace

NAMES = {"vi": "Value Iteration", "mc": "Monte Carlo", "hj": "Hooke-Jeeves", "ga": "Genetic Algorithm"}
COLORS = {"vi": "#2F6FDB", "mc": "#E07A10", "hj": "#139A6B", "ga": "#B0379E"}
SHORT = {"vi": "VI", "mc": "MC", "hj": "HJ", "ga": "GA"}


# ------------------------------------------------------------------ controllers
class TableController:
    """State -> action lookup (used by VI, MC and GA)."""

    def __init__(self, space: StateSpace, table, resolve=None):
        self.space, self.table = space, np.asarray(table, dtype=int)
        self.resolve = np.arange(space.n) if resolve is None else np.asarray(resolve)

    def state(self, r):
        return self.space.encode([r[k] for k in self.space.sensors])

    def __call__(self, r):
        return config.ACTIONS[self.table[self.resolve[self.state(r)]]]


class RuleController:
    """Hand-structured wall follower whose thresholds are tuned by Hooke-Jeeves.
    theta = (turn when front closer than, steer away when left closer than,
             steer towards wall when left farther than)"""

    def __init__(self, theta):
        self.theta = np.asarray(theta, dtype=float)

    def __call__(self, r):
        front_th, left_lo, left_hi = self.theta
        if r["SD_front"] < front_th:
            return "Sharp-Right-Turn"
        if r["SD_left"] < left_lo:
            return "Slight-Right-Turn"
        if r["SD_left"] > left_hi:
            return "Slight-Left-Turn"
        return "Move-Forward"


def nearest_known(space, visits):
    known = np.nonzero(visits > 0)[0]
    if len(known) == 0:
        return np.arange(space.n)
    w = np.array([2 if s in ("SD_front", "SD_left") else 1 for s in space.sensors])
    out = np.empty(space.n, dtype=int)
    for i in range(space.n):
        d = (np.abs(space.vectors[known] - space.vectors[i]) * w).sum(axis=1)
        out[i] = i if visits[i] > 0 else known[np.argmin(d)]
    return out


def evaluate(controller, runs=((False, 0),)):
    """Mean fitness over (noise, seed) runs."""
    return float(np.mean([world.run_mission(controller, n, s, record=False, stall=350,
                                            max_collisions=40)[0]["fitness"] for n, s in runs]))


# ------------------------------------------------------------------ Value Iteration
def value_iteration_controller(result):
    m = result["mdp"]
    return TableController(m.space, result["policy"], result["resolve"])


# ------------------------------------------------------------------ Monte Carlo control
def monte_carlo(sensors, rewards, gamma=0.97, episodes=250, exit_bonus=300.0, explore_starts=0.5, eps_start=0.4, eps_end=0.02,
                seed=1, progress=None):
    space = StateSpace(sensors)
    nS, nA = space.n, len(config.ACTIONS)
    rng = np.random.default_rng(seed)
    Q = np.zeros((nS, nA))
    N = np.zeros((nS, nA))
    r_state = np.array([rewards["left"][space.labels(i)["SD_left"]] +
                        rewards["front"][space.labels(i)["SD_front"]] for i in range(nS)])
    history = []

    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * ep / max(episodes - 1, 1)
        episode = []

        def policy(r):
            s = space.encode([r[k] for k in sensors])
            if rng.random() < eps:
                a = int(rng.integers(nA))
            else:
                q = Q[s]
                a = int(rng.choice(np.flatnonzero(q == q.max())))
            policy.last = (s, a)
            return config.ACTIONS[a]

        def hook(r, a, r_next, ev):
            s, ai = policy.last
            s2 = space.encode([r_next[k] for k in sensors])
            rew = r_state[s2] - 25.0 * ev["collision"] + 20.0 * ev["checkpoint"]
            episode.append((s, ai, rew))

        # exploring starts: half the episodes begin at a random point along the route,
        # so later parts of the mission (like the turn into the exit) get practised often
        k = int(rng.integers(1, len(world.CHECKPOINTS) - 1)) if rng.random() < explore_starts else 0
        start = world.route_start(k) if k else world.START
        m, _ = world.run_mission(policy, noise=True, seed=int(rng.integers(1e9)), record=False,
                                 stall=400, jitter=1.0, step_hook=hook, start=start, cp_start=k)
        if m["completed"]:
            s, a, rew = episode[-1]
            episode[-1] = (s, a, rew + exit_bonus)

        # first-visit Monte Carlo update, computed backwards
        G = 0.0
        first = {}
        for t, (s, a, _) in enumerate(episode):
            first.setdefault((s, a), t)
        for t in range(len(episode) - 1, -1, -1):
            s, a, rew = episode[t]
            G = gamma * G + rew
            if first[(s, a)] == t:
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]
        ret = sum(e[2] for e in episode)
        history.append({"episode": ep + 1, "epsilon": round(eps, 3), "return": round(ret, 1), "start_checkpoint": k,
                        "completed": m["completed"], "checkpoints": m["checkpoints"],
                        "steps": m["steps"]})
        if progress:
            progress((ep + 1) / episodes, f"Monte Carlo episode {ep + 1}/{episodes}")

    visits = N.sum(axis=1)
    table = Q.argmax(axis=1)
    return {"controller": TableController(space, table, nearest_known(space, visits)),
            "Q": Q, "visits": visits, "history": history}


def monte_carlo_best(sensors, rewards, n_runs=3, progress=None, **kw):
    """Train several independent Monte Carlo runs and keep the one that does best on
    validation missions (noise seeds 500+, separate from the robustness test seeds)."""
    runs = []
    for i in range(n_runs):
        p = (lambda f, m, i=i: progress((i + f) / n_runs, f"Monte Carlo run {i + 1}/{n_runs}: {m[12:]}")) \
            if progress else None
        out = monte_carlo(sensors, rewards, seed=1 + i, progress=p, **kw)
        out["validation"] = evaluate(out["controller"], [(False, 0)] + [(True, 500 + s) for s in range(6)])
        out["seed"] = 1 + i
        runs.append(out)
    best = max(runs, key=lambda r: r["validation"])
    best["all_runs"] = [{"training run": r["seed"], "validation fitness": round(r["validation"], 1),
                         "kept": r is best} for r in runs]
    return best


# ------------------------------------------------------------------ Hooke-Jeeves
HJ_BOUNDS = np.array([[0.3, 2.5], [0.2, 1.2], [0.6, 3.0]])
HJ_LABELS = ["Sharp right when front <", "Steer away when left <", "Steer towards wall when left >"]


def hooke_jeeves(x0=(1.4, 0.3, 1.7), step=(0.3, 0.15, 0.3), shrink=0.5, min_step=0.02,
                 max_evals=120, progress=None):
    runs = ((False, 0), (True, 11), (True, 12))

    def f(x):
        return evaluate(RuleController(np.clip(x, HJ_BOUNDS[:, 0], HJ_BOUNDS[:, 1])), runs)

    x = np.clip(np.array(x0, float), HJ_BOUNDS[:, 0], HJ_BOUNDS[:, 1])
    step = np.array(step, float)
    fx = f(x)
    evals, history = 1, [{"evaluation": 1, "best_fitness": fx, "theta": x.round(3).tolist(), "move": "start"}]

    def explore(base, fbase):
        nonlocal evals
        xb, fb = base.copy(), fbase
        for i in range(len(xb)):
            for d in (+1, -1):
                cand = xb.copy(); cand[i] = np.clip(cand[i] + d * step[i], *HJ_BOUNDS[i])
                fc = f(cand); evals += 1
                if fc > fb:
                    xb, fb = cand, fc
                    break
        return xb, fb

    while step.max() > min_step and evals < max_evals:
        xn, fn = explore(x, fx)
        if fn > fx:                                   # pattern moves while they keep improving
            while fn > fx and evals < max_evals:
                x_prev, x, fx = x, xn, fn
                history.append({"evaluation": evals, "best_fitness": fx, "theta": x.round(3).tolist(),
                                "move": "exploratory / pattern"})
                xp = np.clip(x + (x - x_prev), HJ_BOUNDS[:, 0], HJ_BOUNDS[:, 1])
                fp = f(xp); evals += 1
                xn, fn = explore(xp, fp)
        else:
            step *= shrink
            history.append({"evaluation": evals, "best_fitness": fx, "theta": x.round(3).tolist(),
                            "move": f"shrink step to {step.max():.3f}"})
        if progress:
            progress(min(evals / max_evals, 1.0), f"Hooke-Jeeves evaluation {evals}")
    return {"controller": RuleController(x), "theta": x, "fitness": fx, "evaluations": evals,
            "history": history}


# ------------------------------------------------------------------ Genetic Algorithm
def genetic(sensors, pop_size=24, generations=30, p_cross=0.9, p_mut=None, elite=2, tour=3,
            seed=7, progress=None):
    space = StateSpace(sensors)
    nS, nA = space.n, len(config.ACTIONS)
    rng = np.random.default_rng(seed)
    p_mut = p_mut or 1.5 / nS
    runs = ((False, 0), (True, 21), (True, 22), (True, 23))   # several noisy runs so fragile policies lose

    cache = {}

    def fit(ch):                     # identical chromosomes are only simulated once
        key = ch.tobytes()
        if key not in cache:
            cache[key] = evaluate(TableController(space, ch), runs)
        return cache[key]

    pop = rng.integers(nA, size=(pop_size, nS))
    scores = np.array([fit(c) for c in pop])
    history = []
    for g in range(generations):
        order = np.argsort(-scores)
        history.append({"generation": g, "best": float(scores[order[0]]), "mean": float(scores.mean()),
                        "worst": float(scores.min())})
        if progress:
            progress((g + 1) / (generations + 1), f"Genetic algorithm generation {g + 1}/{generations}")
        new = [pop[i].copy() for i in order[:elite]]
        while len(new) < pop_size:
            def pick():
                idx = rng.choice(pop_size, tour, replace=False)
                return pop[idx[np.argmax(scores[idx])]]
            a, b = pick(), pick()
            child = np.where(rng.random(nS) < 0.5, a, b) if rng.random() < p_cross else a.copy()
            mask = rng.random(nS) < p_mut
            child[mask] = rng.integers(nA, size=mask.sum())
            new.append(child)
        pop = np.array(new)
        scores = np.concatenate([scores[order[:elite]], [fit(c) for c in pop[elite:]]])
    best = int(np.argmax(scores))
    history.append({"generation": generations, "best": float(scores[best]), "mean": float(scores.mean()),
                    "worst": float(scores.min())})
    # Genes for states the best robot never visited were never tested by evolution;
    # like VI and MC, those states fall back to the nearest state it did visit.
    ctrl = TableController(space, pop[best])
    visits = np.zeros(nS)
    for n, s in runs + ((True, 31), (True, 32)):
        _, tr = world.run_mission(ctrl, n, s)
        for rd in tr["r"]:
            visits[space.encode([dict(zip(world.SENSOR_DIRS, rd))[k] for k in sensors])] += 1
    ctrl = TableController(space, pop[best], nearest_known(space, visits))
    return {"controller": ctrl, "table": pop[best], "visits": visits,
            "fitness": float(scores[best]), "history": history}
