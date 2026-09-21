"""
mdp_builder.py - Turn the sensor dataset into a Markov Decision Process.

  S : every combination of sensor bins (e.g. front=near, left=ideal, ...)
  A : the 4 movement classes
  P : P(s'|s,a) counted from consecutive rows of the time-ordered dataset
  R : R(s,a) = E[r(s') | s,a] + w * P_expert(a|s)
"""
import itertools
import numpy as np
import pandas as pd
import config


class StateSpace:
    def __init__(self, sensors=config.ALL_SENSORS, bins=config.BINS):
        self.sensors = list(sensors)
        self.bins = {s: bins[s] for s in self.sensors}
        self.states = list(itertools.product(*[self.bins[s][1] for s in self.sensors]))
        self.index = {st: i for i, st in enumerate(self.states)}
        self.n = len(self.states)
        self.vectors = np.array([[self.bins[s][1].index(l) for s, l in zip(self.sensors, st)]
                                 for st in self.states])

    def bin_of(self, sensor, value):
        edges, labels = self.bins[sensor]
        return labels[int(np.searchsorted(edges, value, side="right"))]

    def encode(self, readings):
        """readings: values in the order of self.sensors"""
        return self.index[tuple(self.bin_of(s, v) for s, v in zip(self.sensors, readings))]

    def encode_df(self, df):
        return np.array([self.encode(r) for r in df[self.sensors].to_numpy()])

    def name(self, i):
        return "|".join(f"{s.replace('SD_', '')}={l}" for s, l in zip(self.sensors, self.states[i]))

    def labels(self, i):
        return dict(zip(self.sensors, self.states[i]))


class MDP:
    def __init__(self, df, sensors=config.ALL_SENSORS, gamma=config.GAMMA, rewards=config.REWARDS):
        self.space = StateSpace(sensors)
        self.actions = list(config.ACTIONS)
        self.nS, self.nA = self.space.n, len(self.actions)
        self.gamma, self.rewards = gamma, rewards

        s = self.space.encode_df(df)
        a = df["Class"].map({x: i for i, x in enumerate(self.actions)}).to_numpy()
        self.state_seq, self.action_seq = s, a
        self.visits = np.bincount(s, minlength=self.nS)
        self.P = self._estimate_transitions(s, a)
        self.expert = self._expert_policy_probs(s, a)
        self.r_state = np.array([self.state_reward(i) for i in range(self.nS)])
        self.R = self.P @ self.r_state + rewards["imitation_weight"] * self.expert

    def _estimate_transitions(self, s, a):
        counts = np.zeros((self.nS, self.nA, self.nS))
        np.add.at(counts, (s[:-1], a[:-1], s[1:]), 1)
        P = np.zeros_like(counts)
        marginal = counts.sum(axis=1)
        for i in range(self.nS):
            for j in range(self.nA):
                tot = counts[i, j].sum()
                if tot > 0:
                    P[i, j] = counts[i, j] / tot
                elif marginal[i].sum() > 0:          # action never tried here
                    P[i, j] = marginal[i] / marginal[i].sum()
                else:                                # state never seen: stay put
                    P[i, j, i] = 1.0
        self.transition_counts = counts
        return P

    def _expert_policy_probs(self, s, a):
        c = np.zeros((self.nS, self.nA))
        np.add.at(c, (s, a), 1)
        tot = c.sum(axis=1, keepdims=True)
        return np.divide(c, tot, out=np.zeros_like(c), where=tot > 0)

    def state_reward(self, i):
        lab = self.space.labels(i)
        return self.rewards["left"][lab["SD_left"]] + self.rewards["front"][lab["SD_front"]]

    def nearest_known(self):
        """For every state, the closest state that was observed in the data."""
        known = np.nonzero(self.visits > 0)[0]
        w = np.array([2 if s in ("SD_front", "SD_left") else 1 for s in self.space.sensors])
        out = np.empty(self.nS, dtype=int)
        for i in range(self.nS):
            if self.visits[i] > 0:
                out[i] = i
            else:
                d = (np.abs(self.space.vectors[known] - self.space.vectors[i]) * w).sum(axis=1)
                out[i] = known[np.argmin(d)]
        return out

    def transitions_df(self):
        rows = []
        for i, j, k in zip(*np.nonzero(self.P)):
            rows.append([self.space.name(i), self.actions[j], self.space.name(k),
                         round(self.P[i, j, k], 6), int(self.transition_counts[i, j, k])])
        return pd.DataFrame(rows, columns=["state", "action", "next_state", "probability", "count"])

    def rewards_df(self):
        r = pd.DataFrame(self.R, columns=self.actions).round(4)
        r.insert(0, "state", [self.space.name(i) for i in range(self.nS)])
        r.insert(1, "visits", self.visits)
        return r

    def save(self, out_dir=config.OUTPUT_DIR):
        self.transitions_df().to_csv(f"{out_dir}/transition_probabilities.csv", index=False)
        self.rewards_df().to_csv(f"{out_dir}/rewards.csv", index=False)

    def summary(self):
        return (f"States: {self.nS} ({int((self.visits > 0).sum())} observed) | Actions: {self.nA} | "
                f"Transitions observed: {int(self.transition_counts.sum())}")
