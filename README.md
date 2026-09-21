# Wall-following robot navigator (MDP, Monte Carlo, Hooke-Jeeves, Genetic Algorithm)

The SCITOS G5 wall-following dataset (Kaggle / UCI) turned into a Markov
decision process and solved with value iteration, then compared with three
other methods in the same simulated hall. The Streamlit dashboard shows the
four robots on an architectural floor plan and in 3D.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py              # opens http://localhost:8501
```

Trained results for the default settings (both sensor variants) are already in
`outputs/cache/`, so the app opens in a few seconds. Changing the dataset
variant or reward weights retrains Monte Carlo (and the GA for a new variant);
that takes 1 to 3 minutes and shows a progress bar.

Command line:

```bash
python main.py                    # the exercise itself: MDP + value iteration -> outputs/*.csv
python compare.py                 # trains all four algorithms and prints the comparison
python simulator.py --plot        # mission table + outputs/mission_paths.png
python simulator.py --noise --seed 3
```

## The mission

A 14 m × 9 m main hall (`world.py`):

- 0.25 m walls with windows, a recessed bay in the north wall and a structural
  column on the east wall
- five free-standing pillars in the middle (four square, one round)
- an entry corridor (closed behind the robot) and a separate exit corridor
  with a 2.4 m double door

Each robot starts in the entry corridor and keeps the wall on its **left**, as
the real robot did. It follows the main walls once, around the bay and the
column, and leaves through the exit. Leaving early, or failing to reach a new
checkpoint for 700 steps, counts as a failed mission.

All four algorithms use the same robot: the same four 60° sonar arcs, the
same four actions (forward, slight right, sharp right, slight left) and the
same physics.

## The four algorithms (`algorithms.py`)

| | Type | Learns from | What it produces |
|---|---|---|---|
| **Value Iteration** | model-based dynamic programming | the Kaggle recording only | state → action table |
| **Monte Carlo** | model-free reinforcement learning | 250 simulated episodes | state → action table |
| **Hooke-Jeeves** | derivative-free pattern search | simulated missions | 3 distance thresholds of a wall-following rule |
| **Genetic Algorithm** | evolutionary search | simulated missions | state → action table |

- **Value Iteration.** P(s′|s,a) is counted from consecutive rows of the
  dataset. Bellman updates run until max |ΔV| is below the threshold. It
  never sees the hall.
- **Monte Carlo control.** Uses an ε-greedy policy (ε 0.4 → 0.02) and
  first-visit returns with γ = 0.97. Half the episodes use exploring starts
  from random points on the route. Rewards are the MDP's wall-distance rewards
  plus +20 per checkpoint, +300 for the exit and −25 per collision. Three
  training runs are made and the one that scores best on separate validation
  missions is kept.
- **Hooke-Jeeves.** Tunes three thresholds: turn sharp right when the front is
  closer than a₁, steer away when the left is closer than a₂, and steer towards
  the wall when the left is farther than a₃. It uses exploratory and pattern
  moves, halving the step when nothing improves. It starts from a guess that
  fails and finds 0.88 / 0.30 / 0.80 m, close to the dataset's own bins.
- **Genetic Algorithm.** Each chromosome is one action per MDP state (72
  genes). The population of 24 evolves for 30 generations using tournament
  selection, uniform crossover, mutation and 2 elites. Each candidate is scored
  on one clean and three noisy missions, so policies that only work in a
  perfect simulator lose.

**Fitness**, used by Hooke-Jeeves and the GA and shown for all four: 100 per
checkpoint + 2 × % of time at the ideal wall distance − 10 per collision +
1000 for reaching the exit − 0.5 per step.

## Results (4-sensor data, 10 noisy missions each)

| | Completed | Mean steps | Mean path | Collisions | Matches real robot |
|---|---|---|---|---|---|
| Value Iteration | 10/10 | 491 | 44.7 m | 0 | 92.1% |
| Monte Carlo | 10/10 | 624 | 49.5 m | 0 | 34.2% |
| Hooke-Jeeves | 10/10 | 500 | 44.8 m | 0 | 81.4% |
| Genetic Algorithm | 10/10 | 541 | 48.7 m | 0 | 40.8% |

Only value iteration learns from the recording, so it copies the real robot
most closely. The others find different policies that also reach the exit.

With the **2-sensor** data (front and left only), value iteration completes
9/10 and the GA has a few collisions. That is a real effect of having less
information about the surroundings.

## Dashboard tabs

- **Mission.** The live comparison.
  - *Floor plan:* poché walls, windows, hatched pillars, door swings, entry
    and exit markings, dimension lines and a north arrow.
  - *3D view:* extruded walls, glass, open doors and signs. Cameras are orbit
    (drag and scroll), follow (a tracking shot showing the robot's face) and
    top.
  - *Robots:* each has a face with eyes that look where it is heading and
    blink. The expression changes on sharp turns and at the finish. Wheels
    turn, the body leans into turns, and there are dust puffs, a sonar ping
    ripple and confetti at the exit.
  - *Controls:* play, pause, speed 0.5×–8×, timeline scrubbing, hide or show
    each robot, and a sensor noise toggle with seed.
  - The table below the view gives this run's results.
- **Algorithms.** A robustness table and agreement with the recorded robot. It
  also has, per algorithm, an explanation and chart: VI convergence, the MC
  learning curve and training runs, HJ search progress and thresholds, and GA
  evolution. The last table compares the policies side by side.
- **Value iteration, Policy explorer, Evaluation, Dataset.** The exercise
  itself: `optimal_value_function.csv`, the policy heat-map, transition
  explorer, confusion matrix, γ and threshold sweeps, and the dataset
  explorer, including the 24 raw sonars.

## Files

```
app.py                        Streamlit dashboard
components/mission_view.html  floor plan + 3D view (canvas and three.js)
world.py                      the hall, sonar ray casting, physics, mission rules, fitness
algorithms.py                 Monte Carlo, Hooke-Jeeves, Genetic Algorithm, controllers
compare.py                    trains and caches all four, missions, robustness
config.py                     bins, rewards, gamma, threshold, robot motion
data_loader.py                reads the three Kaggle files
mdp_builder.py                states, actions, P(s'|s,a), R(s,a)
value_iteration.py            Bellman updates
pipeline.py                   dataset -> MDP -> value iteration -> evaluation
evaluate.py                   agreement with the real robot
main.py                       exercise outputs (CSV, plot, report)
simulator.py                  command-line mission runner
data/                         sensor_readings_2 / 4 / 24.csv
outputs/                      CSV outputs, previews, cache/ with trained models
```

## Notes

- The 3D view loads three.js from cdnjs and needs WebGL. The floor plan
  works offline.
- The mission runs are simulated in Python. The browser replays them with
  smooth interpolation, and the robots enter one after another from the same
  door.
- The robots are drawn larger than their 0.22 m collision radius so the faces
  are readable.
