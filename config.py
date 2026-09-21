"""
config.py - All tunable settings for the MDP / Value Iteration project.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ----------------------------------------------------------------- dataset
ALL_SENSORS = ["SD_front", "SD_left", "SD_right", "SD_back"]
ACTIONS = ["Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn"]

# The Kaggle archive ships three files recorded at the same time steps.
VARIANTS = {
    "4": {"file": "sensor_readings_4.csv", "sensors": ALL_SENSORS,
          "label": "4 simplified distances (front, left, right, back)"},
    "2": {"file": "sensor_readings_2.csv", "sensors": ["SD_front", "SD_left"],
          "label": "2 simplified distances (front, left)"},
}
RAW24_FILE = "sensor_readings_24.csv"
# Columns of sensor_readings_24.csv whose minimum gives each simplified distance.
# Verified against sensor_readings_4.csv: 100% exact match on all 5456 rows.
ARC_COLUMNS_24 = {"SD_front": [10, 11, 12, 13, 14], "SD_left": [17, 18, 19],
                  "SD_right": [4, 5, 6, 7, 8], "SD_back": [22, 23]}

DEFAULT_VARIANT = "4"
DATA_PATH = os.path.join(DATA_DIR, VARIANTS[DEFAULT_VARIANT]["file"])

# ----------------------------------------------------------------- state space
# Robot follows the wall on its LEFT (clockwise, per the dataset README),
# keeping SD_left ~0.5-0.9 m, and turns sharp right when SD_front < ~0.9 m.
BINS = {
    "SD_front": ([0.9, 1.5],      ["near", "mid", "far"]),
    "SD_left":  ([0.5, 0.9, 1.5], ["too_close", "ideal", "far", "lost"]),
    "SD_right": ([1.2, 2.0],      ["near", "mid", "far"]),
    "SD_back":  ([1.0],           ["near", "far"]),
}

# ----------------------------------------------------------------- rewards
REWARDS = {
    "left":  {"too_close": -2.0, "ideal": 2.0, "far": -0.5, "lost": -2.0},
    "front": {"near": -3.0, "mid": 0.0, "far": 0.5},
    "imitation_weight": 3.0,   # bonus for choosing the expert robot's action
}

# ----------------------------------------------------------------- value iteration
GAMMA = 0.9
TOLERANCE = 1e-6
MAX_ITERATIONS = 10_000
TRAIN_FRACTION = 0.8

# ----------------------------------------------------------------- simulator
SENSOR_RANGE = 5.0
SENSOR_ARC_DEG = 60
RAYS_PER_ARC = 9
ROBOT_RADIUS = 0.22
ACTION_EFFECTS = {             # (turn degrees, forward metres) per step
    "Move-Forward":      (0.0,   0.10),
    "Slight-Right-Turn": (-8.0,  0.07),
    "Sharp-Right-Turn":  (-25.0, 0.02),
    "Slight-Left-Turn":  (8.0,   0.07),   # tight enough to wrap around convex corners and doorways
}
