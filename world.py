"""
world.py - The mission environment: a main hall with thick walls, a recessed
bay, a wall column, free-standing pillars, and separate entry and exit doors.

The robot starts in the entry corridor, follows the wall on its LEFT (as the
real SCITOS G5 did), makes one pass along the main walls and leaves through
the exit corridor. Also contains a fast, vectorised sonar/physics simulator
shared by every algorithm.
"""
import numpy as np
import config

T = 0.25                      # wall thickness (m)
W, H = 14.0, 9.0              # inner size of the hall (m)
ENTRY = (0.0, 1.6)            # door opening in the bottom wall (x from, x to)
EXIT = (3.0, 5.4)               # double door: wide enough for the 60 degree sonar arc to see the gap
BAY = (5.0, 8.0, 1.5)         # recess in the top wall: x from, x to, depth
CORR = 2.6                    # corridor length outside each door

WALLS = [  # solid rectangles (x0, y0, x1, y1)
    (-T, -CORR - T, 0.0, H + T),                      # west wall, continues down the entry corridor
    (-T, -CORR - T, ENTRY[1] + T, -CORR),             # closed lobby door behind the robot
    (ENTRY[1], -CORR - T, ENTRY[1] + T, 0.0),         # entry corridor, east side
    (ENTRY[1], -T, EXIT[0], 0.0),                     # south wall between the doors
    (EXIT[0] - T, -CORR, EXIT[0], 0.0),               # exit corridor, west side
    (EXIT[1], -CORR, EXIT[1] + T, 0.0),               # exit corridor, east side
    (EXIT[1], -T, W + T, 0.0),                        # south wall east of the exit
    (W, -T, W + T, H + T),                            # east wall
    (-T, H, BAY[0], H + T),                           # north wall, west part
    (BAY[0] - T, H, BAY[0], H + BAY[2] + T),          # bay, west side
    (BAY[0] - T, H + BAY[2], BAY[1] + T, H + BAY[2] + T),  # bay, back wall
    (BAY[1], H, BAY[1] + T, H + BAY[2] + T),          # bay, east side
    (BAY[1], H, W + T, H + T),                        # north wall, east part
    (W - 0.55, 3.6, W, 4.6),                          # structural column on the east wall
]
WINDOWS = [(1.0, 4.2, "N"), (9.2, 12.8, "N"), (2.5, 7.0, "W"), (1.5, 7.5, "E")]  # decoration
PILLARS = [  # ("box", cx, cy, half) or ("round", cx, cy, r)
    ("box", 4.0, 3.2, 0.3), ("box", 4.0, 6.2, 0.3), ("round", 7.0, 4.6, 0.35),
    ("box", 10.0, 3.2, 0.3), ("box", 10.0, 6.2, 0.3),
]
START = (0.75, -2.1, np.pi / 2)          # in the entry corridor, facing into the hall
EXIT_LINE_Y = -1.6                        # crossing this inside the exit corridor = mission complete
CHECKPOINTS = [(0.8, 2.0), (0.8, 6.5), (3.0, 8.3), (5.7, 9.6), (7.3, 9.6), (10.5, 8.3),
               (13.3, 7.0), (13.3, 2.5), (12.0, 0.8), (8.5, 0.8), (6.4, 0.8), (4.2, -1.2)]
CHECKPOINT_RADIUS = 1.3


def _pillar_poly(p):
    kind, cx, cy, s = p
    if kind == "box":
        return [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
    a = np.linspace(0, 2 * np.pi, 17)[:-1]
    return [(cx + s * np.cos(t), cy + s * np.sin(t)) for t in a]


def _segments():
    segs = []
    for x0, y0, x1, y1 in WALLS:
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        segs += [(*pts[i], *pts[(i + 1) % 4]) for i in range(4)]
    for p in PILLARS:
        pts = _pillar_poly(p)
        segs += [(*pts[i], *pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    return np.array(segs, dtype=float)


SEGS = _segments()
SENSOR_DIRS = {"SD_front": 0.0, "SD_left": np.pi / 2, "SD_right": -np.pi / 2, "SD_back": np.pi}
_HALF = np.radians(config.SENSOR_ARC_DEG) / 2
_OFFS = np.linspace(-_HALF, _HALF, config.RAYS_PER_ARC)
_REL = np.concatenate([d + _OFFS for d in SENSOR_DIRS.values()])       # 36 ray angles
_X1, _Y1, _X2, _Y2 = SEGS.T
_EX, _EY = _X2 - _X1, _Y2 - _Y1
_A, _AB = SEGS[:, :2], SEGS[:, 2:] - SEGS[:, :2]
_AB2 = (_AB * _AB).sum(1)
_KEYS = list(SENSOR_DIRS)


def sense(x, y, th):
    """Four simplified distances = min over each 60 degree arc (vectorised)."""
    ang = th + _REL
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    den = dx * _EY - dy * _EX
    px, py = _X1 - x, _Y1 - y
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / den
        t = (px * _EY - py * _EX) * inv
        u = (px * dy - py * dx) * inv
    t[~((t > 0) & (u >= 0) & (u <= 1))] = np.inf
    d = np.minimum(t.min(axis=1), config.SENSOR_RANGE).reshape(4, -1).min(axis=1)
    return {_KEYS[0]: float(d[0]), _KEYS[1]: float(d[1]), _KEYS[2]: float(d[2]), _KEYS[3]: float(d[3])}


def clearance(x, y):
    qx, qy = x - _A[:, 0], y - _A[:, 1]
    t = np.clip((qx * _AB[:, 0] + qy * _AB[:, 1]) / _AB2, 0, 1)
    ddx, ddy = qx - t * _AB[:, 0], qy - t * _AB[:, 1]
    return float(np.sqrt((ddx * ddx + ddy * ddy).min()))


def run_mission(controller, noise=False, seed=0, max_steps=2500, stall=700, record=True,
                start=START, jitter=0.0, step_hook=None, max_collisions=None, cp_start=0):
    """Drive `controller(readings) -> action` from the entry to the exit.

    step_hook(readings, action, next_readings, event) lets learning algorithms
    (Monte Carlo) observe every transition.
    """
    rng = np.random.default_rng(seed)
    x, y, th = start
    if jitter:
        x += rng.uniform(-jitter, jitter) * 0.3
        th += rng.uniform(-jitter, jitter) * 0.2
    cp, last_cp_step, col, dist, band = cp_start, 0, 0, 0.0, 0
    traj = {"x": [], "y": [], "th": [], "a": [], "r": []}
    r = sense(x, y, th)
    completed = False
    steps = 0
    for steps in range(1, max_steps + 1):
        if noise:
            r = {k: float(np.clip(v + rng.normal(0, 0.03), 0.05, config.SENSOR_RANGE)) for k, v in r.items()}
        a = controller(r)
        if record:
            traj["x"].append(x); traj["y"].append(y); traj["th"].append(th); traj["a"].append(a)
            traj["r"].append([r[k] for k in SENSOR_DIRS])
        turn, fwd = config.ACTION_EFFECTS[a]
        if noise:
            turn += rng.normal(0, 1.5); fwd *= 1 + rng.normal(0, 0.05)
        th = (th + np.radians(turn) + np.pi) % (2 * np.pi) - np.pi
        nx, ny = x + fwd * np.cos(th), y + fwd * np.sin(th)
        hit = clearance(nx, ny) < config.ROBOT_RADIUS
        if hit:
            col += 1
        else:
            dist += np.hypot(nx - x, ny - y); x, y = nx, ny
        event = {"collision": hit, "checkpoint": False, "done": False}
        if cp < len(CHECKPOINTS) and np.hypot(x - CHECKPOINTS[cp][0], y - CHECKPOINTS[cp][1]) < CHECKPOINT_RADIUS:
            cp += 1; last_cp_step = steps; event["checkpoint"] = True
        left_early = False
        if y < EXIT_LINE_Y and x > ENTRY[1]:        # through the exit door
            if cp >= len(CHECKPOINTS) - 1:
                completed = True
            else:
                left_early = True                      # skipped the lap: mission failed
        r_next = sense(x, y, th)
        if 0.5 <= r_next["SD_left"] < 0.9:
            band += 1
        event["done"] = (completed or left_early or (steps - last_cp_step > stall) or steps == max_steps
                         or (max_collisions is not None and col >= max_collisions))
        if step_hook:
            step_hook(r, a, r_next, event)
        r = r_next
        if event["done"]:
            break
    if record:
        traj["x"].append(x); traj["y"].append(y); traj["th"].append(th); traj["a"].append(a)
        traj["r"].append([r[k] for k in SENSOR_DIRS])
    m = {"completed": completed, "steps": steps, "distance_m": round(dist, 2), "collisions": col,
         "checkpoints": cp, "ideal_band_pct": round(100 * band / steps, 1)}
    m["fitness"] = fitness(m)
    return m, traj


def route_start(k):
    """A start pose on the route at checkpoint k, facing the next checkpoint (for exploring starts)."""
    (x, y), (nx, ny) = CHECKPOINTS[k], CHECKPOINTS[k + 1]
    return (x, y, float(np.arctan2(ny - y, nx - x)))


def fitness(m):
    """One score used by Hooke-Jeeves and the GA, and to compare every algorithm."""
    f = 100 * m["checkpoints"] + 2 * m["ideal_band_pct"] - 10 * m["collisions"]
    if m["completed"]:
        f += 1000 - 0.5 * m["steps"]
    return round(f, 2)


def geometry():
    """Everything the browser needs to draw the room."""
    return {"walls": WALLS, "pillars": [{"kind": k, "cx": cx, "cy": cy, "s": s} for k, cx, cy, s in PILLARS],
            "windows": WINDOWS, "entry": ENTRY, "exit": EXIT, "bay": BAY, "W": W, "H": H, "T": T,
            "corr": CORR, "start": START, "checkpoints": CHECKPOINTS, "exitLineY": EXIT_LINE_Y}
