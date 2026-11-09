"""
sim_basic.py -- bare-bones setup: grid, agents, movement, measurements.
You write the movement (and anything else you want on top).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------- settings
G = 50              # grid is G x G cells
N = 20              # number of agents
STEPS = 2500        # how many time steps
SMOOTHNESS = 6      # how smooth the hidden temperature map is
NOISE = 2.0         # sensor noise (deg C), same for every agent
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- grid: hidden temperature map
field = gaussian_filter(rng.standard_normal((G, G)), SMOOTHNESS / np.sqrt(2))
field = 20 + 5 * (field - field.mean()) / field.std()       # mean 20 C, std 5 C. field[y, x]

# ---------------------------------------------------------------- agents: positions [x, y]
pos = np.zeros((N,2), dtype=float)

# ---------------------------------------------------------------- movement: YOU WRITE THIS
def random_move(t, pos):
    """Return the new (N, 2) positions of all agents at step t.
    (Placeholder: random walk, so the script runs. Replace with your own.)"""
    angle = rng.uniform(0, 2 * np.pi, N)
    return pos + np.column_stack([np.cos(angle), np.sin(angle)])

# -----------
# Flocking 1: spiral
# Each agent moves spirally towards the center of the grid, with some random noise.
def spiral_move(t, pos):
    """Return the new (N, 2) positions of all agents at step t."""
    center = np.array([G / 2, G / 2])
    direction = center - pos
    distance = np.linalg.norm(direction, axis=1, keepdims=True)
    direction /= distance + 1e-6  # normalize to unit vector
    angle_noise = rng.uniform(-np.pi/8, np.pi/8, N)  # small random angle noise
    rotation_matrix = np.array([[np.cos(angle_noise), -np.sin(angle_noise)],
                                 [np.sin(angle_noise), np.cos(angle_noise)]]).transpose((2, 0, 1))
    direction = np.einsum('ijk,ik->ij', rotation_matrix, direction)  # apply rotation
    step_size = 0.5 + rng.uniform(0, 0.5, N)  # random step size between 0.5 and 1.0
    return pos + direction * step_size



# ---------------------------------------------------------------- measurements
views = np.zeros((G, G))            # how many readings landed in each cell
readings = [[] for _ in range(G * G)]   # list of all readings for each cell (index = y * G + x

for t in range(1, STEPS + 1):
    pos = np.clip(random_move(t, pos), 0, G - 1e-6)                # keep agents inside the grid
    x, y = pos[:, 0].astype(int), pos[:, 1].astype(int)     # cell each agent is in
    y_meas = field[y, x] + rng.normal(0, NOISE, N)          # reading = true value + noise
    for k in range(N):
        views[y[k], x[k]] += 1
        readings[y[k] * G + x[k]].append(y_meas[k])

print(f"total readings: {int(views.sum())}")
print(f"views per cell: mean {views.mean():.1f}, min {int(views.min())}, max {int(views.max())}")
print(f"cells never viewed: {(views == 0).mean() * 100:.1f}%")