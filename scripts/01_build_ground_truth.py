"""Compute Eagar-Tsai melt-pool dimensions on a dense (P, v) grid.

The table is the 'ground truth' process physics.  A smooth interpolant of it
serves as the fast virtual experiment inside the optimisation benchmark
(its interpolation error is checked in 02_validate_and_map.py).
"""
import time
import numpy as np
from lpbf.model import Material, Laser, speed_fields, dims_from_fields
from lpbf.criteria import BOUNDS

mat, las = Material(), Laser()
P_grid = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], 171)      # 2 W steps
v_grid = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], 73)       # 0.025 m/s steps

W = np.zeros((P_grid.size, v_grid.size)); D = np.zeros_like(W); L = np.zeros_like(W)
t0 = time.time()
for j, v in enumerate(v_grid):
    F = speed_fields(v, mat, las)
    for i, P in enumerate(P_grid):
        W[i, j], D[i, j], L[i, j] = dims_from_fields(F, P, mat)
    print(f"v={v:.3f}  done {j+1}/{v_grid.size}  ({time.time()-t0:.0f}s)", flush=True)

np.savez("results/ground_truth.npz", P=P_grid, v=v_grid, W=W, D=D, L=L)
print("saved results/ground_truth.npz")
