"""Checks of the heat model, accuracy of the lookup table, and the best recipe (writes results/validation.json)."""
import json
import numpy as np
from scipy.integrate import quad

from lpbf.model import Material, Laser, et_theta, et_T, rosenthal_T, melt_pool_dims
from lpbf.criteria import (BOUNDS, keyhole_number, KE_MAX, lof_margin, balling_margin,
                           h_max_fused, build_rate, T_LAYER, LW_MAX)
from lpbf.bo import VirtualLPBF

mat, las = Material(), Laser()
out = {}

# ---- (1) quadrature accuracy vs adaptive scipy.quad --------------------------
a, s2 = mat.alpha, las.sigma**2
pref = mat.absorptivity / (np.pi * mat.rho * mat.cp * np.sqrt(4 * np.pi * a))


def theta_quad(xi, y, z, v):
    f = lambda t: t**-0.5 / (2*a*t + s2) * np.exp(-((xi + v*t)**2 + y**2)/(4*a*t + 2*s2) - z**2/(4*a*t))
    edges = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, np.inf]
    return pref * sum(quad(f, lo, hi, limit=200)[0] for lo, hi in zip(edges[:-1], edges[1:]))


rng = np.random.default_rng(1)
errs = []
# hand-picked hot spots near/under the beam, where the integrand is most peaked
for (xi, y, z, v) in [(0, 0, 0, 1.0), (50e-6, 0, 0, 2.0), (-20e-6, 10e-6, 5e-6, 0.2),
                      (0, 30e-6, 0, 1.5), (-100e-6, 0, 20e-6, 0.8)]:
    q = theta_quad(xi, y, z, v)
    errs.append(abs(et_theta(xi, y, z, v, mat, las) - q) / q)
for _ in range(60):
    xi = rng.uniform(-1.2e-3, 0.2e-3); y = rng.uniform(0, 150e-6); z = rng.uniform(0, 100e-6)
    v = rng.uniform(0.2, 2.0)
    q = theta_quad(xi, y, z, v)
    errs.append(abs(et_theta(xi, y, z, v, mat, las) - q) / q)
out["quad_rel_err_max"] = float(np.max(errs)); out["quad_rel_err_median"] = float(np.median(errs))

# ---- (2) Rosenthal limit (sigma -> 0) --------------------------------------
xi = np.linspace(-1.0e-3, 0.15e-3, 300)
T_r = rosenthal_T(xi, 0, 30e-6, 200, 0.8, mat)
T_small = et_T(xi, 0, 30e-6, 200, 0.8, mat, Laser(r0=0.2e-6))
out["rosenthal_limit_rel_err_max"] = float(np.max(np.abs(T_small - T_r) / T_r))

# ---- (3) emulator (cubic interpolation) accuracy vs direct model ------------
lab = VirtualLPBF()
emu_err = []
for _ in range(25):
    P = rng.uniform(*BOUNDS[0]); v = rng.uniform(*BOUNDS[1])
    d = np.array(melt_pool_dims(P, v, mat, las))
    e = np.array([x[0] for x in lab.true_dims(P, v)])
    emu_err.append(np.abs(e - d) / d)
emu_err = np.array(emu_err)
out["emulator_rel_err_max"] = emu_err.max(axis=0).tolist()      # W, D, L
out["emulator_rel_err_mean"] = emu_err.mean(axis=0).tolist()

# ---- (4) share of settings that are printable + true optimum ------------------------------------
opt = lab.true_optimum()
out["optimum"] = {k: float(v) for k, v in opt.items()}
P = np.linspace(*BOUNDS[0], 300); v = np.linspace(*BOUNDS[1], 300)
PP, VV = np.meshgrid(P, v, indexing="ij")
W, D, L = (x.reshape(PP.shape) for x in lab.true_dims(PP, VV))
kh = keyhole_number(PP, VV) > KE_MAX
ball = balling_margin(W, L) > 0
hmax = np.minimum(h_max_fused(W, D), BOUNDS[2, 1])
lof = hmax < BOUNDS[2, 0]                          # no admissible hatch
region = np.zeros(PP.shape)                        # 0 ok, 1 LOF, 2 balling, 3 keyhole
region[lof] = 1; region[ball] = 2; region[kh] = 3
out["fraction_box_printable"] = float(np.mean(region == 0))

json.dump(out, open("results/validation.json", "w"), indent=2)
print(json.dumps(out, indent=2))
