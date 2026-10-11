"""
Virtual experiment + optimisation strategies.

Strategies compared (all share the same budget and the same known keyhole
constraint, which is screened analytically before any 'build'):
    grid       : full-factorial design of experiments (DOE)
    random     : uniform random search
    bo_black   : constrained BO, black-box GP surrogates of each constraint
                 margin over (P, v, h)
    bo_grey    : constrained BO, physics-structured ("grey-box") surrogates:
                 GPs of log W, log D, log L over (P, v) only, composed with the
                 known analytical defect criteria (h enters analytically)
"""
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

from .criteria import (BOUNDS, T_LAYER, keyhole_number, KE_MAX, lof_margin,
                       balling_margin, build_rate, h_max_fused)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

LO, HI = BOUNDS[:, 0], BOUNDS[:, 1]


def to_phys(u):
    return LO + np.asarray(u) * (HI - LO)


def to_unit(x):
    return (np.asarray(x) - LO) / (HI - LO)


class VirtualLPBF:
    """Fast emulator of the Eagar-Tsai model + multiplicative measurement noise."""

    def __init__(self, gt_path="results/ground_truth.npz", noise=0.03):
        g = np.load(gt_path)
        self.P, self.v = g["P"], g["v"]
        self._interp = {k: RegularGridInterpolator((self.P, self.v), np.log(g[k]),
                                                   method="cubic")
                        for k in ("W", "D", "L")}
        self.noise = noise

    def true_dims(self, P, v):
        pts = np.column_stack([np.ravel(P), np.ravel(v)])
        return tuple(np.exp(self._interp[k](pts)) for k in ("W", "D", "L"))

    def measure(self, P, v, rng):
        W, D, L = self.true_dims(P, v)
        e = rng.normal(0.0, self.noise, size=(3, W.size))
        return W * np.exp(e[0]), D * np.exp(e[1]), L * np.exp(e[2])

    # --------------------------------------------------------------- truth
    def true_value(self, x):
        """True build rate if x is truly defect-free, else 0; plus feasibility flag."""
        x = np.atleast_2d(x)
        P, v, h = x.T
        W, D, L = self.true_dims(P, v)
        ok = ((lof_margin(W, D, h) <= 0) & (balling_margin(W, L) <= 0)
              & (keyhole_number(P, v) <= KE_MAX))
        return np.where(ok, build_rate(v, h), 0.0), ok

    def true_optimum(self, nP=500, nv=500):
        P = np.linspace(LO[0], HI[0], nP); v = np.linspace(LO[1], HI[1], nv)
        PP, VV = np.meshgrid(P, v, indexing="ij")
        W, D, L = (a.reshape(PP.shape) for a in self.true_dims(PP, VV))
        h = np.minimum(h_max_fused(W, D), HI[2])
        ok = ((h >= LO[2]) & (balling_margin(W, L) <= 0)
              & (keyhole_number(PP, VV) <= KE_MAX))
        br = np.where(ok, build_rate(VV, h), 0.0)
        i = np.unravel_index(np.argmax(br), br.shape)
        return dict(P=PP[i], v=VV[i], h=h[i], br=br[i], W=W[i], D=D[i], L=L[i])


# ---------------------------------------------------------------------------
def _gp(dim):
    k = (C(1.0, (1e-3, 1e3)) *
         Matern(length_scale=np.full(dim, 0.4), length_scale_bounds=(2e-2, 1e1), nu=2.5)
         + WhiteKernel(1e-3, (1e-6, 1e-1)))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2,
                                    random_state=0)


def _sample_known_feasible(n, rng, sobol=False):
    out = []
    eng = qmc.Sobol(3, scramble=True, seed=rng) if sobol else None
    while sum(len(o) for o in out) < n:
        u = eng.random(64) if sobol else rng.random((max(4 * n, 64), 3))
        x = to_phys(u)
        out.append(u[keyhole_number(x[:, 0], x[:, 1]) <= KE_MAX])
    return np.vstack(out)[:n]


def _observe(lab, X, rng):
    W, D, L = lab.measure(X[:, 0], X[:, 1], rng)
    return np.column_stack([W, D, L])


def _observed_feasible(X, WDL):
    return ((lof_margin(WDL[:, 0], WDL[:, 1], X[:, 2]) <= 0) &
            (balling_margin(WDL[:, 0], WDL[:, 2]) <= 0))


def _trace(lab, X, WDL):
    """For each n: the recommended (best observed-feasible) point's true value."""
    feas = _observed_feasible(X, WDL)
    obj = build_rate(X[:, 1], X[:, 2])
    tv, tok = lab.true_value(X)
    best_val, best_true, best_ok = -np.inf, 0.0, True
    rec_true, rec_ok = [], []
    for i in range(len(X)):
        if feas[i] and obj[i] > best_val:
            best_val, best_true, best_ok = obj[i], tv[i], bool(tok[i])
        rec_true.append(best_true if best_val > -np.inf else 0.0)
        rec_ok.append(best_ok)
    return np.array(rec_true), np.array(rec_ok)


def recommend_robust(X, WDL, n, rng, conf=0.95):
    """Robust recommendation from the first n builds: the highest-build-rate
    evaluated point whose posterior probability of being defect-free (physics-
    structured GP fitted to those builds) is >= conf.  Returns index or None."""
    Xn, Dn = X[:n], WDL[:n]
    U = to_unit(Xn)
    models = [_gp(2).fit(U[:, :2], np.log(Dn[:, k])) for k in range(3)]
    pof = _pof_grey(models, U, rng, n_samp=400)
    ok = pof >= conf
    if not ok.any():
        return None
    obj = build_rate(Xn[:, 1], Xn[:, 2])
    idx = np.nonzero(ok)[0]
    return idx[np.argmax(obj[idx])]


def recommend_model(X, WDL, n, rng, conf=0.95, n_cand=6000):
    """Model-based robust recommendation: the process setting anywhere in the
    (keyhole-screened) space with the highest build rate among those whose
    posterior probability of being defect-free is >= conf.  Uses the
    physics-structured GP fitted to the first n builds.  In practice the
    recommended setting would then be built once to confirm it."""
    Xn, Dn = X[:n], WDL[:n]
    U = to_unit(Xn)
    models = [_gp(2).fit(U[:, :2], np.log(Dn[:, k])) for k in range(3)]
    obj = build_rate(Xn[:, 1], Xn[:, 2])
    Uc = np.vstack([_sample_known_feasible(n_cand, rng), U])
    top = U[np.argsort(obj)[-5:]]                         # local search near good builds
    loc = np.clip(np.repeat(top, 400, 0) + rng.normal(0, 0.03, (2000, 3)), 0, 1)
    xl = to_phys(loc)
    Uc = np.vstack([Uc, loc[keyhole_number(xl[:, 0], xl[:, 1]) <= KE_MAX]])
    pof = _pof_grey(models, Uc, rng, n_samp=200)
    ok = pof >= conf
    if not ok.any():
        return None
    fc = build_rate(*to_phys(Uc)[:, 1:].T)
    idx = np.nonzero(ok)[0]
    return to_phys(Uc[idx[np.argmax(fc[idx])]][None])[0]


def model_trace(lab, X, WDL, checkpoints, rng):
    vals, oks = [], []
    for n in checkpoints:
        x = recommend_model(X, WDL, n, rng)
        if x is None:
            vals.append(0.0); oks.append(True)
        else:
            tv, tok = lab.true_value(x[None])
            vals.append(float(tv[0])); oks.append(bool(tok[0]))
    return np.array(vals), np.array(oks)


def robust_trace(lab, X, WDL, checkpoints, rng):
    tv, tok = lab.true_value(X)
    vals, oks = [], []
    for n in checkpoints:
        i = recommend_robust(X, WDL, n, rng)
        vals.append(0.0 if i is None else tv[i]); oks.append(True if i is None else bool(tok[i]))
    return np.array(vals), np.array(oks)


# ----------------------------------------------------------------- baselines
def grid_design(budget):
    """Largest full-factorial design whose keyhole-screened size fits the budget."""
    best = None
    for nP in range(3, 9):
        for nv in range(3, 9):
            for nh in range(3, 6):
                g = np.array(np.meshgrid(np.linspace(0, 1, nP), np.linspace(0, 1, nv),
                                         np.linspace(0, 1, nh), indexing="ij")).reshape(3, -1).T
                x = to_phys(g)
                g = g[keyhole_number(x[:, 0], x[:, 1]) <= KE_MAX]
                bal = -np.std([nP, nv, nh])          # prefer balanced designs on ties
                if len(g) <= budget and (best is None or (len(g), bal) > (len(best[0]), best[2])):
                    best = (g, (nP, nv, nh), bal)
    return best


def run_grid(lab, budget, rng):
    g = grid_design(budget)[0]
    order = rng.permutation(len(g))          # build order is irrelevant; randomise
    X = to_phys(g[order])
    WDL = _observe(lab, X, rng)
    return X, WDL


def run_random(lab, budget, rng):
    X = to_phys(_sample_known_feasible(budget, rng))
    return X, _observe(lab, X, rng)


# ------------------------------------------------------------ constrained BO
def _candidates(rng, X_unit, best_u, n_global=3000, n_local=800):
    cand = [_sample_known_feasible(n_global, rng)]
    if best_u is not None:
        loc = np.clip(best_u + rng.normal(0, 0.05, (n_local, 3)), 0, 1)
        xl = to_phys(loc)
        cand.append(loc[keyhole_number(xl[:, 0], xl[:, 1]) <= KE_MAX])
    return np.vstack(cand)


def _pof_black(models, U):
    p = np.ones(len(U))
    for m in models:
        mu, sd = m.predict(U, return_std=True)
        p *= norm.cdf(-mu / np.maximum(sd, 1e-9))
    return p


def _pof_grey(models, U, rng, n_samp=64):
    X = to_phys(U)
    draws = []
    for m in models:                          # log W, log D, log L over (P, v)
        mu, sd = m.predict(U[:, :2], return_std=True)
        draws.append(np.exp(mu[None] + sd[None] * rng.standard_normal((n_samp, len(U)))))
    W, D, L = draws
    ok = (lof_margin(W, D, X[None, :, 2]) <= 0) & (balling_margin(W, L) <= 0)
    return ok.mean(axis=0)


def run_bo(lab, budget, rng, mode="grey", n_init=8, noise_aware=False, conf=0.95):
    U = _sample_known_feasible(n_init, rng, sobol=True)
    X = to_phys(U)
    WDL = _observe(lab, X, rng)
    while len(X) < budget:
        obj = build_rate(X[:, 1], X[:, 2])
        if mode == "grey":
            models = [_gp(2).fit(U[:, :2], np.log(WDL[:, k])) for k in range(3)]
        else:
            y1 = lof_margin(WDL[:, 0], WDL[:, 1], X[:, 2])
            y2 = balling_margin(WDL[:, 0], WDL[:, 2])
            models = [_gp(3).fit(U, y1), _gp(3).fit(U, y2)]

        if noise_aware:
            # incumbent = best evaluated point the surrogate is confident is defect-free
            pe = _pof_grey(models, U, rng, n_samp=256) if mode == "grey" else _pof_black(models, U)
            feas = pe >= conf
        else:
            # incumbent = best evaluated point whose (noisy) measurement looked defect-free
            feas = _observed_feasible(X, WDL)
        f_best = obj[feas].max() if feas.any() else None
        best_u = U[feas][np.argmax(obj[feas])] if feas.any() else None

        Uc = _candidates(rng, U, best_u)
        pof = _pof_grey(models, Uc, rng) if mode == "grey" else _pof_black(models, Uc)
        fc = build_rate(*to_phys(Uc)[:, 1:].T)
        if f_best is None:
            acq = pof                                   # first find any feasible point
        else:
            acq = np.maximum(fc - f_best, 0.0) * pof    # expected feasible improvement
            if acq.max() <= 0:
                acq = fc * pof
        u_next = Uc[np.argmax(acq)]
        U = np.vstack([U, u_next])
        x_next = to_phys(u_next[None])
        X = np.vstack([X, x_next])
        WDL = np.vstack([WDL, _observe(lab, x_next, rng)])
    return X, WDL


STRATEGIES = {
    "Grid DOE": run_grid,
    "Random search": run_random,
    "BO (black-box)": lambda lab, b, rng: run_bo(lab, b, rng, mode="black"),
    "BO (physics-structured)": lambda lab, b, rng: run_bo(lab, b, rng, mode="grey"),
    "BO (black-box, noise-aware)": lambda lab, b, rng: run_bo(lab, b, rng, mode="black", noise_aware=True),
    "BO (physics-structured, noise-aware)": lambda lab, b, rng: run_bo(lab, b, rng, mode="grey", noise_aware=True),
}
