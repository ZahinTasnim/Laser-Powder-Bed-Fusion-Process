"""Benchmark: strategies x N_SEEDS repeats x noise levels, budget of 40 builds.

Resumable: combinations already present in results/benchmark.pkl are skipped,
and missing recommendation traces are added.  Seeds are fixed so results are
identical whether run in one go or resumed.

Recommendation rules recorded for every run:
  naive   : best build whose measured melt pool looked defect-free
  robust  : best *built* setting with >= 95 % posterior probability of being
            defect-free (physics-structured GP fitted to that run's data)
  model   : best setting *anywhere* in the process space with >= 95 % posterior
            probability of being defect-free (would be built once to confirm)
"""
import os
import time
import pickle
import numpy as np
from lpbf.bo import VirtualLPBF, STRATEGIES, _trace, robust_trace, model_trace

BUDGET = 40
N_SEEDS = 30
CHECKPOINTS = list(range(8, BUDGET + 1, 4))
MODEL_CHECKPOINTS = [8, 12, 16, 20, 28, 40]
PLAN = {0.0: ["Grid DOE", "Random search", "BO (black-box)", "BO (physics-structured)"],
        0.03: list(STRATEGIES)}          # noise-aware variants only matter with noise
PATH = "results/benchmark.pkl"

if os.path.exists(PATH):
    B = pickle.load(open(PATH, "rb"))
else:
    B = dict(results={}, budget=BUDGET, n_seeds=N_SEEDS)
B.update(checkpoints=CHECKPOINTS, model_checkpoints=MODEL_CHECKPOINTS,
         noises=list(PLAN), plan=PLAN)
R = B["results"]


def save():
    pickle.dump(B, open(PATH + ".tmp", "wb")); os.replace(PATH + ".tmp", PATH)


t0 = time.time()
for noise, names in PLAN.items():
    lab = VirtualLPBF(noise=noise)
    opt = lab.true_optimum()["br"]
    for name in names:
        if (noise, name) not in R:
            runs = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(1000 * seed + 7)
                X, WDL = STRATEGIES[name](lab, BUDGET, rng)
                nv, nok = _trace(lab, X, WDL)
                rv, rok = robust_trace(lab, X, WDL, CHECKPOINTS, np.random.default_rng(seed))
                runs.append(dict(X=X, WDL=WDL, naive_val=nv / opt, naive_ok=nok,
                                 robust_val=rv / opt, robust_ok=rok))
            R[(noise, name)] = runs; save()
            print(f"noise={noise:.2f} {name:38s} runs done ({time.time()-t0:.0f}s)", flush=True)
        if "model_val" not in R[(noise, name)][0]:
            for seed, r in enumerate(R[(noise, name)]):
                mv, mok = model_trace(lab, r["X"], r["WDL"], MODEL_CHECKPOINTS,
                                      np.random.default_rng(seed + 500))
                r["model_val"], r["model_ok"] = mv / opt, mok
            save()
            print(f"noise={noise:.2f} {name:38s} model trace done ({time.time()-t0:.0f}s)", flush=True)
print("saved", PATH)
