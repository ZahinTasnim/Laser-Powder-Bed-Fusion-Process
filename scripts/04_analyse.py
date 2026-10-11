"""Summary statistics and significance tests from results/benchmark.pkl (writes results/summary.json)."""
import json
import pickle
import numpy as np
from scipy.stats import mannwhitneyu


B = pickle.load(open("results/benchmark.pkl", "rb"))
R, CK, MCK, BUDGET = B["results"], np.array(B["checkpoints"]), np.array(B["model_checkpoints"]), B["budget"]
BASE = ["Grid DOE", "Random search", "BO (black-box)", "BO (physics-structured)"]
NA = ["BO (black-box, noise-aware)", "BO (physics-structured, noise-aware)"]
ALL = BASE + NA
S = {}


def arr(noise, name, key):
    return np.array([r[key] for r in R[(noise, name)]])


def at(noise, name, rule, n):
    if rule == "naive":
        return arr(noise, name, "naive_val")[:, n - 1], arr(noise, name, "naive_ok")[:, n - 1]
    ck = CK if rule == "robust" else MCK
    k = list(ck).index(n)
    return arr(noise, name, rule + "_val")[:, k], arr(noise, name, rule + "_ok")[:, k]


for noise, names in B["plan"].items():
    for name in names:
        for rule in ("naive", "robust", "model"):
            for n in (12, 20, BUDGET):
                if rule == "robust" and n not in CK: continue
                v, ok = at(noise, name, rule, n)
                S[f"{noise}|{name}|{rule}|{n}"] = dict(
                    median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                    q75=float(np.percentile(v, 75)), defective=float(np.mean(~ok)))
        # builds to reach thresholds
        for rule, ck, thr in (("naive", np.arange(1, BUDGET + 1), 0.90), ("model", MCK, 0.80)):
            M = arr(noise, name, rule + "_val")
            reach = [ck[np.nonzero(m >= thr)[0][0]] if (m >= thr).any() else np.inf for m in M]
            reach = np.array(reach, float)
            S[f"{noise}|{name}|reach_{rule}_{thr}"] = dict(
                frac=float(np.isfinite(reach).mean()),
                median=None if not np.isfinite(np.median(reach)) else float(np.median(reach)))

tests = {}
for n in (12, 20, BUDGET):
    for a, b in [("BO (physics-structured, noise-aware)", "Random search"),
                 ("BO (black-box, noise-aware)", "Random search"),
                 ("BO (physics-structured, noise-aware)", "BO (black-box, noise-aware)"),
                 ("BO (physics-structured, noise-aware)", "Grid DOE")]:
        x, _ = at(0.03, a, "model", n); y, _ = at(0.03, b, "model", n)
        tests[f"model|{n}|{a} > {b}"] = float(mannwhitneyu(x, y, alternative="greater").pvalue)
    for a, b in [("BO (physics-structured)", "Random search"), ("BO (physics-structured)", "BO (black-box)")]:
        x, _ = at(0.0, a, "naive", n); y, _ = at(0.0, b, "naive", n)
        tests[f"noisefree_naive|{n}|{a} > {b}"] = float(mannwhitneyu(x, y, alternative="greater").pvalue)
S["tests"] = tests
json.dump(S, open("results/summary.json", "w"), indent=1)


print(json.dumps(tests, indent=1))
