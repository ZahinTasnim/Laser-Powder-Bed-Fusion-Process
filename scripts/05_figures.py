"""All figures used in the README.

Reads results/ground_truth.npz, results/validation.json and results/benchmark.pkl
(made by scripts 01-03) and writes six PNG files to figures/. Takes about a minute.
"""
import json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lpbf.model import Material, Laser, et_T
from lpbf.bo import VirtualLPBF
from lpbf.criteria import BOUNDS, T_LAYER, keyhole_number, KE_MAX, balling_margin, h_max_fused

# ------------------------------------------------ style
INK, INK2, GRID, GREY = "#222222", "#555555", "#e6e6e6", "#a9a9a9"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3 (validated)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#b8b8b8", "axes.linewidth": 0.8,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
os.makedirs("figures", exist_ok=True)


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"figures/{name}.png", dpi=200); plt.close(fig)


V = json.load(open("results/validation.json")); o = V["optimum"]
B = pickle.load(open("results/benchmark.pkl", "rb")); R = B["results"]; MCK = list(B["model_checkpoints"]); BUDGET = B["budget"]
arr = lambda noise, name, key: np.array([r[key] for r in R[(noise, name)]])
G, RS, BO, NA = "Grid DOE", "Random search", "BO (physics-structured)", "BO (physics-structured, noise-aware)"
lab = VirtualLPBF(); mat, las = Material(), Laser()

# ------------------------------------------------ the map of defect zones (used by figures 3 and 4)
P = np.linspace(*BOUNDS[0], 320); v = np.linspace(*BOUNDS[1], 320)
PP, VV = np.meshgrid(P, v, indexing="ij")
W, D, L = (a.reshape(PP.shape) for a in lab.true_dims(PP, VV))
region = np.zeros(PP.shape)
region[np.minimum(h_max_fused(W, D), BOUNDS[2, 1]) < BOUNDS[2, 0]] = 1
region[balling_margin(W, L) > 0] = 2
region[keyhole_number(PP, VV) > KE_MAX] = 3
ZONES = ListedColormap(["#dff0d8", "#f8d7d7", "#d6e6f7", "#fbefcf"])


def draw_map(ax, labels=True):
    ax.pcolormesh(VV, PP, region, cmap=ZONES, vmin=-.5, vmax=3.5, shading="auto", rasterized=True)
    if labels:
        for txt, (xv, yp) in {"Printable": (0.95, 168), "Too little energy:\ngaps between tracks\n(lack of fusion)": (1.52, 95),
                              "Too much energy:\ndeep pores (keyholing)": (0.55, 330), "Pool too long:\nbeads (balling)": (1.68, 262)}.items():
            ax.text(xv, yp, txt, ha="center", va="center", fontsize=8.2, color=INK)
    ax.set_xlim(0.2, 2.0); ax.set_ylim(60, 400)


# ------------------------------------------------ Figure 1: three neighbouring tracks at the best recipe
Wc, Dc, hc, t = o["W"] * 1e6, o["D"] * 1e6, o["h"] * 1e6, T_LAYER * 1e6
fig, ax = plt.subplots(figsize=(6.8, 2.9))
ax.fill_between([-165, 165], -85, -t, color="#e3e3e3"); ax.fill_between([-165, 165], -t, 0, color="#fbefcf")
for xi in (-hc, 0, hc):
    xs = np.linspace(xi - Wc / 2, xi + Wc / 2, 300); ys = -Dc * np.sqrt(np.clip(1 - ((xs - xi) / (Wc / 2)) ** 2, 0, None))
    ax.fill_between(xs, ys, 0, color=BLUE, alpha=0.30, lw=0); ax.plot(xs, ys, color=BLUE, lw=1.6)
ax.axhline(-t, color=INK2, ls="--", lw=0.9); ax.axhline(0, color=INK, lw=1.0)
ax.text(-163, 5, "new powder layer (thickness t)", fontsize=8, va="bottom")
ax.text(-163, -79, "earlier layers (already solid)", fontsize=8, color=INK2)


def dim(x0, y0, x1, y1, txt, tx, ty, ha="center"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK))
    ax.text(tx, ty, txt, fontsize=8.6, ha=ha, va="center", bbox=dict(fc="white", ec="none", pad=0.6))


dim(-Wc / 2, 15, Wc / 2, 15, f"W = {Wc:.0f} µm (pool width)", 0, 22)
dim(0, 31, hc, 31, f"h = {hc:.0f} µm (track spacing)", hc / 2, 38)
dim(0, 0, 0, -Dc, f"D = {Dc:.0f} µm\n(pool depth)", -4, -Dc / 2 - 4, ha="right")
dim(-Wc / 2 - hc - 12, 0, -Wc / 2 - hc - 12, -t, "t", -Wc / 2 - hc - 20, -t / 2)
ax.annotate("the dip between two pools just reaches\nthe bottom of the new layer: the limit\nof the lack-of-fusion rule",
            xy=(hc / 2, -t), xytext=(hc + 36, -58), fontsize=7.8, arrowprops=dict(arrowstyle="->", lw=0.8, color=INK2), va="center")
ax.set_xlim(-170, 175); ax.set_ylim(-88, 48); ax.axis("off")
save(fig, "fig1_tracks")

# ------------------------------------------------ Figure 2: what the heat model computes
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
Y = np.linspace(-0.16e-3, 0.16e-3, 161); Z = np.linspace(0, 0.09e-3, 61); YY, ZZ = np.meshgrid(Y, Z)
Tmax = np.max([et_T(x, YY, ZZ, 250, 0.8, mat, las) for x in np.linspace(-0.4e-3, 0.1e-3, 61)], axis=0)
ax[0].contourf(YY * 1e6, -ZZ * 1e6, Tmax, levels=[0, 800, 1300, mat.T_liq, 1e7], colors=["#fff3e6", "#fdd9b4", "#f9b27a", ORANGE])
ax[0].contour(YY * 1e6, -ZZ * 1e6, Tmax, levels=[mat.T_liq], colors=INK, linewidths=1.4)
ax[0].text(0, -22, "liquid metal\n(above 1,928 K)", ha="center", va="center", fontsize=8, color="white", weight="bold")
ax[0].text(118, -80, "solid, heated", ha="center", va="center", fontsize=7.6, color=INK2)
ax[0].set_aspect("equal"); ax[0].set_xlabel("Distance across the track (µm)"); ax[0].set_ylabel("Depth (µm)")
ax[0].set_title("Melt pool cut across one track\n(250 W, 0.8 m/s)", fontsize=9, loc="left")
gt = np.load("results/ground_truth.npz"); k = int(np.argmin(np.abs(gt["v"] - 1.0)))
for key, col, name in (("L", GREEN, "length L"), ("W", BLUE, "width W"), ("D", ORANGE, "depth D")):
    vals = gt[key][:, k] * 1e6
    ax[1].plot(gt["P"], vals, color=col, lw=2, label=name)
    ax[1].text(gt["P"][-1] + 6, vals[-1], name, va="center", fontsize=8.2)
ax[1].set_xlim(60, 470); ax[1].set_xticks([100, 200, 300, 400]); ax[1].set_xlabel("Laser power (W)"); ax[1].set_ylabel("Melt pool size (µm)")
ax[1].set_title("Melt pool size at 1.0 m/s", fontsize=9, loc="left"); tidy(ax[1])
save(fig, "fig2_heat_model")

# ------------------------------------------------ Figure 3: map of safe settings and the best recipe
fig, ax = plt.subplots(figsize=(6.4, 3.6))
draw_map(ax)
ax.plot(o["v"], o["P"], "*", color=ORANGE, ms=16, mec=INK, mew=0.7)
ax.annotate(f"best recipe\n{o['P']:.0f} W, {o['v']:.2f} m/s", (o["v"], o["P"]), (1.08, 232), fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("Scan speed (m/s)"); ax.set_ylabel("Laser power (W)")
save(fig, "fig3_map")

# ------------------------------------------------ Figure 4: where each method puts its 40 test prints (one run, 3% error)
run = 3
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
for a, name, ttl in zip(ax, [G, RS, BO], ["Grid", "Random search", "Bayesian optimization"]):
    X = R[(0.03, name)][run]["X"]
    draw_map(a, labels=False)
    a.scatter(X[:, 1], X[:, 0], s=16, color=INK, edgecolors="white", linewidths=0.5, zorder=3)
    a.plot(o["v"], o["P"], "*", color=ORANGE, ms=13, mec=INK, mew=0.6, zorder=4)
    a.set_xlim(0.14, 2.06); a.set_ylim(48, 412)
    a.set_title(ttl, fontsize=9, loc="left"); a.set_xlabel("Scan speed (m/s)")
ax[0].set_ylabel("Laser power (W)")
save(fig, "fig4_where_tested")

# ------------------------------------------------ Figure 5: search progress with exact measurements
n = np.arange(1, BUDGET + 1)
fig, ax = plt.subplots(figsize=(6.6, 3.0))
for name, col, lab_ in ((BO, BLUE, "Bayesian optimization"), (RS, ORANGE, "Random search"), (G, GREY, "Grid")):
    m = 100 * np.median(arr(0.0, name, "naive_val"), axis=0)
    ax.plot(n, m, color=col, lw=2, label=lab_)
    ax.text(40.6, m[-1] + {BO: 0, RS: 2.5, G: -2.5}[name], f"{lab_}: {m[-1]:.0f}%", va="center", fontsize=8.2)
ax.axhline(90, color=INK2, ls=":", lw=1.1); ax.text(39.5, 86.5, "90% of the best possible speed", fontsize=8, color=INK2, ha="right")
ax.set_xlabel("Number of test prints used"); ax.set_ylabel("Speed of the best safe recipe\nfound so far (% of best possible)")
ax.set_xlim(1, 54); ax.set_xticks([1, 10, 20, 30, 40]); ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=8.2, loc="lower right", bbox_to_anchor=(0.74, 0.04)); tidy(ax)
save(fig, "fig5_exact_measurements")

# ------------------------------------------------ Figure 6: with a 3% measurement error
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
ax[0].plot(n, 100 * (1 - arr(0.03, BO, "naive_ok").mean(0)), color=ORANGE, lw=2, label="Bayesian optimization: best-looking recipe")
ax[0].plot(n, 100 * (1 - arr(0.03, RS, "naive_ok").mean(0)), color=GREEN, lw=1.8, ls="--", label="Random search: best-looking recipe")
ax[0].plot(MCK, [100 * (1 - arr(0.03, NA, "model_ok")[:, MCK.index(c)].mean()) for c in MCK], "o-", color=BLUE, lw=2, ms=4,
           label="Cautious recipe (model 95% sure)")
ax[0].set_xlabel("Number of test prints used"); ax[0].set_ylabel("Runs where the chosen recipe\nis really defective (%)")
ax[0].set_xlim(1, 40); ax[0].set_ylim(-3, 105); ax[0].set_title("How often the chosen recipe is wrong", fontsize=9, loc="left")
ax[0].legend(frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.24)); tidy(ax[0])
labs = ["Grid", "Random search", "Bayesian optimization", "Bayesian optimization,\ncautious version"]
sc = [100 * np.median(arr(0.03, nm, "model_val")[:, MCK.index(BUDGET)]) for nm in (G, RS, BO, NA)]
y = np.arange(4)[::-1]
ax[1].barh(y, sc, color=[GREY, GREY, GREY, BLUE], height=0.6, edgecolor="white", lw=1)
for yi, vv in zip(y, sc): ax[1].text(vv + 1.5, yi, f"{vv:.0f}%", va="center", fontsize=8.5)
ax[1].set_yticks(y); ax[1].set_yticklabels(labs, fontsize=8.2); ax[1].set_xlim(0, 112); ax[1].set_xticks([0, 25, 50, 75, 100])
ax[1].axvline(100, color=INK2, ls=":", lw=1.1)
ax[1].set_xlabel("Speed after 40 prints\n(% of best possible)")
ax[1].set_title("Cautious recipes: all defect-free,\nbut slower", fontsize=9, loc="left"); tidy(ax[1], "x")
save(fig, "fig6_measurement_error")
print("figures written to figures/:", sorted(os.listdir("figures")))
