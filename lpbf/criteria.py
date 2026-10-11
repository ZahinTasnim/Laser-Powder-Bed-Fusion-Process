"""
Defect criteria and objective for LPBF process-window optimisation.

Design variables x = (P [W], v [m/s], h [m]); layer thickness t is fixed.

Known (analytical) constraint - evaluated before any build, applied equally
to every strategy:
    Keyholing: normalised keyhole number Ke (Gan et al. 2021) <= KE_MAX

Unknown constraints - require the melt-pool geometry (W, D, L), which in
practice comes from printing and cross-sectioning single tracks / coupons:
    Lack of fusion  : (h/W)^2 + (t/D)^2 <= 1        (Tang, Pistorius & Beuth 2017)
    Balling         : L/W <= LW_MAX                 (Plateau-Rayleigh type limit)

Objective (maximise): volumetric build rate  BR = v * h * t   [mm^3/s]
"""
import numpy as np
from .model import Material, Laser

T_LAYER = 30e-6          # powder layer thickness, m
KE_MAX = 6.0             # conduction -> keyhole transition (Gan et al. 2021)
LW_MAX = np.pi / np.sqrt(2.0 / 3.0)   # = 3.85, balling when pi*W/L < sqrt(2/3)

BOUNDS = np.array([[60.0, 400.0],      # P, W
                   [0.2, 2.0],         # v, m/s
                   [40e-6, 160e-6]])   # h, m


def keyhole_number(P, v, mat: Material = Material(), laser: Laser = Laser()):
    return (mat.absorptivity * P /
            ((mat.T_liq - mat.T0) * np.pi * mat.rho * mat.cp *
             np.sqrt(mat.alpha * v * laser.r0**3)))


def known_feasible(P, v):
    return keyhole_number(P, v) <= KE_MAX


def lof_margin(W, D, h, t=T_LAYER):
    """log of the Tang overlap sum; <= 0 means fully fused."""
    W = np.maximum(W, 1e-9); D = np.maximum(D, 1e-9)
    return np.log((h / W) ** 2 + (t / D) ** 2)


def balling_margin(W, L):
    """log(L/W / LW_MAX); <= 0 means stable track."""
    W = np.maximum(W, 1e-9)
    return np.log(np.maximum(L, 1e-9) / W / LW_MAX)


def build_rate(v, h, t=T_LAYER):
    """mm^3/s"""
    return v * h * t * 1e9


def h_max_fused(W, D, t=T_LAYER):
    """Largest hatch spacing that satisfies the Tang criterion (0 if none)."""
    arg = 1.0 - (t / np.maximum(D, 1e-12)) ** 2
    return np.where(arg > 0, W * np.sqrt(np.clip(arg, 0, None)), 0.0)
