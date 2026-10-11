"""
Melt-pool thermal models for laser powder bed fusion (LPBF).

Implements
  * Rosenthal (1946) moving point source, closed form (used for verification)
  * Eagar & Tsai (1983) moving Gaussian surface source, numerical time integral

Both assume a semi-infinite body, constant thermophysical properties and
conduction-mode heat transfer (no latent heat, no fluid flow, no keyhole).

Coordinates are in the frame moving with the laser: xi is along the scan
direction (positive = ahead of the beam), y is transverse, z is depth.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Material:
    """Representative constant properties for Ti-6Al-4V (see report, Table 1)."""
    name: str = "Ti-6Al-4V"
    rho: float = 4420.0        # density, kg/m^3
    cp: float = 700.0          # specific heat, J/(kg K)  (temperature-averaged)
    k: float = 20.0            # thermal conductivity, W/(m K) (temperature-averaged)
    T_liq: float = 1928.0      # liquidus temperature, K
    T0: float = 300.0          # powder-bed / substrate temperature, K
    absorptivity: float = 0.40 # fraction of laser power absorbed (conduction mode)

    @property
    def alpha(self) -> float:  # thermal diffusivity, m^2/s
        return self.k / (self.rho * self.cp)


@dataclass(frozen=True)
class Laser:
    r0: float = 50e-6          # 1/e^2 beam radius, m (spot diameter 100 um)

    @property
    def sigma(self) -> float:  # Gaussian standard deviation, m (r0 = 2 sigma)
        return self.r0 / 2.0


# ---------------------------------------------------------------------------
# Rosenthal point source (closed form)
# ---------------------------------------------------------------------------
def rosenthal_T(xi, y, z, P, v, mat: Material):
    q = mat.absorptivity * P
    R = np.sqrt(xi**2 + y**2 + z**2)
    R = np.maximum(R, 1e-12)
    return mat.T0 + q / (2 * np.pi * mat.k * R) * np.exp(-v * (xi + R) / (2 * mat.alpha))


# ---------------------------------------------------------------------------
# Eagar-Tsai Gaussian source
# ---------------------------------------------------------------------------
# T - T0 = q / (pi rho c sqrt(4 pi a)) * Int_0^inf  t^-1/2 / (2 a t + s^2)
#          * exp( -[(xi + v t)^2 + y^2] / (4 a t + 2 s^2) - z^2 / (4 a t) ) dt
# The integral is evaluated with the trapezoid rule on a logarithmic grid in t
# (integrand * t, d ln t), which resolves both the fast early-time behaviour and
# the slow t^-3/2 tail.  Accuracy is checked against scipy.integrate.quad.
_LOG_T = np.linspace(np.log(1e-11), np.log(1.0), 1100)
_T_NODES = np.exp(_LOG_T)
_W_TRAP = np.full(_LOG_T.size, _LOG_T[1] - _LOG_T[0])
_W_TRAP[[0, -1]] *= 0.5


def et_theta(xi, y, z, v, mat: Material, laser: Laser, t_nodes=_T_NODES, w=_W_TRAP):
    """Temperature rise per watt of *incident* power, (T - T0)/P, in K/W.

    xi, y, z must be broadcastable arrays; the time axis is appended last.
    Because the model is linear in power, one field per speed serves all powers.
    """
    a, s2 = mat.alpha, laser.sigma**2
    pref = mat.absorptivity / (np.pi * mat.rho * mat.cp * np.sqrt(4 * np.pi * a))
    xi, y, z = (np.asarray(u, float)[..., None] for u in (xi, y, z))
    t = t_nodes
    denom = 4 * a * t + 2 * s2
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        expo = -((xi + v * t) ** 2 + y**2) / denom - z**2 / (4 * a * t)
        f = t**-0.5 / (2 * a * t + s2) * np.exp(expo) * t   # * t for d(ln t)
    f = np.nan_to_num(f)
    return pref * (f * w).sum(axis=-1)


def et_T(xi, y, z, P, v, mat: Material, laser: Laser):
    return mat.T0 + P * et_theta(xi, y, z, v, mat, laser)


# ---------------------------------------------------------------------------
# Melt-pool dimensions
# ---------------------------------------------------------------------------
def _upper_crossing(coord, prof, c):
    """Largest coord at which a (decreasing-in-tail) profile still exceeds c."""
    idx = np.nonzero(prof >= c)[0]
    if idx.size == 0:
        return 0.0
    i = idx[-1]
    if i == len(coord) - 1:
        return coord[-1]
    # linear interpolation between i (>= c) and i+1 (< c)
    f = (prof[i] - c) / (prof[i] - prof[i + 1])
    return coord[i] + f * (coord[i + 1] - coord[i])


def speed_fields(v, mat: Material, laser: Laser, xi_back=-1.4e-3, xi_front=0.25e-3,
                 dxi=5e-6, rmax=0.3e-3, dr=2e-6):
    """Pre-compute the normalised profiles needed to extract W, D, L at one speed.

    Returns a dict with
        xi, theta_line(xi)      : centre-line surface profile (y=z=0)
        r, theta_w(r)           : max over xi of theta(xi, y=r, z=0)
        r, theta_d(r)           : max over xi of theta(xi, y=0, z=r)
    """
    xi = np.arange(xi_back, xi_front + dxi / 2, dxi)
    r = np.arange(0.0, rmax + dr / 2, dr)
    line = et_theta(xi, 0.0, 0.0, v, mat, laser)
    # 2-D maps computed in chunks to bound memory
    tw = np.empty(r.size)
    td = np.empty(r.size)
    chunk = 20
    for j in range(0, r.size, chunk):
        rr = r[j:j + chunk]
        tw[j:j + chunk] = et_theta(xi[None, :], rr[:, None], 0.0, v, mat, laser).max(axis=1)
        td[j:j + chunk] = et_theta(xi[None, :], 0.0, rr[:, None], v, mat, laser).max(axis=1)
    return dict(xi=xi, line=line, r=r, tw=tw, td=td)


def dims_from_fields(fields, P, mat: Material):
    """Melt-pool width, depth, length (m) for incident power P (W)."""
    c = (mat.T_liq - mat.T0) / P
    W = 2 * _upper_crossing(fields["r"], fields["tw"], c)
    D = _upper_crossing(fields["r"], fields["td"], c)
    xi, line = fields["xi"], fields["line"]
    idx = np.nonzero(line >= c)[0]
    if idx.size == 0:
        L = 0.0
    else:
        # front and back crossings with linear interpolation
        i0, i1 = idx[0], idx[-1]
        back = xi[i0] if i0 == 0 else xi[i0 - 1] + (c - line[i0 - 1]) / (line[i0] - line[i0 - 1]) * (xi[i0] - xi[i0 - 1])
        front = xi[i1] if i1 == len(xi) - 1 else xi[i1] + (line[i1] - c) / (line[i1] - line[i1 + 1]) * (xi[i1 + 1] - xi[i1])
        L = front - back
    return W, D, L


def melt_pool_dims(P, v, mat: Material = Material(), laser: Laser = Laser()):
    return dims_from_fields(speed_fields(v, mat, laser), P, mat)
