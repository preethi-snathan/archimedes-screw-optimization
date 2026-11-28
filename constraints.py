"""
constraints.py -- five-variable design problem (Ro, rho, P, alpha) for a fixed N.

Decision vector  x = [Ro (m), rho, P (m), alpha (deg)]   (N is fixed per sub-problem)
Objective        maximise VT = nu(N, rho, lam) * pi * Ro^2 * P      (water per turn, m^3)
    with lam = P * tan(alpha) / (2*pi*Ro)

All constraints are written as  g(x) >= 0  (SciPy 'ineq' convention).
"""
import numpy as np
from nu import nu

# ---- fixed parameters (from the proposal) ---------------------------------
H = 3.0          # lift height [m]
Q_REQ = 0.10     # required flow [m^3/s]
L_MAX = 5.0      # horizontal footprint limit [m]
LAM_MIN = 0.05   # practical minimum pitch  (constraint 3)
LAM_MAX = 1.0    # buckets must form        (constraint 2)

# ---- simple bounds ---------------------------------------------------------
# The 6 m pitch cap is an additional design assumption, not implied by the
# normalized-pitch constraints. review_checks.py tests relaxing this cap.
BOUNDS = [(0.2, 1.0),     # Ro  [m]
          (0.2, 0.7),     # rho
          (0.05, 6.0),    # P   [m]
          (20.0, 40.0)]   # alpha [deg]
X_NAMES = ["Ro", "rho", "P", "alpha_deg"]


def lam_of(x):
    Ro, rho, P, a = x
    return P * np.tan(np.radians(a)) / (2 * np.pi * Ro)


def rpm_of(Ro):
    """Nagel limit: rpm = 50 / D^(2/3), D = 2*Ro."""
    return 50.0 / (2.0 * Ro) ** (2.0 / 3.0)


def volume_per_turn(x, N):
    Ro, rho, P, a = x
    return nu(N, rho, lam_of(x)) * np.pi * Ro**2 * P


def flow_rate(x, N):
    return volume_per_turn(x, N) * rpm_of(x[0]) / 60.0


def objective(x, N):
    """Value to MINIMISE (negative water per turn)."""
    return -volume_per_turn(x, N)


def g_vector(x, N, volume=None):
    """All inequality constraints g >= 0, normalised to O(1)."""
    Ro, rho, P, a = x
    lam = lam_of(x)
    Q = flow_rate(x, N) if volume is None else volume * rpm_of(Ro) / 60.0
    L = H / np.sin(np.radians(a))
    return np.array([
        Q / Q_REQ - 1.0,                    # g1: Q >= Qreq
        (LAM_MAX - lam),                                  # g2: P tan(a) <= 2 pi Ro
        (lam - LAM_MIN),                                  # g3: P >= 0.05*2 pi Ro/tan(a)
        (L_MAX - L * np.cos(np.radians(a))) / L_MAX,      # g4: footprint
    ])


CONSTRAINT_NAMES = ["flow Q>=Qreq", "buckets form (lam<=1)", "min pitch (lam>=0.05)", "footprint"]


def scipy_constraints(N):
    return [{"type": "ineq", "fun": lambda x, N=N: g_vector(x, N)}]


def max_violation(x, N):
    """Largest constraint violation (0 if feasible) including simple bounds."""
    if not np.all(np.isfinite(x)):
        return float('inf')
    g = g_vector(x, N)
    if not np.all(np.isfinite(g)):
        return float('inf')
    v = max(0.0, float(-g.min()))
    for xi, (lo, hi) in zip(x, BOUNDS):
        v = max(v, lo - xi, xi - hi)
    return max(0.0, v)


def random_start(rng):
    return np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])
