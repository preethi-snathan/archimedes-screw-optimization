"""
nu.py -- volumetric capacity factor nu(N, rho, lam) of an Archimedes screw.

Definitions (Ro = 1 after non-dimensionalising):
    N    number of blades (flights)
    rho  = Ri / Ro                       radius ratio
    lam  = P * tan(alpha) / (2*pi*Ro)    dimensionless pitch/tilt parameter
    nu   = (water per turn) / (pi * Ro^2 * P)      so   VT = nu * pi * Ro^2 * P

Method (first-principles geometric model, our reading of Rorres 2000):
  * Work in a frame fixed to the screw. A point at radius r, angle phi
    (phi = 0 at the top of the screw) and axial offset v inside a blade
    channel has non-dimensional height
            z' = lam*phi + v + r*cos(phi)          (divided by cos(alpha))
    where v in [0, 2*pi*lam/N] spans one channel between two blades.
  * Each channel is a helix. Water sits in "pools" along it. The pool's
    free surface is horizontal, and the highest it can rise is the lowest
    barrier (crest) separating it from the next pool down the channel.
  * Crest height h: lowest edge of the channel (v = 0), taking the
    inner radius on the upper half and the outer radius on the lower half.
  * Pool volume = integral of (r dr dphi dv) over the part of the channel
    below the free surface, between the crest and the far wall of the pool.
  * One pitch of axial length holds N pools, so VT = N * (pool volume).

The radial integral is done in closed form; the angular integral uses
Simpson's rule on `n_phi` points.

IMPORTANT: this is OUR geometric reconstruction. Always run
`python validate_nu.py` and compare to Rorres (2000) before trusting it.
"""
import numpy as np
from scipy.optimize import brentq

TWO_PI = 2.0 * np.pi


def _crest_height(rho, lam):
    """Height h of the crest (barrier) on the lower side of the pool."""
    if lam < rho:
        phi_c = np.arcsin(lam / rho)
        return phi_c, lam * phi_c + np.sqrt(rho**2 - lam**2)
    # lam >= rho: crest sits at the kink phi = pi/2 where radius switches
    return np.pi / 2, lam * np.pi / 2


def _radial_integral(a, c, W, rho):
    """Closed form of  int_{rho}^{1} r * clip(a - c*r, 0, W) dr  (vectorised in a, c)."""
    a = np.asarray(a, float)
    c = np.asarray(c, float)
    safe_c = np.where(np.abs(c) < 1e-12, 1e-12, c)
    b1 = (a - W) / safe_c          # where a - c r = W
    b2 = a / safe_c                # where a - c r = 0
    lo = np.minimum(b1, b2)
    hi = np.maximum(b1, b2)
    edges = [np.full_like(a, rho),
             np.clip(lo, rho, 1.0),
             np.clip(hi, rho, 1.0),
             np.full_like(a, 1.0)]
    total = np.zeros_like(a)
    for k in range(3):
        r0, r1 = edges[k], edges[k + 1]
        rm = 0.5 * (r0 + r1)
        val = a - c * rm
        full = val >= W
        empty = val <= 0
        const_part = W * (r1**2 - r0**2) / 2.0
        lin_part = a * (r1**2 - r0**2) / 2.0 - c * (r1**3 - r0**3) / 3.0
        seg = np.where(full, const_part, np.where(empty, 0.0, lin_part))
        total += seg
    return total


def nu(N, rho, lam, n_phi=2001):
    """Volumetric capacity factor. Returns 0 if no water-holding bucket forms."""
    if not (0.0 < rho < 1.0) or lam <= 0.0 or lam >= 1.0:
        return 0.0
    W = TWO_PI * lam / N                      # channel width in v
    phi_c, h = _crest_height(rho, lam)

    # far wall of the pool: the point on the rising side where the channel's
    # lowest edge z_min(phi) climbs back up to the free-surface height h.
    def zmin(p):
        r_edge = np.where(np.cos(p) > 0, rho, 1.0)
        return lam * p + r_edge * np.cos(p)

    phi_min = np.pi - np.arcsin(lam)          # floor of the pool
    phi_next = TWO_PI + phi_c                 # next crest, at height h + 2*pi*lam
    g = lambda p: float(zmin(p)) - h
    if g(phi_min) >= 0:                       # surface at/below the floor: no pool
        return 0.0
    phi_r = brentq(g, phi_min, phi_next)

    phi = np.linspace(phi_c, phi_r, n_phi)
    a = h - lam * phi
    c = np.cos(phi)
    f = _radial_integral(a, c, W, rho)

    # Simpson's rule (n_phi odd)
    dphi = (phi_r - phi_c) / (n_phi - 1)
    integral = dphi / 3.0 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())

    # nu = N * pool_volume / (pi * Ro^2 * P), pool_volume = integral * (1/tan(alpha))
    # and P = 2*pi*lam / tan(alpha)  ->  nu = N * integral / (2 * pi^2 * lam)
    return float(N * integral / (2.0 * np.pi**2 * lam))


if __name__ == "__main__":
    print("nu(8, 0.5, 3/8) =", nu(8, 0.5, 3 / 8), "  (Rorres 2000: ~0.1703)")
