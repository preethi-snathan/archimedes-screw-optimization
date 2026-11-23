"""
optimize.py -- solvers for the two-variable (Rorres) and five-variable problems.
Every solver returns a dict with x, objective value, evaluations, time, violation.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from nu import nu
import constraints as C


class Counter:
    """Wraps a function to count evaluations and record the best-so-far history."""
    def __init__(self, f):
        self.f, self.n, self.history = f, 0, []
    def __call__(self, x):
        v = self.f(x)
        self.n += 1
        best = v if not self.history else min(self.history[-1], v)
        self.history.append(best)
        return v


# ----------------------------------------------------------------------------
# Stage 1: reproduce Rorres (2000): maximise lam*nu over (rho, lam) for given N
# ----------------------------------------------------------------------------
def solve_two_var(N, x0, method="SLSQP"):
    f = Counter(lambda x: -x[1] * nu(N, x[0], x[1]))
    t0 = time.perf_counter()
    res = minimize(f, x0, method=method, bounds=[(0.2, 0.7), (0.02, 0.98)],
                   options={"maxiter": 200, "ftol": 1e-10})
    return dict(N=N, x=res.x, obj=-res.fun, nfev=f.n,
                time=time.perf_counter() - t0, success=bool(res.success))


# ----------------------------------------------------------------------------
# Stage 2: five-variable problem, one N at a time
# ----------------------------------------------------------------------------
def _pack(N, x, f, t0, success=True):
    return dict(N=N, x=np.asarray(x), obj=-f.f(x) if False else -C.objective(x, N),
                nfev=f.n, time=time.perf_counter() - t0,
                violation=C.max_violation(x, N), success=success,
                history=list(f.history))


def solve_slsqp(N, x0):
    f = Counter(lambda x: C.objective(x, N))
    t0 = time.perf_counter()
    res = minimize(f, x0, method="SLSQP", bounds=C.BOUNDS,
                   constraints=C.scipy_constraints(N),
                   options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-6})
    return _pack(N, res.x, f, t0, bool(res.success))


def solve_nelder_mead(N, x0, mu=1e5):
    """Nelder-Mead with a quadratic penalty on constraint violation."""
    def pen(x):
        g = C.g_vector(x, N)
        return C.objective(x, N) + mu * np.sum(np.minimum(g, 0.0) ** 2)
    f = Counter(pen)
    t0 = time.perf_counter()
    res = minimize(f, x0, method="Nelder-Mead", bounds=C.BOUNDS,
                   options={"maxiter": 4000, "maxfev": 6000, "xatol": 1e-7, "fatol": 1e-10})
    return _pack(N, res.x, f, t0, bool(res.success))


def solve_de(N, seed=0, popsize=15, maxiter=150):
    """Differential evolution (global); constraints via SciPy's feasibility rules."""
    f = Counter(lambda x: C.objective(x, N))
    nlc = NonlinearConstraint(lambda x: C.g_vector(x, N), 0.0, np.inf)
    t0 = time.perf_counter()
    res = differential_evolution(f, C.BOUNDS, constraints=nlc, seed=seed,
                                 popsize=popsize, maxiter=maxiter, tol=1e-8,
                                 polish=False)
    return _pack(N, res.x, f, t0, bool(res.success))


def feasible_start(N, rng, tries=200):
    """Random start that satisfies the constraints (helps SLSQP/NM). Falls back to any."""
    best = None
    for _ in range(tries):
        x = C.random_start(rng)
        v = C.max_violation(x, N)
        if v == 0.0:
            return x
        if best is None or v < best[0]:
            best = (v, x)
    return best[1]
