"""Reproducible solvers; histories contain best feasible physical volume.

Model-call counts include objective, constraint, and final diagnostic evaluations,
excluding feasible-start generation. Timings include solver and final diagnostics.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from nu import nu
import constraints as C

FEAS_TOL = 1e-6
SLSQP_OPTIONS = dict(maxiter=300, ftol=1e-10, eps=1e-6)


def solve_two_var(N, x0, method="SLSQP", n_phi=2001, eps=1e-6):
    """Maximise lambda*nu over the paper's unit square (endpoints excluded)."""
    calls = 0
    def objective(x):
        nonlocal calls
        calls += 1
        return -x[1] * nu(N, x[0], x[1], n_phi=n_phi)
    t0 = time.perf_counter()
    options = dict(maxiter=300)
    if method == "SLSQP":
        options.update(ftol=1e-10, eps=eps)
    elif method == "Nelder-Mead":
        options.update(xatol=1e-8, fatol=1e-10)
    res = minimize(objective, x0, method=method,
                   bounds=[(1e-6, 1-1e-6)] * 2, options=options)
    return dict(N=N, x=res.x, x0=np.asarray(x0), obj=float(-res.fun),
                nfev=calls, time=time.perf_counter()-t0,
                success=bool(res.success), status=int(res.status),
                message=str(res.message), nit=int(res.nit))


class Evaluator:
    """Count every geometry integration without global monkey-patching/caching."""
    def __init__(self, N, n_phi=2001):
        self.N, self.n_phi = N, n_phi
        self.model_calls = self.nfev = self.ncev = 0
        self.history = []
        self.best = None

    def volume(self, x):
        self.model_calls += 1
        return nu(self.N, x[1], C.lam_of(x), n_phi=self.n_phi)*np.pi*x[0]**2*x[2]

    def constraints(self, x):
        self.ncev += 1
        return C.g_vector(x, self.N, volume=self.volume(x))

    def objective(self, x, penalty=0.0):
        self.nfev += 1
        v = self.volume(x)
        g = C.g_vector(x, self.N, volume=v)
        violation = constraint_violation(x, g)
        if np.isfinite(v) and violation <= FEAS_TOL:
            self.best = v if self.best is None else max(self.best, v)
        # Negative sign retained for compatibility with plots.py.
        self.history.append(None if self.best is None else -self.best)
        return -v + penalty*np.sum(np.minimum(g, 0.0)**2)


def constraint_violation(x, g):
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)):
        return float("inf")
    bounds = np.asarray(C.BOUNDS)
    return float(max(0., -np.min(g), np.max(bounds[:, 0]-x), np.max(x-bounds[:, 1])))


def _pack(N, res, ev, t0, x0=None):
    v = ev.volume(res.x)
    g = C.g_vector(res.x, N, volume=v)
    violation = constraint_violation(res.x, g)
    return dict(N=N, x=np.asarray(res.x), x0=None if x0 is None else np.asarray(x0),
                obj=float(v), nfev=ev.nfev, ncev=ev.ncev, model_calls=ev.model_calls,
                time=time.perf_counter()-t0, violation=violation,
                feasible=violation <= FEAS_TOL, success=bool(res.success),
                status=int(getattr(res, 'status', 0 if res.success else -1)),
                message=str(res.message), nit=int(res.nit), history=ev.history,
                residuals=g.tolist())


def solve_slsqp(N, x0, n_phi=2001, eps=1e-6):
    ev = Evaluator(N, n_phi)
    t0 = time.perf_counter()
    res = minimize(ev.objective, x0, method="SLSQP", bounds=C.BOUNDS,
                   constraints=[dict(type="ineq", fun=ev.constraints)],
                   options={**SLSQP_OPTIONS, "eps": eps})
    return _pack(N, res, ev, t0, x0)


def solve_nelder_mead(N, x0, mu=1e5):
    ev = Evaluator(N)
    t0 = time.perf_counter()
    res = minimize(lambda x: ev.objective(x, penalty=mu), x0, method="Nelder-Mead",
                   bounds=C.BOUNDS, options=dict(maxiter=4000, maxfev=6000,
                                                 xatol=1e-7, fatol=1e-10))
    return _pack(N, res, ev, t0, x0)


def solve_de(N, seed=0, popsize=15, maxiter=150):
    """Stochastic comparison; finite runs do not certify a global optimum."""
    ev = Evaluator(N)
    t0 = time.perf_counter()
    res = differential_evolution(ev.objective, C.BOUNDS,
        constraints=NonlinearConstraint(ev.constraints, 0., np.inf), seed=seed,
        popsize=popsize, maxiter=maxiter, tol=1e-8, polish=False)
    return _pack(N, res, ev, t0)


def feasible_start(N, rng, tries=200):
    """Reject infeasible points; fail explicitly if none can be found."""
    for _ in range(tries):
        x = C.random_start(rng)
        if C.max_violation(x, N) <= FEAS_TOL:
            return x
    raise RuntimeError(f"No feasible start found for N={N} after {tries} attempts")
