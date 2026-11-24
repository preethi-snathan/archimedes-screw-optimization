"""Numerical checks against Rorres (2000), Table 1 (p. 76), N=1..8.

Source: https://www.cs.drexel.edu/~crorres/screw/screw.pdf
DOI: 10.1061/(ASCE)0733-9429(2000)126:1(72)
Values transcribed from the indexed primary-source table, 2026-11-23.
The parameter tolerance (5e-4) allows published numerical-search differences;
objective/benchmark tolerance (5e-5) is half the four-decimal reporting unit.
These checks validate numerical agreement, not every physical assumption.
"""
import csv
import json
from pathlib import Path
import numpy as np
from nu import nu
import optimize as O

REF_NU = 0.1703
# N: rho*, lambda*, lambda*nu* (published rounded values, never fitted).
REFERENCES = {
    1: (.5358, .1285, .0361), 2: (.5369, .1863, .0512),
    3: (.5357, .2217, .0598), 4: (.5353, .2456, .0655),
    5: (.5352, .2630, .0696), 6: (.5353, .2763, .0727),
    7: (.5354, .2869, .0752), 8: (.5354, .2957, .0771),
}
PARAM_TOL, OBJECTIVE_TOL, GRID_TOL = 5e-4, 5e-5, 1e-7


def write_csv(path, rows):
    # Always replace previous output, including when no valid designs remain.
    with Path(path).open('w', newline='') as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(out='results', n_starts=8, seed=1):
    if n_starts < 1:
        raise ValueError('n_starts must be positive')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    benchmark = nu(8, .5, .375)
    checks = dict(benchmark=abs(benchmark-REF_NU) <= OBJECTIVE_TOL)
    comparisons, runs, grids = [], [], []
    for N, reference in REFERENCES.items():
        candidates = []
        # Positive-capacity start region; bounds still cover the unit square.
        for i in range(n_starts):
            r = O.solve_two_var(N, [rng.uniform(.2, .7), rng.uniform(.05, .4)])
            runs.append(dict(N=N, run=i, rho0=r['x0'][0], lambda0=r['x0'][1],
                rho=r['x'][0], lam=r['x'][1], objective=r['obj'],
                success=r['success'], status=r['status'], message=r['message'],
                nfev=r['nfev'], nit=r['nit'], seconds=r['time']))
            if r['success'] and np.isfinite(r['obj']) and r['obj'] > 0:
                candidates.append(r)
        if not candidates:
            raise RuntimeError(f'No converged validation solution for N={N}')
        b = max(candidates, key=lambda r: r['obj'])
        actual = [*b['x'], b['obj']]
        errors = np.abs(np.array(actual)-reference)
        passed = bool(np.all(errors <= [PARAM_TOL, PARAM_TOL, OBJECTIVE_TOL]))
        checks[f'table_N{N}'] = passed
        comparisons.append(dict(N=N, rho_ref=reference[0], rho=actual[0],
            rho_abs_error=errors[0], lambda_ref=reference[1], lam=actual[1],
            lambda_abs_error=errors[1], objective_ref=reference[2], objective=actual[2],
            objective_abs_error=errors[2], objective_relative_error=errors[2]/reference[2],
            successful_starts=len(candidates), starts=n_starts, passed=passed))
        for n in (501, 1001, 2001, 4001, 8001):
            grids.append(dict(case=f'optimum_N{N}', N=N, rho=actual[0], lam=actual[1],
                              n_phi=n, nu=nu(N, *b['x'], n_phi=n)))
    for n in (501, 1001, 2001, 4001, 8001):
        grids.append(dict(case='benchmark', N=8, rho=.5, lam=.375,
                          n_phi=n, nu=nu(8, .5, .375, n_phi=n)))
    for case in sorted({r['case'] for r in grids}):
        values = {r['n_phi']: r['nu'] for r in grids if r['case'] == case}
        checks[f'grid_{case}'] = abs(values[2001]-values[8001]) <= GRID_TOL
    write_csv(out/'validation.csv', comparisons)
    write_csv(out/'validation_runs.csv', runs)
    write_csv(out/'integration_refinement.csv', grids)
    summary = dict(passed=all(checks.values()), benchmark=benchmark,
        reference=REF_NU, benchmark_abs_error=abs(benchmark-REF_NU),
        tolerances=dict(parameters=PARAM_TOL, objective=OBJECTIVE_TOL, grid=GRID_TOL),
        seed=seed, n_starts=n_starts, checks=checks,
        source='https://www.cs.drexel.edu/~crorres/screw/screw.pdf, Table 1, p. 76')
    (out/'validation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Validation: {"PASS" if summary["passed"] else "FAIL"}; benchmark={benchmark:.9f}')
    if not summary['passed']:
        raise RuntimeError('Validation failed: '+', '.join(k for k,v in checks.items() if not v))
    return summary


if __name__ == '__main__':
    run()
