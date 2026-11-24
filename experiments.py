"""Seeded SLSQP study and optional comparisons, with report-ready CSV outputs."""
import json
import hashlib
from importlib.metadata import version
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
import constraints as C
import optimize as O
from nu import nu
from validate_nu import write_csv

N_STARTS = 20
N_DE_SEEDS = 3
OUT = 'results'


def design_details(x, N, volume=None):
    # Reuse the solver's volume so sensitivity rows use that run's grid.
    if volume is None:
        volume = C.volume_per_turn(x, N)
    g = C.g_vector(x, N, volume=volume)
    active = [name for name, value in zip(C.CONSTRAINT_NAMES, g) if abs(value) <= 1e-5]
    for name, xi, (lo, hi) in zip(C.X_NAMES, x, C.BOUNDS):
        if abs(xi-lo) <= 1e-5:
            active.append(name+'_lower')
        if abs(xi-hi) <= 1e-5:
            active.append(name+'_upper')
    return dict(lam=C.lam_of(x), Q=volume*C.rpm_of(x[0])/60.0, rpm=C.rpm_of(x[0]),
        length_m=C.H/np.sin(np.radians(x[3])), footprint_m=C.H/np.tan(np.radians(x[3])),
        **{f'g{i+1}': value for i,value in enumerate(g)},
        active_constraints='; '.join(active))


def result_row(r, name, i, seed):
    return dict(N=r['N'], method=name, run=i, seed=seed,
        **{k:r[k] for k in ('obj','nfev','ncev','model_calls','nit','time',
                             'violation','feasible','success','status','message')},
        **dict(zip(C.X_NAMES, r['x'])),
        **{f'x0_{k}': '' if r['x0'] is None else r['x0'][j] for j,k in enumerate(C.X_NAMES)},
        **design_details(r['x'], r['N'], volume=r['obj']))


def run(n_starts=N_STARTS, n_de=N_DE_SEEDS, seed=0, verbose=True,
        comparisons=False, out=OUT):
    if n_starts < 1 or n_de < 0:
        raise ValueError('Require positive starts and nonnegative DE runs')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, hist = [], {}
    for N in (1,2,3,4):
        starts = [O.feasible_start(N, rng) for _ in range(n_starts)]
        solvers = [('SLSQP', O.solve_slsqp)]
        if comparisons:
            solvers.append(('NelderMead', O.solve_nelder_mead))
        for i,x0 in enumerate(starts):
            for name,solver in solvers:
                r = solver(N,x0)
                rows.append(result_row(r,name,i,seed))
                hist[f'{name}_N{N}_run{i}'] = r['history']
                if i == 0:
                    hist[f'{name}_N{N}'] = r['history']
        if comparisons:
            for i in range(n_de):
                r = O.solve_de(N,seed=seed+i)
                rows.append(result_row(r,'DiffEvolution',i,seed+i))
                hist[f'DiffEvolution_N{N}_run{i}'] = r['history']
                if i == 0:
                    hist[f'DiffEvolution_N{N}'] = r['history']
        if verbose:
            print(f'N={N} finished', flush=True)
    write_csv(out/'results.csv', rows)
    (out/'histories.json').write_text(json.dumps(hist, allow_nan=False)+'\n')
    summaries, best = summarise(rows, verbose=verbose)
    write_csv(out/'method_summary.csv',summaries)
    write_csv(out/'best_designs.csv',best)
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), matplotlib=version("matplotlib"), seed=seed, n_starts=n_starts,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).resolve().parent.glob("*.py")},
        comparisons=comparisons, n_de=n_de if comparisons else 0,
        solver_options=O.SLSQP_OPTIONS, feasibility_tolerance=O.FEAS_TOL,
        n_phi=2001, bounds=C.BOUNDS, H=C.H, Q_REQ=C.Q_REQ, L_MAX=C.L_MAX,
        lam_bounds=[C.LAM_MIN,C.LAM_MAX],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True).strip(),
        timing_scope='solver plus final diagnostics; excludes start generation and report postprocessing',
        history='negative best feasible physical VT at objective evaluations; null until feasible',
        model_calls_scope='all nu calls inside solver and final diagnostics; excludes start generation and report postprocessing')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return rows


def summarise(rows, verbose=True):
    summaries, best = [], []
    for N, method in sorted({(r['N'],r['method']) for r in rows}):
        rs = [r for r in rows if r['N']==N and r['method']==method]
        feas = [r for r in rs if r['feasible']]
        good = [r for r in feas if r['success']]
        vals = [r['obj'] for r in good]
        summary = dict(N=N, method=method, runs=len(rs), feasible_runs=len(feas),
            successful_feasible_runs=len(good),
            best_converged_VT=max(vals) if vals else '',
            median_converged_VT=float(np.median(vals)) if vals else '',
            std_converged_VT=float(np.std(vals)) if vals else '',
            best_feasible_VT=max(r['obj'] for r in feas) if feas else '',
            mean_model_calls=float(np.mean([r['model_calls'] for r in rs])),
            mean_seconds=float(np.mean([r['time'] for r in rs])))
        summaries.append(summary)
        if good:
            b = max(good,key=lambda r:r['obj']).copy()
            x = np.array([b[k] for k in C.X_NAMES])
            refined = nu(N,x[1],C.lam_of(x),n_phi=8001)*np.pi*x[0]**2*x[2]
            b.update(refined_VT=refined, grid_abs_difference=abs(refined-b['obj']))
            best.append(b)
        if verbose:
            print(f'N={N} {method}: converged+feasible {len(good)}/{len(rs)}, best VT={max(vals) if vals else "none"}')
    return summaries,best


def sensitivity(rows, out=OUT):
    """Rerun each best SLSQP run from its original start at varied resolution/step."""
    output=[]
    for N in (1,2,3,4):
        candidates=[r for r in rows if r['N']==N and r['method']=='SLSQP' and r['success'] and r['feasible']]
        if not candidates:
            raise RuntimeError(f'No converged feasible SLSQP result for N={N}')
        best=max(candidates,key=lambda r:r['obj'])
        x0=np.array([best['x0_'+k] for k in C.X_NAMES])
        for n_phi in (1001,2001,4001):
            for eps in (1e-5,1e-6,1e-7):
                r=O.solve_slsqp(N,x0,n_phi=n_phi,eps=eps)
                row=result_row(r,'SLSQP',0,best['seed'])
                row.update(n_phi=n_phi,eps=eps,VT_difference_from_baseline=r['obj']-best['obj'])
                output.append(row)
    write_csv(Path(out)/'solver_sensitivity.csv',output)
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--comparisons',action='store_true')
    parser.add_argument('--starts',type=int,default=20)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    run(n_starts=args.starts,seed=args.seed,comparisons=args.comparisons)
