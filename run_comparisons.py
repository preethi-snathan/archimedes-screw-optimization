"""Complete the three-method study without overwriting the original SLSQP outputs.

Run from the repository root: python3 run_comparisons.py
Main settings are fixed before the full study: 20 paired starts per N, 3 DE seeds,
NM penalty 1e7 (the original 1e5 allowed infeasible termination in pilot checks).
The separate penalty sweep retains all outcomes; it is not used to cherry-pick
main-study runs. These finite searches do not prove global optimality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import constraints as C
import experiments as E
import optimize as O
import validate_nu as V
import analysis
import review_checks
from nu import nu


def audit_histories(rows, out):
    histories=json.loads((out/'histories.json').read_text())
    counts=json.loads((out/'history_model_calls.json').read_text())
    for row in rows:
        key=f"{row['method']}_N{row['N']}_run{row['run']}"
        h,c=histories[key],counts[key]
        finite=[v for v in h if v is not None]
        if len(h)!=row['nfev'] or len(c)!=len(h):
            raise RuntimeError('History length does not match objective evaluations')
        if any(b<=a for a,b in zip(c,c[1:])) or (c and c[-1]>=row['model_calls']):
            raise RuntimeError('Incorrect cumulative model-call history')
        if not np.all(np.isfinite(finite)) or any(b>a for a,b in zip(finite,finite[1:])):
            raise RuntimeError('Best-feasible history is nonfinite or nonmonotone')
        if finite and any(v is None for v in h[next(i for i,v in enumerate(h) if v is not None):]):
            raise RuntimeError('History lost its feasible incumbent')


def penalty_study(rows, out):
    output = []
    for N in (1,2,3,4):
        starts = [r for r in rows if r['N']==N and r['method']=='SLSQP'][:3]
        for baseline in starts:
            x0 = np.array([baseline['x0_'+k] for k in C.X_NAMES])
            for mu in (1e5,1e7,1e9):
                r = O.solve_nelder_mead(N,x0,mu=mu)
                row = E.result_row(r,'NelderMead',baseline['run'],baseline['seed'])
                row.update(penalty_mu=mu, paired_SLSQP_VT=baseline['obj'])
                output.append(row)
    V.write_csv(out/'penalty_sensitivity.csv',output)
    return output


def audit_results(rows, out):
    """Verify exported values and report solution quality at a common resolution."""
    audit_histories(rows,out)
    summaries = []
    checks = []
    for r in rows:
        x = np.array([r[k] for k in C.X_NAMES])
        v = C.volume_per_turn(x,r['N'])
        g = C.g_vector(x,r['N'],volume=v)
        violation = O.constraint_violation(x,g)
        if not np.isclose(v,r['obj'],atol=1e-12,rtol=0):
            raise RuntimeError('Stored objective does not match model')
        if (violation<=O.FEAS_TOL) != r['feasible']:
            raise RuntimeError('Stored feasibility does not match constraints')
        if r['model_calls'] != r['nfev']+r['ncev']+1:
            raise RuntimeError('Incorrect model-call accounting')
        refined = nu(r['N'],x[1],C.lam_of(x),n_phi=8001)*np.pi*x[0]**2*x[2]
        refined_g = C.g_vector(x,r['N'],volume=refined)
        refined_feasible=O.constraint_violation(x,refined_g)<=O.FEAS_TOL
        if abs(refined-v)>1e-6 or refined_feasible!=r['feasible']:
            raise RuntimeError('Final design is sensitive to integration resolution')
        checks.append(dict(N=r['N'],method=r['method'],run=r['run'],
            VT_2001=v,VT_8001=refined,abs_grid_difference=abs(refined-v),
            feasible_2001=r['feasible'],feasible_8001=refined_feasible))
    for N in (1,2,3,4):
        local = [r for r in rows if r['N']==N and r['method']=='SLSQP']
        nm = [r for r in rows if r['N']==N and r['method']=='NelderMead']
        if len(local) != len(nm):
            raise RuntimeError('Comparison methods must have paired starts')
        for a,b in zip(local,nm):
            if a['run'] != b['run'] or [a['x0_'+k] for k in C.X_NAMES] != [b['x0_'+k] for k in C.X_NAMES]:
                raise RuntimeError('Comparison starting points do not match')
        baseline = max(r['obj'] for r in local if r['success'] and r['feasible'])
        for method in ('SLSQP','NelderMead','DiffEvolution'):
            rs = [r for r in rows if r['N']==N and r['method']==method]
            good = [r for r in rs if r['success'] and r['feasible']]
            near = [r for r in good if abs(r['obj']-baseline)<=1e-5*abs(baseline)]
            summaries.append(dict(N=N,method=method,runs=len(rs),
                converged_feasible=len(good),within_1e_minus_5_relative_of_SLSQP=len(near),
                worst_converged_VT=min(r['obj'] for r in good) if good else '',
                best_converged_VT=max(r['obj'] for r in good) if good else '',
                SLSQP_reference_VT=baseline,
                best_difference_from_SLSQP=max(r['obj'] for r in good)-baseline if good else ''))
    V.write_csv(out/'solution_consistency.csv',summaries)
    V.write_csv(out/'final_grid_checks.csv',checks)
    return summaries,checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='results/comparison')
    parser.add_argument('--figures',default='figures/comparison')
    args=parser.parse_args()
    out=Path(args.out)
    if out.resolve()==Path('results').resolve():
        parser.error('Use a separate output directory to preserve the original study')
    V.run(out=out)
    rows=E.run(comparisons=True,out=out,nm_mu=1e7)
    print('Checking SLSQP numerical sensitivity...',flush=True)
    E.sensitivity(rows,out=out)
    print('Checking penalty sensitivity...',flush=True)
    penalty=penalty_study(rows,out)
    consistency,checks=audit_results(rows,out)
    print('Checking convexity and suboptimal Nelder-Mead endpoints...',flush=True)
    polish=analysis.run(rows,out)
    print('Cross-checking integration, reduced formulation and pitch bound...',flush=True)
    review=review_checks.run(rows,out)
    import comparison_plots
    comparison_plots.generate(out,Path(args.figures))
    m=json.loads((out/'metadata.json').read_text())
    m.update(penalty_sensitivity=dict(coefficients=[1e5,1e7,1e9],
                 starts='first three paired starts per N',run_count=len(penalty)),
             numerical_sensitivity=dict(n_phi=[1001,2001,4001],eps=[1e-5,1e-6,1e-7],
                 starts='original start of the best baseline SLSQP run per N'),
             final_grid_check=dict(n_phi=8001,runs=len(checks)),
             model_review=review,
             nm_polish_diagnostics=dict(runs=len(polish),
                 selection='converged feasible NM endpoints more than 1e-5 relative below the best SLSQP value for that N',
                 timing='separate diagnostic; excluded from main comparison'),
             nm_penalty_rationale='Pilot checks at N=1 and N=4 found 1e5 infeasible and 1e7 feasible under the common 1e-6 threshold. Full study uses fixed 1e7; sweep reported separately.',
             source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in Path(__file__).resolve().parent.glob('*.py')})
    (out/'metadata.json').write_text(json.dumps(m,indent=2)+'\n')
    print('Complete. Tables:',out,'Figures:',args.figures,flush=True)


if __name__=='__main__':
    main()
