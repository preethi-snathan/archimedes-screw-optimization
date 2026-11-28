"""Additional model checks, separate from the main solver timing comparison.

Uses nested adaptive quadrature to cross-check the production radial formula and
Simpson integration; checks a reduced formulation and the extra pitch box bound.
This tests the implemented ideal geometry, not real hydraulic performance.
"""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, differential_evolution, minimize
import constraints as C
import optimize as O
from nu import nu
from validate_nu import write_csv


def adaptive_nu(N, rho, lam):
    """Same pool geometry, independent radial/angular quadrature implementation."""
    start = np.arcsin(lam/rho) if lam < rho else np.pi/2
    floor = lambda phi: lam*phi + min(rho*np.cos(phi), np.cos(phi))
    height = floor(start)
    end = brentq(lambda phi: floor(phi)-height,
                 np.pi-np.arcsin(lam), 2*np.pi+start)
    width = 2*np.pi*lam/N

    def radial(phi):
        a, c = height-lam*phi, np.cos(phi)
        splits = [r for r in (a/c, (a-width)/c) if rho < r < 1] if abs(c)>1e-14 else []
        # Integrate the clipped water depth directly, without _radial_integral.
        return quad(lambda r: r*max(0., min(width, a-c*r)), rho, 1.,
                    points=splits, epsabs=1e-10, epsrel=1e-10)[0]

    value, error = quad(radial, start, end, epsabs=1e-9, epsrel=1e-9, limit=200)
    scale = N/(2*np.pi**2*lam)
    return value*scale, error*scale


def integration_checks():
    cases = [(8, .5, .375)]
    # Cover both crest branches, their junction, and near-empty buckets.
    for N in (1, 2, 3, 4):
        cases.extend((N, r, l) for r,l in [(.2,.05),(.7,.5),(.2,.6),
                                          (.5,.5-1e-8),(.5,.5+1e-8),(.5,.99)])
    rows=[]
    for N, rho, lam in cases:
        reference, error = adaptive_nu(N, rho, lam)
        computed = nu(N, rho, lam)
        difference = abs(computed-reference)
        physical = 0 <= computed <= 1-rho**2 + 1e-10
        rows.append(dict(N=N, rho=rho, lam=lam, nu_simpson=computed,
                         nu_adaptive=reference, abs_difference=difference,
                         outer_quad_error_estimate=error,
                         within_annular_volume_bound=physical,
                         passed=difference<=1e-7 and physical))
    return rows


def design_checks(rows):
    """Two separate checks: fixed-boundary reduction and relaxed pitch cap."""
    alpha_min=max(C.BOUNDS[3][0], np.degrees(np.arctan(C.H/C.L_MAX)))
    radius_max=C.BOUNDS[0][1]
    pitch_max=2*np.pi*radius_max*C.LAM_MAX/np.tan(np.radians(alpha_min))
    expanded=list(C.BOUNDS)
    expanded[2]=(C.BOUNDS[2][0],pitch_max)
    reduced, relaxed = [], []
    for N in (1,2,3,4):
        baseline=[r for r in rows if int(r['N'])==N and r['method']=='SLSQP']
        accepted=[r for r in baseline if str(r['success'])=='True' and str(r['feasible'])=='True']
        reference=max(float(r['obj']) for r in accepted)
        # VT = 2*pi^2*Ro^3*(lambda*nu)/tan(alpha). At radius_max and alpha_min,
        # solve only for rho and lambda. Feasibility of the resulting design is
        # checked explicitly; finite DE is not a certified bound on lambda*nu.
        res=differential_evolution(lambda z: -z[1]*nu(N,z[0],z[1]),
            [C.BOUNDS[1],(C.LAM_MIN,C.LAM_MAX)],seed=2026,
            popsize=15,maxiter=200,tol=1e-10,polish=False)
        rho,lam=res.x
        x=np.array([radius_max,rho,2*np.pi*radius_max*lam/np.tan(np.radians(alpha_min)),alpha_min])
        volume=C.volume_per_turn(x,N)
        reduced.append(dict(N=N,Ro=x[0],rho=x[1],P=x[2],alpha_deg=x[3],
            obj=volume,success=bool(res.success),feasible=C.max_violation(x,N)<=O.FEAS_TOL,
            original_SLSQP_VT=reference,absolute_difference=abs(volume-reference),
            agrees=abs(volume-reference)<=1e-6))

        starts=[(int(r['run']),np.array([float(r['x0_'+k]) for k in C.X_NAMES]),'paired') for r in baseline]
        starts.append((-1,np.array([radius_max,.53524,6.5,alpha_min]),'beyond_original_pitch_cap'))
        for run,x0,kind in starts:
            res=minimize(lambda x: -C.volume_per_turn(x,N),x0,method='SLSQP',
                bounds=expanded,constraints=[dict(type='ineq',fun=lambda x:C.g_vector(x,N))],
                options=O.SLSQP_OPTIONS)
            g=C.g_vector(res.x,N)
            bound_array=np.asarray(expanded)
            violation=float(max(0.,-np.min(g),np.max(bound_array[:,0]-res.x),
                                np.max(res.x-bound_array[:,1])))
            volume=C.volume_per_turn(res.x,N)
            relaxed.append(dict(N=N,run=run,start_kind=kind,pitch_upper= pitch_max,
                **dict(zip(C.X_NAMES,res.x)),
                **{'x0_'+k:x0[i] for i,k in enumerate(C.X_NAMES)},
                obj=volume,success=bool(res.success),message=str(res.message),
                feasible=violation<=O.FEAS_TOL,violation=violation,
                original_SLSQP_VT=reference,absolute_difference=abs(volume-reference),
                agrees=abs(volume-reference)<=1e-6))
    return reduced,relaxed


def run(rows, out):
    out=Path(out)
    integration=integration_checks()
    reduced,relaxed=design_checks(rows)
    write_csv(out/'independent_integration.csv',integration)
    write_csv(out/'reduced_formulation.csv',reduced)
    write_csv(out/'pitch_bound_sensitivity.csv',relaxed)
    summary=dict(integration_cases=len(integration),
        integration_passed=all(r['passed'] for r in integration),
        integration_max_abs_difference=max(r['abs_difference'] for r in integration),
        reduced_cases=len(reduced),
        reduced_all_agree=all(r['success'] and r['feasible'] and r['agrees'] for r in reduced),
        relaxed_pitch_cases=len(relaxed),
        relaxed_pitch_all_agree=all(r['success'] and r['feasible'] and r['agrees'] for r in relaxed),
        scope='Same ideal geometric model; independent quadrature does not test leakage, inlet filling or finite-length end effects. Finite reduced DE does not certify global optimality.')
    (out/'review_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not summary['integration_passed']:
        raise RuntimeError('Independent integration cross-check failed')
    return summary


if __name__=='__main__':
    out=Path('results/comparison')
    with (out/'results.csv').open() as f:
        rows=list(csv.DictReader(f))
    print(json.dumps(run(rows,out),indent=2))
