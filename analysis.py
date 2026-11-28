"""Reproducible evidence for the report's optimization discussion.

These diagnostics are separate from the main timing comparison. Finite searches,
solver termination and numerical counterexamples are not global certificates.
"""
import json
from pathlib import Path
import numpy as np
import constraints as C
import experiments as E
import optimize as O
from nu import nu
from validate_nu import write_csv


def evaluate_point(x, N, n_phi=8001):
    x = np.asarray(x, dtype=float)
    volume = nu(N, x[1], C.lam_of(x), n_phi=n_phi)*np.pi*x[0]**2*x[2]
    g = C.g_vector(x, N, volume=volume)
    violation = O.constraint_violation(x, g)
    return dict(x=x.tolist(), VT=float(volume), negative_VT=float(-volume),
                residuals=g.tolist(), violation=violation,
                feasible=violation <= O.FEAS_TOL)


def convexity_evidence():
    # An affine line in the original variables with fixed rho, alpha and lambda.
    # On this line f=-VT=-k*Ro^3, k>0, so f''=-6*k*Ro<0.
    rho, lam, angle = .53524, .2456, 32.
    points = [np.array([r, rho, 2*np.pi*r*lam/np.tan(np.radians(angle)), angle])
              for r in (.8, .9, 1.)]
    evaluations = [evaluate_point(x, 4) for x in points]
    gap = evaluations[1]['negative_VT'] - (
        evaluations[0]['negative_VT'] + evaluations[2]['negative_VT'])/2
    if not all(p['feasible'] for p in evaluations) or gap <= 1e-3:
        raise RuntimeError('Nonconvex objective witness is no longer valid')

    # Fixed counterexample discovered by a seeded feasible-pair search. Saving
    # exact endpoints makes verification independent of random sampling order.
    a = np.array([.9530920930730385, .21395028460045595,
                  5.8642775620518535, 31.18544662785248])
    b = np.array([.4062282735178429, .5603669819207846,
                  1.2971498738639076, 33.5976230020679])
    region = [evaluate_point(x, 4) for x in (a, (a+b)/2, b)]
    if not (region[0]['feasible'] and region[2]['feasible']
            and region[1]['residuals'][0] < -1e-3):
        raise RuntimeError('Nonconvex feasible-region witness is no longer valid')
    return dict(N=4, n_phi=8001,
        objective=dict(points= evaluations, midpoint_Jensen_gap=gap,
            explanation='For minimization f=-VT, convexity requires f(mid)<=mean(f(endpoints)). The positive gap violates that inequality. Analytically f=-k*Ro^3 on this affine feasible segment, k>0.'),
        feasible_region=dict(points=region,
            explanation='Both endpoints satisfy the constraints; their midpoint violates minimum flow. This counterexample shows the continuous N=4 feasible region is nonconvex under the implemented model.'),
        limitations='Nonconvexity does not establish multiple local optima or prove that the best computed point is global. N is discrete and is enumerated separately.')


def polish_suboptimal_nm(rows, out):
    """Test solver stagnation without including rescue costs in the main study."""
    output = []
    for N in (1, 2, 3, 4):
        ref = max(r['obj'] for r in rows if r['N']==N and r['method']=='SLSQP'
                  and r['success'] and r['feasible'])
        for original in rows:
            if not (original['N']==N and original['method']=='NelderMead'
                    and original['success'] and original['feasible']
                    and ref-original['obj'] > 1e-5*abs(ref)):
                continue
            x0 = np.array([original[k] for k in C.X_NAMES])
            r = O.solve_slsqp(N, x0)
            record = E.result_row(r, 'SLSQP_from_NM', original['run'], original['seed'])
            record.update(original_NM_VT=original['obj'], SLSQP_reference_VT=ref,
                          improvement=r['obj']-original['obj'],
                          relative_gap_to_reference=(ref-r['obj'])/abs(ref))
            output.append(record)
    write_csv(Path(out)/'nm_polish_diagnostics.csv', output)
    return output


def run(rows, out):
    evidence = convexity_evidence()
    (Path(out)/'convexity_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return polish_suboptimal_nm(rows, out)
