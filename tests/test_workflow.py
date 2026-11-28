"""Regression checks for scientific reporting and solver plumbing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import constraints as C
import optimize as O
import validate_nu as V
import experiments as E
import analysis as A
import run_comparisons as R
import json
from nu import nu
from review_checks import adaptive_nu


class WorkflowTests(unittest.TestCase):
    def test_sensitivity_diagnostics_use_solver_volume(self):
        x=np.array([1., .535, 2.5, 31.])
        volume=0.75
        with patch.object(C, 'volume_per_turn', side_effect=AssertionError('unexpected integration')):
            details=E.design_details(x,4,volume=volume)
        self.assertAlmostEqual(details['Q'],volume*C.rpm_of(x[0])/60.)
        self.assertAlmostEqual(details['g1'],details['Q']/C.Q_REQ-1.)

    def test_empty_output_replaces_stale_designs(self):
        with tempfile.TemporaryDirectory() as out:
            path=Path(out)/'best_designs.csv'
            V.write_csv(path,[dict(N=4,obj=2.15)])
            V.write_csv(path,[])
            self.assertEqual(path.read_text(),'')

    def test_constraint_reuse_preserves_physical_model(self):
        x=np.array([1., .535, 2.5, 31.])
        np.testing.assert_allclose(C.g_vector(x,4), C.g_vector(x,4,volume=C.volume_per_turn(x,4)))

    def test_history_excludes_infeasible_trials_and_penalty(self):
        ev=O.Evaluator(3)
        infeasible=np.array([1.,.535,2.3,20.])
        feasible=np.array([1.,.535,2.3,32.])
        ev.objective(infeasible,penalty=1e5)
        self.assertIsNone(ev.history[-1])
        ev.objective(feasible,penalty=1e5)
        expected=-C.volume_per_turn(feasible,3)
        self.assertAlmostEqual(ev.history[-1],expected)
        ev.objective(infeasible)
        self.assertAlmostEqual(ev.history[-1],expected)
        self.assertEqual(ev.model_calls,3)
        ev.constraints(feasible)
        self.assertEqual(ev.model_calls,4)
        ev.objective(feasible)
        self.assertEqual(ev.history_model_calls,[1,2,3,5])

    def test_nonfinite_is_not_feasible(self):
        self.assertEqual(O.constraint_violation(np.ones(4),[np.nan]*4),float('inf'))
        self.assertEqual(C.max_violation([np.nan,.5,2.,32.],4),float('inf'))
        with patch.object(C,'g_vector',return_value=np.array([np.nan]*4)):
            self.assertEqual(C.max_violation([1.,.5,2.,32.],4),float('inf'))

    def test_invalid_quadrature_inputs_fail_explicitly(self):
        for n_phi in (0,1,2,2000,2001.5):
            with self.assertRaises(ValueError):
                nu(4,.5,.25,n_phi=n_phi)
        for N in (0,-1,1.5,True):
            with self.assertRaises(ValueError):
                nu(N,.5,.25)
        with self.assertRaises(ValueError):
            nu(4,.5,np.nan)

    def test_independent_quadrature_checks_both_crest_branches(self):
        for N,rho,lam in ((8,.5,.375),(4,.2,.6)):
            reference,_=adaptive_nu(N,rho,lam)
            self.assertLess(abs(nu(N,rho,lam)-reference),1e-7)

    def test_summary_does_not_select_failed_solver(self):
        base=dict(N=1,method='SLSQP',feasible=True,success=False,obj=999.,model_calls=1,nfev=1,nit=1,time=.1,violation=0.)
        summaries,best=E.summarise([base],verbose=False)
        self.assertEqual(best,[])
        self.assertEqual(summaries[0]['successful_feasible_runs'],0)
        self.assertEqual(summaries[0]['best_feasible_VT'],999.)

    def test_corrupted_benchmark_fails_validation(self):
        with tempfile.TemporaryDirectory() as out, patch.object(V,'REF_NU',.5):
            with self.assertRaisesRegex(RuntimeError,'benchmark'):
                V.run(out=out,n_starts=1)

    def test_slsqp_returns_feasible_diagnostics(self):
        r=O.solve_slsqp(3,[.8,.5,1.8,35.])
        self.assertTrue(r['success'])
        self.assertTrue(r['feasible'])
        self.assertGreater(r['model_calls'],r['nfev'])
        self.assertEqual(len(r['history']),r['nfev'])
        self.assertEqual(len(r['history_model_calls']),r['nfev'])
        self.assertEqual(r['model_calls'],r['nfev']+r['ncev']+1)
        self.assertGreater(r['obj'],1.9)

    def test_convexity_witnesses_survive_refinement(self):
        evidence=A.convexity_evidence()
        for name in ('objective','feasible_region'):
            points=evidence[name]['points']
            np.testing.assert_allclose(points[1]['x'],
                (np.array(points[0]['x'])+points[2]['x'])/2)
            for p in points:
                coarse=A.evaluate_point(p['x'],4,n_phi=2001)
                self.assertAlmostEqual(coarse['VT'],p['VT'],places=6)
        self.assertGreater(evidence['objective']['midpoint_Jensen_gap'],.01)
        self.assertLess(evidence['feasible_region']['points'][1]['residuals'][0],-.2)

    def test_history_audit_rejects_misleading_convergence(self):
        row=dict(method='SLSQP',N=1,run=0,nfev=3,model_calls=8)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'histories.json').write_text(json.dumps({'SLSQP_N1_run0':[None,-1.,-.5]}))
            (out/'history_model_calls.json').write_text(json.dumps({'SLSQP_N1_run0':[1,3,7]}))
            with self.assertRaisesRegex(RuntimeError,'nonmonotone'):
                R.audit_histories([row],out)
            (out/'histories.json').write_text(json.dumps({'SLSQP_N1_run0':[None,-1.,-1.5]}))
            R.audit_histories([row],out)

    def test_de_limit_keeps_failure_status_and_counts(self):
        r=O.solve_de(4,seed=0,popsize=5,maxiter=1)
        self.assertFalse(r['success'])
        self.assertEqual(r['model_calls'],r['nfev']+r['ncev']+1)
        self.assertEqual(len(r['history_model_calls']),r['nfev'])


if __name__=='__main__':
    unittest.main()
