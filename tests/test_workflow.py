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

    def test_nonfinite_is_not_feasible(self):
        self.assertEqual(O.constraint_violation(np.ones(4),[np.nan]*4),float('inf'))

    def test_summary_does_not_select_failed_solver(self):
        base=dict(N=1,method='SLSQP',feasible=True,success=False,obj=999.,model_calls=1,time=.1)
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
        self.assertGreater(r['obj'],1.9)


if __name__=='__main__':
    unittest.main()
