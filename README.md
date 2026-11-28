# Archimedes Screw Optimization — SYDE 411

Optimize the ideal water volume transported per revolution by an Archimedes
screw. The model varies outer radius, radius ratio, pitch and inclination for
flight counts `N = 1, 2, 3, 4`, subject to flow, footprint and geometric constraints.

The project validates the model against Rorres (2000), then uses multi-start
SLSQP with Nelder–Mead and differential evolution comparisons. The analysis
covers convergence, computational cost, initial guesses, constraint satisfaction
and numerical sensitivity. Results describe ideal capacity; finite searches do
not prove global optimality.

## Repository layout

| File or folder | Purpose |
| --- | --- |
| `nu.py`, `constraints.py` | Water-volume model, design bounds and constraints |
| `optimize.py`, `experiments.py` | Solvers, seeded experiments and result summaries |
| `validate_nu.py` | Published benchmarks and integration checks |
| `analysis.py`, `review_checks.py` | Nonconvexity, solver diagnostics and independent numerical checks |
| `run_all.py` | Original SLSQP study |
| `run_comparisons.py` | Complete three-method analysis |
| `plots.py`, `comparison_plots.py` | Generate figures from the model and saved results |
| `tests/` | Regression tests |
| `results/`, `figures/` | Original SLSQP outputs |
| `results/comparison/`, `figures/comparison/` | Full analysis tables, diagnostics and PNG/PDF figures |
| `requirements.txt` | Python dependencies |

## Setup and usage

Use Python 3.10 or newer. Run all commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

Run the tests and complete analysis:

```bash
python3 -m unittest discover -s tests -v
python3 run_comparisons.py
```

The full analysis validates the model, compares all three solvers, runs
sensitivity and verification checks, and generates six figures. Outputs go to
`results/comparison/` and `figures/comparison/`; rerunning replaces those outputs.
Start with `best_designs.csv` for selected designs, `method_summary.csv` for
performance, and `results.csv` for every main solver run.

Other useful commands:

```bash
python3 validate_nu.py                  # Benchmark validation only
python3 run_all.py                      # Original SLSQP study
python3 run_all.py --starts 5 --no-plots # Fewer design starts, without plots
python3 plots.py                        # Original-study figures
python3 comparison_plots.py             # Comparison figures from saved outputs
python3 review_checks.py                # Extra checks using saved comparison runs
```

Validation and the original study write to `results/`; original figures go to
`figures/`. Existing saved outputs are included, so plotting and review commands
can use them without rerunning the full optimization. One regression test
intentionally prints `Validation: FAIL`; that is expected inside the test, while
standalone validation and the study must pass.
