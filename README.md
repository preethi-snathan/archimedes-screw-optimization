# Archimedes screw optimization (SYDE 411)

Maximise ideal water volume transported per revolution. Four continuous variables
`x = [Ro (m), rho, P (m), alpha (deg)]` are optimized for each discrete flight count
`N = 1, 2, 3, 4`. The geometry and assumptions in `nu.py` remain the model lead's
responsibility; this workflow validates agreement with published numerical results.

## Reproduce the optimization lead's results

From the repository root, with Python 3.10 or newer:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 run_all.py
```

The full command first validates the benchmark and Table 1 for N=1..8 (eight
starts each), then runs 20 feasible random starts per N with seed 0, followed by
36 integration-resolution / finite-difference sensitivity runs and four figures.
A failed validation raises an error and stops before design experiments.
Existing results and figures are overwritten. No report text is generated.

Other commands:

```bash
python3 validate_nu.py                       # validation only
python3 run_all.py --starts 5 --no-plots      # smaller check
python3 run_all.py --comparisons             # adds NM and three DE seeds per N
python3 plots.py                            # regenerate figures from saved data
```

Optional comparison methods are not part of the default SLSQP deliverable.
The experimental seed can be set with `--seed`; validation independently uses
seed 1. Exact runtimes vary with the machine. Package versions, code hashes,
Git state, settings and timestamp are recorded in `results/metadata.json`.
`requirements.txt` gives compatible version ranges, not a fully locked environment.

## Output files

| File in results/ | Contents |
| --- | --- |
| validation.csv | Published and computed rho, lambda, lambda*nu; absolute and objective-relative errors |
| validation_runs.csv | Every two-variable start and solver outcome |
| validation_summary.json | Benchmark, tolerances, source and pass/fail checks |
| integration_refinement.csv | 501/1001/2001/4001/8001-point evaluations at benchmark and fitted optima |
| results.csv | All design starts, final variables, status/message, iterations, evaluations, timing, flow and constraints |
| best_designs.csv | Best **converged and feasible** design per method and N, with 8001-point volume recheck |
| method_summary.csv | Separate feasible and converged-feasible counts; objective spread and cost |
| solver_sensitivity.csv | Each N rerun at 1001/2001/4001 points and FD steps 1e-5/1e-6/1e-7 |
| histories.json | Negative best feasible physical VT per objective evaluation; null before first feasible point |
| metadata.json | Environment, source fingerprints, settings and provenance |

## Formulation and validation

`lambda = P*tan(alpha)/(2*pi*Ro)`; `VT = nu*pi*Ro**2*P`;
`Q = VT*rpm/60`; `rpm = 50/(2*Ro)**(2/3)`.
Angles are converted from degrees before trigonometric evaluation.
The two-variable validation maximises `lambda*nu` over the unit square, excluding
endpoints by 1e-6. Starts are drawn from rho in [0.2,0.7], lambda in [0.05,0.4]
to avoid zero-capacity plateaus. This is not an exhaustive search.

Reference: C. Rorres (2000), *The Turn of the Screw: Optimal Design of an
Archimedes Screw*, Journal of Hydraulic Engineering 126(1), 72-80.
[Primary-source PDF, Table 1 on p. 76](https://www.cs.drexel.edu/~crorres/screw/screw.pdf).
DOI: 10.1061/(ASCE)0733-9429(2000)126:1(72).
The N=1..8 numerical references were transcribed from the indexed primary source.

The benchmark and objective tolerance is 5e-5 (half a four-decimal reporting unit).
The parameter agreement tolerance is 5e-4, allowing numerical-search differences
between the published Nelder-Mead calculation and SLSQP; it is not a claim of
four-decimal parameter identity. Integration refinement requires the difference
between 2001 and 8001 points to be at most 1e-7 in nu.

## Solver and interpretation details

SLSQP minimises negative VT using finite-difference gradients (`eps=1e-6` in the
original units), `ftol=1e-10`, and at most 300 iterations. Bounds and constraints
are in `constraints.py`. Feasibility requires maximum violation <=1e-6, using
the normalized nonlinear residuals and raw bound violations. Activity uses 1e-5.
CSV g1..g4 correspond to flow, upper lambda, lower lambda and footprint.
Raw Q and footprint values are also saved for physical interpretation.

Each objective evaluation integrates nu once. Constraint evaluations and final
solver diagnostics are also included in `model_calls`; `nfev` counts objective
calls only. Start generation and report postprocessing are excluded from both
solver timing and model-call counts. Convergence histories are evaluation-based,
not iteration-based, and exclude infeasible improvements and NM penalty terms.
Null history entries mean that no feasible objective trial has occurred yet.

Multi-start agreement and finite DE searches do not prove global optimality.
Feasible but unconverged runs remain visible in raw results and counts, but are
excluded from the selected-design table and final-objective plots.
Sensitivity reruns use the original start of the best baseline run for each N;
they are numerical stability checks, not additional independent multi-start studies.

Under the current idealized model, `VT = 2*pi**2*Ro**3*lambda*nu/tan(alpha)`.
Larger radius and smaller inclination are favored when pitch limits permit.
The footprint gives alpha >= arctan(3/5) = 30.9638 degrees. Expect radius and
footprint constraints to be active and minimum flow to be slack. Removing the
cost objective removes its size penalty. This behavior should be discussed in
the report; it is not automatically a solver error. The P upper bound of 6 m
is a numerical design bound in the current implementation, not a Rorres limit.

## Ownership and remaining handoff

- Model lead: inspect geometric derivation, assumptions and engineering bounds.
- Optimization lead: validation tables, SLSQP settings, design results, diagnostics.
- Analysis lead: extend comparisons, interpret convergence/performance and assemble figures.
