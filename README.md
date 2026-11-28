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
python3 run_comparisons.py                  # full analysis study in separate directories
python3 plots.py                            # regenerate figures from saved data
python3 comparison_plots.py                 # regenerate comparison PNG/PDF figures
```

Comparison methods are not part of the default `run_all.py` SLSQP study.
Use `run_comparisons.py` for the completed three-method analysis described below.
The older `run_all.py --comparisons` shortcut uses the original NM penalty of 1e5
and overwrites the baseline directory; it is not the final comparison workflow.
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

## Complete analysis study (Person 3 code)

```bash
python3 -m unittest discover -s tests -v
python3 run_comparisons.py
```

This command preserves the original `results/` and `figures/` files and writes to
`results/comparison/` and `figures/comparison/`. It reproduces validation, runs
172 design solves, performs numerical and penalty sensitivity studies, checks
all final designs on a finer integration grid, and produces six figures in both
200 dpi PNG and vector PDF formats. Additional review checks independently
integrate 25 geometries, solve four reduced problems and run 84 pitch-bound
sensitivity solves. Running it again overwrites its own outputs.
The test that deliberately corrupts a benchmark prints `Validation: FAIL`; that
message is expected only inside that regression test. The study itself must pass
validation before proceeding.

### Methods and experimental design

| Method | Constraint handling | Settings and repetitions |
| --- | --- | --- |
| SLSQP | Direct nonlinear inequalities and bounds | eps=1e-6, ftol=1e-10, maxiter=300; 20 starts per N |
| Nelder-Mead | Bounds plus `-VT + mu*sum(min(g,0)^2)` | mu=1e7, xatol=1e-7, fatol=1e-10, maxiter=4000, maxfev=6000; same 20 starts per N |
| Differential evolution | Direct nonlinear constraints and bounds | best1bin, population multiplier 15 (60 members), maxiter=150, tol=1e-8, no polishing; seeds 0, 1, 2 per N |

SLSQP is the primary solver because it directly handles the four coupled
inequalities and bounds in each continuous subproblem. Finite differences avoid
deriving gradients through the numerical geometry integration; the step/grid
study checks their numerical stability. Nelder-Mead supplies a derivative-free
comparison related to the method used in the original two-variable paper, but
the added penalty is specific to this constrained extension. Differential
evolution checks agreement using an independently initialized population and
no SLSQP polishing. These are separate solves of the same design problem, not
three stages through which each candidate is passed.

SLSQP and Nelder-Mead use exactly paired feasible starting points from a seeded
rejection sampler (seed 0). DE uses its own Latin-hypercube population, including
infeasible candidates. Repetitions therefore measure starting-point sensitivity
for the first two methods and population-seed sensitivity for DE. Starting-point
generation is excluded from timing. The comparison uses the recorded termination
settings, not identical evaluation budgets or identical convergence tolerances;
the measured efficiency does not establish a universal ranking of algorithms.

The main NM coefficient is fixed at 1e7 following pilot checks. The separate
36-run penalty sweep uses 1e5, 1e7 and 1e9 from the first three paired starts per N.
It retains all outcomes rather than selecting a coefficient separately for each
run. A finite quadratic penalty permits small constraint violations: tiny NM
volumes above the SLSQP value should not be described as superior designs.

### Procedure and files for the report

1. Validate nu and the two-variable SLSQP solutions against Rorres Table 1.
2. Enumerate N=1,2,3,4; draw paired feasible starts and solve with SLSQP and NM;
   run three independent DE populations for each N.
3. Save every outcome, including failures, feasibility, raw residuals, design
   variables, time, iterations, objective calls and total volume-model calls.
4. Select only converged, feasible final results for the best-design tables;
   distinguish solver acceptance from reaching the best observed value.
5. Check FD/grid sensitivity, NM penalty sensitivity, convergence histories and
   final 8001-point reevaluations. Investigate poor NM endpoints separately.
6. Generate figures and interpret the results with the limitations below.

The comparison directory contains all standard output files listed earlier, plus:

| File | Evidence supplied |
| --- | --- |
| history_model_calls.json | Actual cumulative integration counts at each history sample; includes constraint evaluations between objective calls |
| solution_consistency.csv | Counts within 1e-5 relative of the best SLSQP value, and worst/best accepted values |
| penalty_sensitivity.csv | All 36 NM penalty experiments, including infeasible results |
| final_grid_checks.csv | All 172 final volumes/feasibility flags reevaluated with 8001 points |
| convexity_evidence.json | Exact endpoints, midpoint, residuals and objective values for nonconvexity counterexamples |
| nm_polish_diagnostics.csv | Separate SLSQP restarts from NM endpoints more than 1e-5 relative below the reference |
| independent_integration.csv | 25 comparisons against nested adaptive quadrature, including both crest branches and their junction |
| reduced_formulation.csv | Four reduced two-variable DE checks with radius and inclination fixed at their favored boundaries |
| pitch_bound_sensitivity.csv | 84 SLSQP reruns with the extra 6 m pitch cap relaxed to the geometric envelope |
| review_summary.json | Counts, tolerances/results and physical scope of the additional model checks |

`run_comparisons.py` orchestrates and audits the study; `analysis.py` supplies
convexity and endpoint diagnostics; `comparison_plots.py` creates the figures.
`optimize.py` provides solvers and evaluation accounting; `experiments.py` handles
starts, tables, metadata and sensitivity. `nu.py` and `constraints.py` remain the
shared model. All methods use that same model.
`review_checks.py` supplies the independent quadrature and formulation checks;
`python3 review_checks.py` reruns those checks using the saved main-study starts.

| Evaluation topic | Outputs to use |
| --- | --- |
| Solution quality and constraint activity | best_designs.csv, results.csv, designs_and_success.png |
| Convergence and computational efficiency | convergence.png, computational_cost.png, method_summary.csv |
| Effect of initial guesses | initial_guess_sensitivity.png, solution_consistency.csv |
| Constraint handling and numerical stability | penalty_sensitivity.csv, solver_sensitivity.csv, final_grid_checks.csv |
| Feasible region and convexity | feasible_region_slice.png, convexity_checks.png, convexity_evidence.json |
| Local/global and multimodality discussion | Cross-method agreement, nm_polish_diagnostics.csv and the interpretation below |

### Interpretation and limits

The reported primary design is the best SLSQP result, not the tiny penalty-enabled
NM overshoot. For N=4 it is approximately Ro=1 m, rho=0.535239, P=2.572285 m,
alpha=30.963757 degrees, with VT=2.155355 m³/rev and Q=1.131490 m³/s.
The screw length is 5.830952 m for the fixed 3 m lift. This is ideal displaced
water per revolution; it is not a measured hydraulic efficiency or a length
optimization. The radius upper bound and footprint constraint are active;
minimum flow is slack. The selected N=4 is best among the four enumerated counts,
not among arbitrarily large flight counts.

For each N, all 20 SLSQP starts and all three DE populations agree within 1e-5
relative. NM reaches this neighborhood in 19/20, 19/20, 17/20 and 16/20 runs for
N=1,2,3,4 respectively, despite every main run satisfying termination and
feasibility criteria. A solver's successful termination is not an optimality
certificate. The separate endpoint restarts test whether these poor NM outcomes
can be improved by SLSQP; they are not added to the main accuracy or cost results.
All nine restarts recover the SLSQP reference to within 3e-10 relative, so these
NM endpoints should not be presented as verified distinct local maxima.

The saved run used about 176, 557 and 10,539 volume-model calls per solve on
average for SLSQP, NM and DE respectively. Use `method_summary.csv` for the
runtime measurements from the latest run; times depend on machine load. All 36 FD/grid sensitivity runs
were converged and feasible; the largest volume change was 1.54e-8 m³/rev. All
172 final designs retained feasibility on the 8001-point grid, with maximum
volume change 1.39e-8 m³/rev. In the penalty sweep, 0/12 runs at mu=1e5 met the
common feasibility threshold, compared with 12/12 at each of 1e7 and 1e9.
These sensitivity experiments explain the chosen settings and belong in the
discussion, even though the main study's acceptance rate is 100%.

The minimization objective is nonconvex: along an affine design segment with
fixed rho, alpha and lambda (P proportional to Ro), `f=-VT=-k*Ro^3`, k>0, and
`f''=-6*k*Ro<0`. The saved witness has feasible endpoints and midpoint. The
continuous N=4 feasible region is also nonconvex: the saved second pair has
feasible endpoints but its midpoint fails minimum flow, verified with 8001-point
integration. The two examples use different segments. Discreteness of N is an
additional reason the full problem is not a continuous convex program.

Nonconvexity alone does not imply multimodality. The sampled landscape and finite
multi-start/DE runs support a consistently found best design but do not prove
unimodality, uniqueness, or global optimality. The geometry uses clipping and
piecewise crest formulas, so global smoothness should not be assumed from a
surface plot. Grid and step sensitivity support numerical stability near the
tested solutions; they do not establish differentiability everywhere. A contour
slice fixes Ro and rho and cannot display the entire feasible region.

### Formulation review and additional checks

Independent nested adaptive quadrature agrees with the production closed-form
radial integral plus Simpson quadrature to within 1.81e-8 in nu for 25 cases.
These cover N=1..4, the N=8 benchmark, both crest branches, their junction and
near-empty buckets. All satisfy `0 <= nu <= 1-rho^2`, the annular-volume bound.
Both integrations share the same ideal pool geometry; agreement tests its
numerical implementation, not the physical validity of the assumptions.

The extra pitch cap `P <= 6 m` is **not implied** by the proposal's normalized
pitch constraints. For example, N=4, Ro=1, rho=0.53524, P=6.5 m and the minimum
feasible inclination satisfy the original flow and footprint constraints.
The original constraints imply only the envelope
`P <= 2*pi*Ro_max*lambda_max/tan(alpha_min) = 10.471976 m`.
The report must list the 6 m cap as an extra modeling bound if it describes the
main implementation. Relaxing it to this envelope in 84 reruns (20 original
starts plus one beyond-cap start for each N) recovered the same best volume for
every run to within 1e-6 m³/rev. It does not drive the reported optimum.

There is also useful mathematical structure behind the boundary solution:

`VT = 2*pi^2*Ro^3*(lambda*nu(N,rho,lambda))/tan(alpha)`.

Because Ro <= 1 and tan(alpha) >= H/Lmax = 0.6, every feasible design obeys
`VT <= (2*pi^2/0.6)*(lambda*nu)` at its own rho and lambda. A separate DE search
over rho and lambda, fixing Ro=1 and alpha=atan(0.6), reproduces all four full
SLSQP optima within 5.1e-10 m³/rev; the resulting pitches and flows are feasible.
If the reduced maximum were globally certified, this bound would also certify
the full optimum. The finite DE calculation is a cross-check, not that proof.
Explain this structure in the report: the formulation has five design variables,
but the ideal-volume objective strongly favors boundary values for two of them.

### Context and engineering assumptions for the report

Ro/rho/alpha bounds, N=1..4, H=3 m, Qreq=0.1 m³/s, the 5 m footprint and the
minimum lambda=0.05 are case-study choices inherited from the proposal. They
should not be called universal engineering limits. The speed rule is an assumed
operating condition, not a sixth optimized variable. State uniform geometry,
negligible blade thickness/end effects, full inlet feeding and zero leakage.
The capacity being optimized is ideal volume per turn, not cost, pumping power,
energy efficiency or measured delivered flow.

Beyond Rorres, useful primary literature includes:

- [YoosefDoost and Lubitz (2021), Archimedes Screw Design: An Analytical Model for Rapid Estimation of Archimedes Screw Geometry](https://www.mdpi.com/1996-1073/14/22/7812).
  Provides geometric/site context, the lift-length relation and a recommended
  rotational-speed expression equivalent to 50/D^(2/3) rpm. Its application is
  primarily screw generators; do not transfer its performance conclusions to
  pumps without justification.
- [Experimental Investigation of Archimedes Screw Pump](https://ascelibrary.com/doi/10.1061/%28ASCE%29HY.1943-7900.0001786).
  Its laboratory investigation varies inlet water level and rotation speed,
  providing context for the limitations of a fully fed geometric capacity model.

The full study supplies evidence for the numerical rubric, but the report still
needs the derivation, literature synthesis, physical justification, interpretation
of figures, conclusions and honest limitations. No test suite determines a grade.

Suggested future work includes a certified bound/global search, more physically
supported size and speed limits, leakage and hydraulic losses, blade thickness,
manufacturing costs or a volume/cost tradeoff, and sensitivity to lift height and
required flow. These change the engineering problem and are not claimed as
implemented analyses.

## Ownership and remaining handoff

- Model lead: inspect geometric derivation, assumptions and engineering bounds.
- Optimization lead: validation tables, SLSQP settings, design results, diagnostics.
- Analysis code and figures: complete through `run_comparisons.py`; use the saved
  evidence to write the discussion, conclusions and future-work sections.
- Whole group: finish and peer-review the report (main body less than ten pages),
  references and required disclosure, code appendix, five-slide deck and three-minute
  rehearsal. Numerical validation does not replace the model lead's engineering
  justification of assumptions, bounds and the speed/flow relation.
