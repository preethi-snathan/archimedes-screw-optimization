"""
experiments.py -- multi-start / multi-method study for N = 1..4.
Writes results/results.csv and results/histories.json.
"""
import json, os, csv, warnings
import numpy as np
import constraints as C
import optimize as O

N_STARTS = 20
N_DE_SEEDS = 3
OUT = "results"


def run(n_starts=N_STARTS, n_de=N_DE_SEEDS, seed=0, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, hist = [], {}
    for N in (1, 2, 3, 4):
        starts = [O.feasible_start(N, rng) for _ in range(n_starts)]
        for i, x0 in enumerate(starts):
            for name, solver in (("SLSQP", O.solve_slsqp), ("NelderMead", O.solve_nelder_mead)):
                r = solver(N, x0)
                rows.append(dict(N=N, method=name, run=i, obj=r["obj"], nfev=r["nfev"],
                                 time=r["time"], violation=r["violation"], success=r["success"],
                                 **{k: v for k, v in zip(C.X_NAMES, r["x"])}))
                if i == 0:
                    hist[f"{name}_N{N}"] = r["history"]
        for s in range(n_de):
            r = O.solve_de(N, seed=s)
            rows.append(dict(N=N, method="DiffEvolution", run=s, obj=r["obj"], nfev=r["nfev"],
                             time=r["time"], violation=r["violation"], success=r["success"],
                             **{k: v for k, v in zip(C.X_NAMES, r["x"])}))
            if s == 0:
                hist[f"DiffEvolution_N{N}"] = r["history"]
        if verbose:
            print(f"N={N} done")
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(f"{OUT}/histories.json", "w") as f:
        json.dump(hist, f)
    return rows


def summarise(rows):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    """Print a method comparison table (feasible runs only count as 'ok')."""
    import collections
    print(f"{'N':>2} {'method':<14} {'best VT':>8} {'median VT':>9} {'std':>7} {'%feas':>6} {'mean nfev':>9} {'mean s':>7}")
    for N in (1, 2, 3, 4):
        for m in ("SLSQP", "NelderMead", "DiffEvolution"):
            rs = [r for r in rows if r["N"] == N and r["method"] == m]
            feas = [r for r in rs if r["violation"] < 1e-4]
            objs = np.array([r["obj"] for r in feas]) if feas else np.array([np.nan])
            print(f"{N:>2} {m:<14} {np.nanmax(objs):8.4f} {np.nanmedian(objs):9.4f} {np.nanstd(objs):7.4f} "
                  f"{100*len(feas)/len(rs):6.0f} {np.mean([r['nfev'] for r in rs]):9.0f} "
                  f"{np.mean([r['time'] for r in rs]):7.2f}")


if __name__ == "__main__":
    summarise(run())
