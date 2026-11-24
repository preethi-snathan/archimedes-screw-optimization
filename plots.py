"""plots.py -- reads results/ and writes figures to figures/."""
import json, os
from pathlib import Path
# Keep plotting caches in the project when the user has not configured them.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".mplconfig" / "cache"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from nu import nu


METHODS = ["SLSQP", "NelderMead", "DiffEvolution"]


def load():
    with open("results/results.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k in ("N", "obj", "violation", "time", "nfev"):
                r[k] = float(r[k])
            elif k in ("success", "feasible"):
                r[k] = r[k] == "True"
        r["N"] = int(r["N"])
    with open("results/histories.json") as f:
        hist = json.load(f)
    return rows, hist


def fig_nu_surface():
    rho = np.linspace(0.2, 0.7, 60); lam = np.linspace(0.03, 0.6, 60)
    Z = np.array([[lam_i * nu(3, r, lam_i) for r in rho] for lam_i in lam])
    plt.figure(figsize=(5.5, 4))
    plt.contourf(rho, lam, Z, 25); plt.colorbar(label=r"$\lambda\,\nu$ (N=3)")
    plt.xlabel(r"$\rho$"); plt.ylabel(r"$\lambda$"); plt.title("Two-variable objective landscape")
    plt.tight_layout(); plt.savefig("figures/landscape_N3.png", dpi=200); plt.close()


def fig_best_vs_N(rows):
    plt.figure(figsize=(5.5, 4))
    methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
    for i, m in enumerate(methods):
        vals = []
        for N in (1, 2, 3, 4):
            o = [r["obj"] for r in rows if r["N"] == N and r["method"] == m and r["feasible"] and r["success"]]
            vals.append(max(o) if o else np.nan)
        plt.bar(np.arange(1, 5) + (i - (len(methods)-1)/2) * 0.27, vals, width=0.27, label=m)
    plt.xlabel("Number of blades N"); plt.ylabel("Best water per turn VT [m$^3$]")
    plt.legend(); plt.tight_layout(); plt.savefig("figures/best_vs_N.png", dpi=200); plt.close()


def fig_spread(rows):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True)
    for ax, N in zip(axs, (1, 2, 3, 4)):
        data = []
        methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
        for m in methods:
            data.append([r["obj"] for r in rows if r["N"] == N and r["method"] == m and r["feasible"] and r["success"]])
        ax.boxplot([d if d else [np.nan] for d in data])
        ax.set_xticks(range(1, len(methods)+1)); ax.set_xticklabels(methods)
        ax.set_title(f"N={N}")
    axs[0].set_ylabel("Final VT [m$^3$]")
    fig.suptitle("Variation across starts: converged, feasible runs")
    plt.tight_layout(); plt.savefig("figures/spread_over_starts.png", dpi=200); plt.close()


def fig_convergence(hist):
    plt.figure(figsize=(5.5, 4))
    for m in METHODS:
        h = hist.get(f"{m}_N3")
        if h:
            plt.plot(np.arange(1, len(h) + 1), -np.array([np.nan if v is None else v for v in h], dtype=float), label=m)
    plt.xscale("log"); plt.xlabel("Function evaluations"); plt.ylabel("Best feasible VT so far [m$^3$] (N=3)")
    plt.legend(); plt.tight_layout(); plt.savefig("figures/convergence_N3.png", dpi=200); plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    rows, hist = load()
    fig_nu_surface(); fig_best_vs_N(rows); fig_spread(rows); fig_convergence(hist)
    print("figures written to figures/")
