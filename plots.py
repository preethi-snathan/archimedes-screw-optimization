"""plots.py -- reads results/ and writes figures to figures/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from nu import nu

os.makedirs("figures", exist_ok=True)
METHODS = ["SLSQP", "NelderMead", "DiffEvolution"]


def load():
    with open("results/results.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k not in ("method",):
                r[k] = float(r[k]) if k != "success" else r[k]
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
    for i, m in enumerate(METHODS):
        vals = []
        for N in (1, 2, 3, 4):
            o = [r["obj"] for r in rows if r["N"] == N and r["method"] == m and r["violation"] < 1e-4]
            vals.append(max(o) if o else np.nan)
        plt.bar(np.arange(1, 5) + (i - 1) * 0.27, vals, width=0.27, label=m)
    plt.xlabel("Number of blades N"); plt.ylabel("Best water per turn VT [m$^3$]")
    plt.legend(); plt.tight_layout(); plt.savefig("figures/best_vs_N.png", dpi=200); plt.close()


def fig_spread(rows):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True)
    for ax, N in zip(axs, (1, 2, 3, 4)):
        data = []
        for m in METHODS:
            data.append([r["obj"] for r in rows if r["N"] == N and r["method"] == m and r["violation"] < 1e-4])
        ax.boxplot([d if d else [np.nan] for d in data])
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["SLSQP", "NM", "DE"])
        ax.set_title(f"N={N}")
    axs[0].set_ylabel("Final VT over starts [m$^3$] (feasible runs)")
    plt.tight_layout(); plt.savefig("figures/spread_over_starts.png", dpi=200); plt.close()


def fig_convergence(hist):
    plt.figure(figsize=(5.5, 4))
    for m in METHODS:
        h = hist.get(f"{m}_N3")
        if h:
            plt.plot(np.arange(1, len(h) + 1), -np.array(h), label=m)
    plt.xscale("log"); plt.xlabel("Function evaluations"); plt.ylabel("Best VT so far [m$^3$] (N=3)")
    plt.legend(); plt.tight_layout(); plt.savefig("figures/convergence_N3.png", dpi=200); plt.close()


if __name__ == "__main__":
    rows, hist = load()
    fig_nu_surface(); fig_best_vs_N(rows); fig_spread(rows); fig_convergence(hist)
    print("figures written to figures/")
