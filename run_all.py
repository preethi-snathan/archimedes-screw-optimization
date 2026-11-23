"""One command to reproduce everything:  python run_all.py"""
import validate_nu, experiments, plots

if __name__ == "__main__":
    print("=== 1. Validation against Rorres (2000) ===")
    v = validate_nu.nu(8, 0.5, 3 / 8)
    print(f"nu(8,1/2,3/8) = {v:.5f} (reference 0.1703)")
    print("\n=== 2. Experiments ===")
    rows = experiments.run()
    experiments.summarise(rows)
    print("\n=== 3. Figures ===")
    rows, hist = plots.load()
    plots.fig_nu_surface(); plots.fig_best_vs_N(rows); plots.fig_spread(rows); plots.fig_convergence(hist)
    print("Done. See results/ and figures/.")
