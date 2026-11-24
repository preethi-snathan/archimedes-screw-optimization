"""Run validation, SLSQP experiments, sensitivity and figures from the repo root."""
import argparse
from pathlib import Path
import validate_nu
import experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--comparisons', action='store_true',
                        help='Also run penalised Nelder-Mead and differential evolution')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    validate_nu.run()  # Raises on failure: never proceed with an invalid model.
    rows = experiments.run(n_starts=args.starts, seed=args.seed, comparisons=args.comparisons)
    experiments.sensitivity(rows)
    if not args.no_plots:
        import plots
        Path('figures').mkdir(exist_ok=True)
        rows, hist = plots.load()
        plots.fig_nu_surface()
        plots.fig_best_vs_N(rows)
        plots.fig_spread(rows)
        plots.fig_convergence(hist)
    print('Done. See results/ for validation, designs, diagnostics and sensitivity.')


if __name__ == '__main__':
    main()
