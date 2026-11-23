"""Validation of nu and of the two-variable optimum against Rorres (2000)."""
import numpy as np
from nu import nu
import optimize as O

REF_NU = 0.1703   # Rorres (2000): nu(8, 1/2, 3/8)
REF_RHO = 0.54    # Rorres (2000): optimal rho ~ 0.54

if __name__ == "__main__":
    v = nu(8, 0.5, 3 / 8)
    print(f"nu(8, 1/2, 3/8) = {v:.5f}   reference {REF_NU}   abs diff {abs(v-REF_NU):.2e}")
    print("\nTwo-variable optimum (maximise lam*nu), SLSQP, best of 5 starts:")
    print(" N   rho*     lam*     lam*nu    nfev")
    rng = np.random.default_rng(1)
    for N in range(1, 9):
        runs = [O.solve_two_var(N, [rng.uniform(.3, .65), rng.uniform(.1, .4)]) for _ in range(5)]
        b = max(runs, key=lambda r: r["obj"])
        print(f" {N}  {b['x'][0]:.4f}  {b['x'][1]:.4f}  {b['obj']:.5f}  {b['nfev']}")
    print(f"\nRorres reports rho* ~ {REF_RHO}. Compare lam* with Table 1 of the paper "
          "(not reproduced here -- copy the published values into this script).")
