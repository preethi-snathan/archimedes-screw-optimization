# Archimedes screw optimization (SYDE 411)

Run order (each file depends only on the ones above it):
1. `nu.py`           volumetric capacity factor nu(N, rho, lam)
2. `constraints.py`  5-variable problem: objective, constraints g(x) >= 0, bounds
3. `optimize.py`     SLSQP, penalised Nelder-Mead, differential evolution, 2-variable Rorres solver
4. `validate_nu.py`  checks nu(8,1/2,3/8) ~ 0.1703 and rho* ~ 0.54
5. `experiments.py`  multi-start study for N = 1..4 -> results/
6. `plots.py`        figures/
7. `run_all.py`      does 4-6 in one go:  python run_all.py

Requirements: numpy, scipy, matplotlib.

Variables x = [Ro (m), rho, P (m), alpha (deg)]; N is enumerated 1..4.
lam = P*tan(alpha)/(2*pi*Ro);  VT = nu * pi * Ro^2 * P;  Q = VT * rpm/60;  rpm = 50/(2Ro)^(2/3).

Still to do by the group: copy Table 1 of Rorres (2000) into validate_nu.py and compare lam*.
