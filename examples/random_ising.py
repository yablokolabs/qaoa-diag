"""Frustrated random Ising model: where depth stops helping.

Random couplings on a ring of eight spins. The ground energy is exact and cheap,
so the table shows the real cost of a shallow circuit on a frustrated problem.

Run: python examples/random_ising.py
"""

import numpy as np

from qaoa_diag import diagnose, multi_start
from qaoa_diag.problem import from_spin_terms

n = 8
rng = np.random.default_rng(3)
pairs = [(i, (i + 1) % n, float(rng.choice([-1.0, 1.0]))) for i in range(n)]
problem = from_spin_terms(n, zpairs=pairs, name="random-ising")

print(problem)
print("couplings:", ", ".join(f"J{i}{j}={c:+.0f}" for i, j, c in pairs))
print(f"exact ground energy {problem.ground:.4f}  (found by diagonalising 2^{n} states)\n")

print("depth  ceiling   ratio    gap     verdict")
for depth in (1, 2, 3, 4, 6):
    params, ceiling = multi_start(problem, depth, starts=8, steps=150, seed=depth)
    d = diagnose(params, problem, depth, ceiling=ceiling)
    print(f"  {depth}    {d.ceiling:8.4f}  {d.ratio:.4f}  {d.ceiling - problem.ground:7.4f}  {d.verdict}")
