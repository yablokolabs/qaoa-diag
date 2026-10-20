"""The zero that started this library: how initialization decides the result.

Two traps, both common:

1. Exactly zero angles are a stationary point of the QAOA energy. The state is
   |+>^n and every derivative vanishes, so gradient descent never moves and
   reports "converged" at ratio 0.0.
2. Tiny angles are nearly stationary. A step size suited to angles of order one
   crawls there, so a fixed budget stops far from anywhere useful. Adam escapes
   because it normalises the gradient; plain gradient descent does not.

Run: python examples/zero_init_trap.py
"""

import numpy as np

from qaoa_diag import diagnose, multi_start, vertex_cover
from qaoa_diag.optimize import energy_function, minimize

edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
depth = 4
problem = vertex_cover(edges)
f = energy_function(problem, depth)

_, ceiling = multi_start(problem, depth, starts=10, steps=150, seed=0)

print(problem)
print(f"energy of |+>^n: {problem.diag.mean():.6f}"
      f"  (ratio {problem.ratio(problem.diag.mean()):.6f})")
print(f"depth-{depth} ceiling: {ceiling:.6f}  (ratio {problem.ratio(ceiling):.6f})\n")

print(f"{'init':18s} {'optimizer':10s} {'energy':>10s} {'ratio':>8s}  verdict")
for label, scale in (("exactly zero", 0.0), ("uniform(0, 0.05)", 0.05), ("uniform(0, 2*pi)", 2 * np.pi)):
    if scale == 0.0:
        start = np.zeros((2, depth))
    else:
        start = np.random.default_rng(7).uniform(0, scale, (2, depth))
    for method in ("gd", "adam"):
        params, _ = minimize(f, start, steps=150, lr=0.02, method=method)
        d = diagnose(params, problem, depth, ceiling=ceiling)
        print(f"{label:18s} {method:10s} {d.energy:10.6f} {d.ratio:8.6f}  {d.verdict}")

print(
    "\nGradient descent is frozen on the first row and slow on the second."
    "\nSpread the angles over (0, 2*pi) or normalise the gradient, then read the"
    "\nverdict before you trust the ratio."
)
