"""Quickstart: build a problem, run QAOA, ask why it fell short.

Run: python examples/quickstart.py
"""

import numpy as np

from qaoa_diag import diagnose, energy_function, minimize, vertex_cover

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
depth = 4

problem = vertex_cover(edges)
print(problem)

params0 = np.random.default_rng(0).uniform(0, 2 * np.pi, (2, depth))
params, value = minimize(energy_function(problem, depth), params0, steps=300, lr=0.05)

print(f"\noptimized energy {value:.6f}")
print(diagnose(params, problem, depth, starts=10, steps=150))
