"""How many optimizer steps do you actually need?

Sweep the budget from one fixed start and compare against the depth ceiling. The
ratio climbs, then flattens below the ceiling: one start is stuck in its own
basin, and no extra step gets out. Budget and restarts fix different problems.

Run: python examples/budget_sweep.py
"""

from qaoa_diag import budget_sweep, multi_start, vertex_cover

edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
depth = 4
problem = vertex_cover(edges)

_, ceiling = multi_start(problem, depth, starts=16, steps=300)
ceiling_ratio = problem.ratio(ceiling)

print(problem)
print(f"depth-{depth} ceiling: ratio {ceiling_ratio:.4f}  (gap {ceiling - problem.ground:.4f})\n")

print("steps  ratio     share of ceiling")
for row in budget_sweep(problem, depth, budgets=(0, 5, 10, 25, 50, 100, 200, 400, 800)):
    share = row["ratio"] / ceiling_ratio if ceiling_ratio else 0.0
    print(f"{row['steps']:5d}  {row['ratio']:.6f}  {share:6.2%}")

print(
    "\nThe last row barely moves, yet it sits below the multi-start ceiling."
    "\nThat gap is a local minimum, and no extra step closes it: restart instead."
)
