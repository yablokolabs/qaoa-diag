"""Case study: the QAOA run whose ratio was *not* the optimum.

This reproduces the situation that motivated the library. A minimum-vertex-cover
QAOA task was graded against reference values, and the better answer was marked
wrong: the reference had stopped short of the ceiling. Two measurements tell the
two cases apart, the depth ceiling and the gradient norm at the final angles.

Run: python examples/vertex_cover_trap.py
"""

import numpy as np

from qaoa_diag import budget_sweep, depth_frontier, diagnose, multi_start, vertex_cover
from qaoa_diag.optimize import energy_function, minimize

GRAPHS = {
    "triangle+tail": [(0, 1), (1, 2), (0, 2), (2, 3)],
    "4-cycle": [(0, 1), (1, 2), (2, 3), (3, 0)],
}
DEPTH = 4


def main() -> None:
    for name, edges in GRAPHS.items():
        problem = vertex_cover(edges)
        print(f"=== {name} {edges} ===")
        print(problem)
        print(f"exact minimum vertex cover: {sorted(problem.best_bits()[0].nonzero()[0].tolist())}"
              f"  (energy {problem.ground:.4f})\n")

        print("depth ceiling:")
        frontier = depth_frontier(problem, depths=(1, 2, 3, 4), starts=10, steps=150)
        for depth, row in frontier.items():
            print(f"  p={depth}  ratio {row['ratio']:.6f}  gap {row['gap']:.6f}")

        _, ceiling = multi_start(problem, DEPTH, starts=10, steps=150, seed=0)

        zero = diagnose(np.zeros((2, DEPTH)), problem, DEPTH, ceiling=ceiling)
        print(f"\nzero init      ratio {zero.ratio:.6f}  |grad| {zero.grad_norm:.1e}  -> {zero.verdict}")

        f = energy_function(problem, DEPTH)
        start = np.random.default_rng(13).uniform(0, 2 * np.pi, (2, DEPTH))
        params, _ = minimize(f, start, steps=50, lr=0.02)
        modest = diagnose(params, problem, DEPTH, ceiling=ceiling)
        print(f"50-step start  ratio {modest.ratio:.6f}  |grad| {modest.grad_norm:.1e}  -> {modest.verdict}")

        best, full_ceiling = multi_start(problem, DEPTH, starts=10, steps=150, seed=0)
        full = diagnose(best, problem, DEPTH, ceiling=full_ceiling)
        print(f"best of 10     ratio {full.ratio:.6f}  -> {full.verdict}")

        print("\nbudget sweep from the 50-step start:")
        for row in budget_sweep(problem, DEPTH, budgets=(0, 25, 50, 100, 200, 400)):
            print(f"  steps {row['steps']:>4}  ratio {row['ratio']:.6f}")
        print()


if __name__ == "__main__":
    main()
