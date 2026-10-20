"""One seed is not a result: the spread of single-run ratios at one depth.

Forty independent single runs of depth-4 QAOA on the same graph report forty
different approximation ratios. The spread is optimizer variance. It says
nothing about what the circuit can reach.

Run: python examples/seed_study.py
"""

import numpy as np

from qaoa_diag import energy_function, minimize, multi_start, vertex_cover

EDGES = [(0, 1), (1, 2), (0, 2), (2, 3)]
DEPTH = 4
RUNS = 40


def main() -> None:
    problem = vertex_cover(EDGES)
    f = energy_function(problem, DEPTH)
    _, ceiling = multi_start(problem, DEPTH, starts=14, steps=200, seed=0)
    ceiling_ratio = problem.ratio(ceiling)

    energies = []
    for seed in range(RUNS):
        x0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, (2, DEPTH))
        _, value = minimize(f, x0, steps=50, lr=0.02)
        energies.append(value)

    ratios = np.array([problem.ratio(e) for e in energies])
    p10, p50, p90 = np.percentile(ratios, [10, 50, 90])
    near = int(np.sum(np.abs(np.array(energies) - ceiling) <= 1e-3))

    print(problem)
    print(f"depth-{DEPTH} ceiling (best of 14 starts): ratio {ceiling_ratio:.4f}")
    print(f"{RUNS} single runs, 50 Adam steps each, seeds 0..{RUNS - 1}\n")
    print(f"  min    {ratios.min():.4f}")
    print(f"  p10    {p10:.4f}")
    print(f"  median {p50:.4f}")
    print(f"  p90    {p90:.4f}")
    print(f"  max    {ratios.max():.4f}")
    print(f"  mean   {ratios.mean():.4f}")
    print(f"  within 1e-3 of the ceiling: {near} of {RUNS}\n")

    lo, hi = float(ratios.min()), float(ratios.max())
    edges = np.linspace(lo, hi, 11)
    counts = np.histogram(ratios, bins=edges)[0]
    print("ratio spread")
    for i, count in enumerate(counts):
        print(f"  {edges[i]:.3f} {count:2d} {'#' * int(count)}")

    print(
        f"\nA single run reports anything between {lo:.3f} and {hi:.3f} for this one"
        " problem at one depth, so report restarts, not one seed."
        "\nThat spread is optimizer variance: the ceiling never moved."
    )


if __name__ == "__main__":
    main()
