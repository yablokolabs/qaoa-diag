"""Max-Cut on a 5-cycle: a frustrated instance, so the ceiling is visible.

An even ring is bipartite and QAOA solves it outright, which makes for a dull
table. An odd ring cannot be cut edge-for-edge, so the optimum is 4 of 5 edges
and the depth ceiling shows how fast the circuit gets there.

Run: python examples/max_cut_frontier.py
"""

from qaoa_diag import diagnose, max_cut, multi_start

N = 5
EDGES = [(i, (i + 1) % N) for i in range(N)]
DEPTHS = (1, 2, 3, 4, 5)


def main() -> None:
    problem = max_cut(EDGES)
    print(f"max-cut on a {N}-ring (odd, frustrated), {len(EDGES)} edges")
    print(problem)
    print(f"exact optimum: {problem.ground:.4f} = {-problem.ground:.0f} of {len(EDGES)} edges cut\n")

    print("depth  ceiling   cuts   ratio    gap     verdict")
    for depth in DEPTHS:
        params, ceiling = multi_start(problem, depth, starts=10, steps=150, seed=depth)
        d = diagnose(params, problem, depth, ceiling=ceiling)
        print(
            f"  {depth}    {d.ceiling:7.4f}  {-d.ceiling:4.1f}   {d.ratio:.4f}  "
            f"{d.ceiling - problem.ground:6.4f}  {d.verdict}"
        )

    print(
        "\nRatio 1.000 means the circuit found the best possible cut at that depth."
        "\nA gap that stops shrinking says the depth, not the optimizer, is the limit."
    )


if __name__ == "__main__":
    main()
