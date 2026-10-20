"""Emit a markdown table you can paste into a README, an issue or a PR.

Each row is one graph: the exact optimum, the ratio the depth-4 circuit reaches,
and the verdict on that run.

Run: python examples/markdown_report.py
"""

from qaoa_diag import diagnose, multi_start, vertex_cover

GRAPHS = {
    "path-4": [(0, 1), (1, 2), (2, 3)],
    "cycle-4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "triangle+tail": [(0, 1), (1, 2), (0, 2), (2, 3)],
    "star-5": [(0, 1), (0, 2), (0, 3), (0, 4)],
}
DEPTH = 4


def main() -> None:
    print("## depth-4 vertex-cover QAOA")
    print()
    print("| graph | n | ground | ratio | verdict |")
    print("| --- | --- | --- | --- | --- |")
    for name, edges in GRAPHS.items():
        problem = vertex_cover(edges)
        params, ceiling = multi_start(problem, DEPTH, starts=10, steps=150, seed=5)
        d = diagnose(params, problem, DEPTH, ceiling=ceiling)
        print(f"| {name} | {problem.n} | {problem.ground:.4f} | {d.ratio:.6f} | {d.verdict} |")
    print()
    print("A ratio of 1.0 means the depth-4 circuit reached the exact optimum.")


if __name__ == "__main__":
    main()
