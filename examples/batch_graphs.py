"""Which graphs are hard for depth-4 QAOA? Run a batch and tabulate.

Each row is one graph: exact optimum, the best energy the depth-4 circuit
reaches, and the verdict. Cycles and stars solve exactly. Denser graphs do not.

Run: python examples/batch_graphs.py
"""

from qaoa_diag import diagnose, multi_start, vertex_cover

GRAPHS = {
    "path-4": [(0, 1), (1, 2), (2, 3)],
    "cycle-4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "cycle-6": [(i, (i + 1) % 6) for i in range(6)],
    "triangle+tail": [(0, 1), (1, 2), (0, 2), (2, 3)],
    "star-5": [(0, 1), (0, 2), (0, 3), (0, 4)],
    "complete-4": [(i, j) for i in range(4) for j in range(i + 1, 4)],
    "grid-2x3": [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)],
}

DEPTH = 4

print(f"depth-{DEPTH} vertex-cover QAOA, best of 10 random starts\n")
print(f"{'graph':14s} {'n':>2s} {'m':>3s} {'ground':>8s} {'ceiling':>8s} {'ratio':>7s}  verdict")
for name, edges in GRAPHS.items():
    problem = vertex_cover(edges)
    params, ceiling = multi_start(problem, DEPTH, starts=10, steps=150, seed=5)
    d = diagnose(params, problem, DEPTH, ceiling=ceiling)
    print(
        f"{name:14s} {problem.n:2d} {len(edges):3d} {problem.ground:8.3f} "
        f"{d.ceiling:8.3f} {d.ratio:7.3f}  {d.verdict}"
    )
