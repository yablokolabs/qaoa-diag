"""Command line report: ``python -m qaoa_diag --edges 0-1,1-2,2-3,3-0``."""

from __future__ import annotations

import argparse

import numpy as np

from .diagnose import diagnose
from .optimize import budget_sweep, depth_frontier, energy_function, minimize
from .problem import max_cut, vertex_cover


def parse_edges(text: str):
    edges = []
    for chunk in text.replace(" ", "").split(","):
        if not chunk:
            continue
        left, right = chunk.split("-")
        edges.append((int(left), int(right)))
    if not edges:
        raise SystemExit("no edges given")
    return edges


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="qaoa_diag",
        description="Run depth-p QAOA and explain how far the result is from the ceiling.",
    )
    parser.add_argument("--edges", required=True, help='e.g. "0-1,1-2,2-3,3-0"')
    parser.add_argument("--problem", choices=["vertex-cover", "max-cut"], default="vertex-cover")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50, help="optimizer steps for the reported run")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--method", choices=["adam", "gd", "lbfgs"], default="adam",
                        help="optimizer for the reported run (ceiling always uses --ceiling-method)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--starts", type=int, default=20, help="runs used to measure the ceiling")
    parser.add_argument("--frontier", action="store_true", help="also print depths 1..depth table")
    parser.add_argument("--sweep", action="store_true", help="also print ratio vs step budget")
    args = parser.parse_args(argv)

    edges = parse_edges(args.edges)
    problem = vertex_cover(edges) if args.problem == "vertex-cover" else max_cut(edges)
    print(problem)

    f = energy_function(problem, args.depth)
    x0 = np.random.default_rng(args.seed).uniform(0, 2 * np.pi, (2, args.depth))
    params, value = minimize(f, x0, steps=args.steps, lr=args.lr, method=args.method)

    print()
    print(diagnose(params, problem, args.depth, starts=args.starts, seed=args.seed))

    if args.frontier:
        print("\ndepth ceiling")
        rows = depth_frontier(problem, depths=tuple(range(1, args.depth + 1)), starts=args.starts)
        for depth, row in rows.items():
            print(f"  p={depth}  energy {row['energy']:.6f}  ratio {row['ratio']:.6f}"
                  f"  gap {row['gap']:.6f}")

    if args.sweep:
        print("\nbudget sweep")
        for row in budget_sweep(problem, args.depth, seed=args.seed, lr=args.lr,
                                method=args.method):
            print(f"  steps {row['steps']:>4}  energy {row['energy']:.6f}"
                  f"  ratio {row['ratio']:.6f}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
