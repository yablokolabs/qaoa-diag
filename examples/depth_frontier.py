"""How much does one extra QAOA layer buy on one graph?

The ceiling rises with depth and saturates. Where it saturates is the answer to
"should I add a layer or spend the budget on the optimizer": if depth 3 and
depth 5 reach the same energy, the circuit stopped being the limit.

Run: python examples/depth_frontier.py
"""

from qaoa_diag import depth_frontier, vertex_cover

GRAPHS = {
    "triangle+tail": [(0, 1), (1, 2), (0, 2), (2, 3)],
    "cycle-6": [(i, (i + 1) % 6) for i in range(6)],
}
DEPTHS = (1, 2, 3, 4, 5, 6)


def main() -> None:
    for name, edges in GRAPHS.items():
        problem = vertex_cover(edges)
        print(f"=== {name}: {len(edges)} edges, ground {problem.ground:.3f} ===")
        rows = depth_frontier(problem, depths=DEPTHS, starts=12, steps=150, seed=1)
        print("depth  energy     gap     ratio   gain over previous depth")
        previous = None
        for depth, row in rows.items():
            gain = "-" if previous is None else f"{previous - row['energy']:.4f}"
            print(
                f"  {depth}    {row['energy']:7.4f}  {row['gap']:6.4f}"
                f"  {row['ratio']:.4f}   {gain}"
            )
            previous = row["energy"]

        last, before = rows[DEPTHS[-1]]["energy"], rows[DEPTHS[-2]]["energy"]
        print(f"a layer from {DEPTHS[-2]} to {DEPTHS[-1]} buys {before - last:.4f}\n")


if __name__ == "__main__":
    main()
