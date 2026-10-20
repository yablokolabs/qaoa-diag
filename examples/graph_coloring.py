"""Three-colour a triangle as a QUBO, then check that the answer is a colouring.

A constraint-satisfaction problem becomes a diagonal Hamiltonian once every
violated constraint gets a price. Nine binary variables: x[i, c] is 1 when
vertex i takes colour c, stored at qubit index i*3 + c. Breaking "one colour
per vertex" or "endpoints differ" costs 1 each, so energy 0 is a valid
colouring and the ratio is trustworthy only if the decoded bits really check
out.

Run: python examples/graph_coloring.py
"""

from qaoa_diag import depth_frontier, most_likely, multi_start, qubo

K = 3
VERTICES = 3
EDGES = [(0, 1), (1, 2), (0, 2)]
COLORS = ["red", "green", "blue"]
DEPTH = 3


def index(vertex: int, color: int) -> int:
    return vertex * K + color


def build():
    """QUBO whose ground state is a proper 3-colouring of the triangle."""
    pairs = []
    for v in range(VERTICES):
        for c in range(K):
            for c2 in range(c + 1, K):
                pairs.append((index(v, c), index(v, c2), 2.0))
    for i, j in EDGES:
        for c in range(K):
            pairs.append((index(i, c), index(j, c), 1.0))
    linear = [(index(v, c), -1.0) for v in range(VERTICES) for c in range(K)]
    return qubo(
        VERTICES * K,
        pairs=pairs,
        linear=linear,
        const=float(VERTICES),
        name="3-coloring",
    )


def colors_of(bits):
    """Colour per vertex, None where the vertex is not exactly one-hot."""
    chosen = []
    for v in range(VERTICES):
        picks = [c for c in range(K) if bits[index(v, c)]]
        chosen.append(COLORS[picks[0]] if len(picks) == 1 else None)
    return chosen


def main() -> None:
    problem = build()
    print(problem)
    print(f"energy 0 means every constraint holds; ground energy {problem.ground:.4f}\n")

    params, energy = multi_start(problem, DEPTH, starts=10, steps=150, seed=2)
    bits, prob = most_likely(params, problem, DEPTH)
    colors = colors_of(bits)
    print(f"depth-{DEPTH} best run: energy {energy:.4f}"
          f"  (constraint cost, 0 is a valid colouring)")
    print(f"most likely bitstring {''.join(str(int(b)) for b in bits)}"
          f"  probability {prob:.4f}")
    print(f"colour per vertex  {colors}")
    for i, j in EDGES:
        ok = colors[i] is not None and colors[j] is not None and colors[i] != colors[j]
        print(f"  edge {i}-{j}: {colors[i]} vs {colors[j]}  {'ok' if ok else 'violated'}")

    print("\ndepth  energy    gap")
    rows = depth_frontier(problem, depths=(1, 2, 3, 4), starts=8, steps=120)
    for depth, row in rows.items():
        print(f"  {depth}    {row['energy']:7.4f}  {row['gap']:7.4f}")

    print(
        "\nThe optimum here is 0, so `ratio` can only be 1 or 0 and the gap is"
        "\nthe readable column: it is the price of the constraints still broken."
        "\nA gap that shrinks slowly with depth says the circuit, not the"
        "\noptimizer, is the limit."
    )


if __name__ == "__main__":
    main()
