"""Use case: number partition, a QUBO you can check by hand.

Split a list of numbers into two groups whose sums are as close as possible.
Writing x_i = 1 for "in group A" and S = sum a_i x_i, the cost of a split is

    (sum_A - sum_B)**2 = (2S - T)**2,   T = sum of all numbers,

which expands to a QUBO:

    pair_ij  =  8 * a_i * a_j
    linear_i =  4 * a_i**2 - 4 * T * a_i
    const    =  T**2

Run: python examples/number_partition.py
"""

from qaoa_diag import depth_frontier, qubo

NUMBERS = [3, 1, 4, 1, 5, 9, 2, 6]
TOTAL = sum(NUMBERS)
DEPTHS = (1, 2, 3, 4, 5)


def build():
    n = len(NUMBERS)
    pairs = [
        (i, j, 8 * NUMBERS[i] * NUMBERS[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    linear = [(i, 4 * NUMBERS[i] ** 2 - 4 * TOTAL * NUMBERS[i]) for i in range(n)]
    return qubo(n, pairs=pairs, linear=linear, const=TOTAL**2, name="number-partition")


def main() -> None:
    problem = build()
    print(problem)
    print(f"numbers {NUMBERS}, total {TOTAL} (odd, so a perfect split is impossible)")
    print()

    bits = problem.best_bits()[0]
    group_a = [NUMBERS[i] for i, bit in enumerate(bits) if bit]
    group_b = [NUMBERS[i] for i, bit in enumerate(bits) if not bit]
    diff = sum(group_a) - sum(group_b)
    print(f"best split: {group_a} = {sum(group_a)}  vs  {group_b} = {sum(group_b)}")
    print(f"difference {diff}, squared {diff**2}, ground energy {problem.ground:.1f}")
    print()

    print("depth   energy      gap     ratio")
    frontier = depth_frontier(problem, depths=DEPTHS, starts=10, steps=150, seed=2)
    for depth, row in frontier.items():
        print(f"  {depth}    {row['energy']:8.3f}  {row['gap']:7.3f}  {row['ratio']:.4f}")

    deepest = frontier[DEPTHS[-1]]
    best_at_depth = (2 * sum(group_a) - TOTAL) ** 2
    print()
    print(
        f"The best split costs {best_at_depth}, and the depth-{DEPTHS[-1]} circuit"
        f" stops {deepest['gap']:.3f} above it. The quartic penalty turns most"
        "\nbasis states into near-degenerate local minima, so a shallow circuit"
        "\nreports a cost it cannot beat: deepen the circuit, or rewrite the"
        "\npenalty before blaming the optimizer."
    )


if __name__ == "__main__":
    main()
