"""Use case: knapsack as a QUBO, and how the penalty weight decides the answer.

    H(x) = -sum p_i x_i + P * (C - sum w_i x_i)**2

A small P lets the exact ground state of H break the capacity, so QAOA is
optimizing a badly posed problem and no amount of depth repairs it. A large
enough P puts the best feasible subset at the bottom of the spectrum.

Run: python examples/knapsack_penalty.py
"""

from qaoa_diag import multi_start, qubo

ITEMS = ["laptop", "camera", "tent", "guitar"]
PROFIT = [7, 5, 4, 3]
WEIGHT = [4, 3, 2, 2]
CAPACITY = 6
DEPTH = 3


def build(penalty: float):
    """Knapsack QUBO with the capacity term expanded into linear/pair weights."""
    n = len(ITEMS)
    linear = [
        (i, -PROFIT[i] - 2 * penalty * CAPACITY * WEIGHT[i] + penalty * WEIGHT[i] ** 2)
        for i in range(n)
    ]
    pairs = [
        (i, j, 2 * penalty * WEIGHT[i] * WEIGHT[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return qubo(n, pairs=pairs, linear=linear, const=penalty * CAPACITY**2, name="knapsack")


def main() -> None:
    n = len(ITEMS)
    print(f"capacity {CAPACITY}, items {', '.join(ITEMS)}")
    print()
    print(f"{'penalty':>8s} {'ground':>8s} {'weight':>7s} {'fits':>5s} {'ratio':>7s}  chosen")

    for penalty in (0.1, 0.3, 1.5, 5.0):
        problem = build(penalty)
        _, ceiling = multi_start(problem, DEPTH, starts=10, steps=150, seed=1)
        picked = [i for i, bit in enumerate(problem.best_bits()[0]) if bit]
        load = sum(WEIGHT[i] for i in picked)
        print(
            f"{penalty:8.2f} {problem.ground:8.3f} {load:7d} "
            f"{'yes' if load <= CAPACITY else 'NO':>5s} "
            f"{problem.ratio(ceiling):7.4f}  "
            f"{', '.join(ITEMS[i] for i in picked) or '-'}"
        )

    best_profit, best_pick = -1, []
    for mask in range(2**n):
        picked = [i for i in range(n) if (mask >> (n - 1 - i)) & 1]
        if sum(WEIGHT[i] for i in picked) > CAPACITY:
            continue
        profit = sum(PROFIT[i] for i in picked)
        if profit > best_profit:
            best_profit, best_pick = profit, picked

    print()
    print(
        f"best feasible subset: {', '.join(ITEMS[i] for i in best_pick)}  "
        f"profit {best_profit}  weight {sum(WEIGHT[i] for i in best_pick)}"
    )
    print()
    print("At penalty 0.1 the optimum of H is whatever earns most, capacity ignored.")
    print("From 1.5 up the ground state is the feasible answer, and the depth-3")
    print("ratio then only says how much of that answer QAOA reaches.")


if __name__ == "__main__":
    main()
