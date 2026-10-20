"""Use case outside graphs: pick items under a budget (knapsack as a QUBO).

Maximise profit while staying inside capacity, as a binary quadratic objective

    H(x) = -sum profit_i x_i + penalty * (capacity - sum weight_i x_i)^2

expanded into pairs so it matches the QUBO form. The penalty term is what makes
the problem quadratic, and it is also what a shallow circuit struggles with.

Run: python examples/custom_qubo.py
"""

from qaoa_diag import diagnose, multi_start, qubo

items = ["laptop", "camera", "tent", "guitar"]
profit = [7, 5, 4, 3]
weight = [4, 3, 2, 2]
capacity = 6
penalty = 1.5

# H = -sum profit_i x_i + penalty * (capacity - sum w_i x_i)^2, expanded:
#   linear_i = -profit_i - 2*penalty*capacity*w_i + penalty*w_i^2   (x_i^2 = x_i)
#   pair_ij  = 2*penalty*w_i*w_j
#   const    = penalty*capacity^2
linear = [
    (i, -profit[i] - 2 * penalty * capacity * weight[i] + penalty * weight[i] ** 2)
    for i in range(len(items))
]
pairs = [
    (i, j, 2 * penalty * weight[i] * weight[j])
    for i in range(len(items))
    for j in range(i + 1, len(items))
]
const = penalty * capacity**2

problem = qubo(len(items), pairs=pairs, linear=linear, const=const, name="knapsack")
print(problem)
print(f"capacity {capacity}, penalty {penalty}\n")

best = problem.best_bits()[0]
chosen = [items[i] for i, b in enumerate(best) if b]
print(f"exact optimum: {chosen}  (profit {sum(p for p, b in zip(profit, best) if b)},"
      f" weight {sum(w for w, b in zip(weight, best) if b)})")
print(f"energy {problem.ground:.4f}\n")

print("depth  ratio   verdict")
for depth in (1, 2, 3, 4):
    params, ceiling = multi_start(problem, depth, starts=10, steps=150, seed=depth)
    d = diagnose(params, problem, depth, ceiling=ceiling)
    print(f"  {depth}    {d.ratio:.4f}  {d.verdict}")

print("\nSame diagnostics as the graph problems: ground truth, ceiling, verdict.")
