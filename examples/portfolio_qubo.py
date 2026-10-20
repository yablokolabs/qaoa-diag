"""Pick two assets under a risk-return objective, then grade the run by hand.

Portfolio selection with a fixed number of holdings is a QUBO: returns enter
linearly, risk enters quadratically through the covariance matrix, and the
budget enters as a penalty. Four assets means sixteen subsets, so the exact
answer is one loop and the QAOA result has something honest to be compared
against.

Run: python examples/portfolio_qubo.py
"""

import itertools

import numpy as np

from qaoa_diag import most_likely, multi_start, qubo

NAMES = ["tech", "health", "energy", "bonds"]
MU = np.array([0.10, 0.07, 0.05, 0.03])
SIGMA = np.array([0.20, 0.12, 0.10, 0.06])
CORRELATION = 0.3
RISK = 2.0
BUDGET = 1.0
HOLD = 2
DEPTH = 4

N = len(MU)
COV = np.array(
    [[SIGMA[i] * SIGMA[j] * (1.0 if i == j else CORRELATION) for j in range(N)]
     for i in range(N)]
)


def objective(chosen) -> float:
    """-return + risk * variance + budget penalty, the thing being minimised."""
    x = np.zeros(N)
    x[list(chosen)] = 1.0
    return float(-MU @ x + RISK * x @ COV @ x + BUDGET * (x.sum() - HOLD) ** 2)


def build():
    pairs = [
        (i, j, 2.0 * RISK * COV[i, j] + 2.0 * BUDGET)
        for i in range(N)
        for j in range(i + 1, N)
    ]
    linear = [
        (i, -MU[i] + RISK * COV[i, i] + BUDGET * (1 - 2 * HOLD))
        for i in range(N)
    ]
    return qubo(N, pairs=pairs, linear=linear, const=BUDGET * HOLD**2, name="portfolio")


def main() -> None:
    best = min(itertools.combinations(range(N), HOLD), key=objective)
    best_value = objective(best)
    weights = np.zeros(N)
    weights[list(best)] = 1.0
    variance = float(weights @ COV @ weights)

    print("assets:", ", ".join(NAMES))
    print(f"returns {np.round(MU, 2).tolist()}  vols {np.round(SIGMA, 2).tolist()}"
          f"  correlation {CORRELATION}")
    print(f"\nbrute force over {2 ** N} subsets, holding exactly {HOLD}:")
    print(f"  best subset {[NAMES[i] for i in best]}")
    print(f"  return {MU @ weights:.4f}  variance {variance:.5f}"
          f"  objective {best_value:.4f}")

    problem = build()
    print(f"\n{problem}")
    params, energy = multi_start(problem, DEPTH, starts=10, steps=150, seed=3)
    bits, prob = most_likely(params, problem, DEPTH)
    picked = [NAMES[i] for i in range(N) if bits[i]]

    print(f"depth-{DEPTH} QAOA: energy {energy:.4f}  ratio {problem.ratio(energy):.4f}")
    print(f"  most likely bitstring {''.join(str(int(b)) for b in bits)}"
          f"  probability {prob:.4f}  holdings {picked}")
    print(f"  objective from the bitstring {problem.value(bits):.4f}")
    match = abs(problem.value(bits) - best_value) < 1e-9
    print(f"  matches the brute-force optimum: {'yes' if match else 'no'}")

    print(
        "\nThe ratio grades the energy the circuit reached, not the portfolio it"
        "\nprinted. Decode the bits and check the objective before you report it."
    )


if __name__ == "__main__":
    main()
