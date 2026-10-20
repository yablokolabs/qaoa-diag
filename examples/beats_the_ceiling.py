"""What a measured ceiling means: the search, not the circuit.

A ceiling is the best energy a search found at a given depth. Beat it and you
have proved the search was too small, not that the diagnostic is broken.

Run: python examples/beats_the_ceiling.py
"""

import numpy as np

from qaoa_diag import diagnose, energy_function, minimize, multi_start, vertex_cover

EDGES = [(0, 1), (1, 2), (0, 2), (2, 3)]
DEPTH = 4


def report(label, diagnosis) -> None:
    print(label)
    print(f"  ratio    {diagnosis.ratio:.4f}")
    print(f"  ceiling  {diagnosis.ceiling_ratio:.4f}")
    print(f"  verdict  {diagnosis.verdict}")
    print(f"  advice   {diagnosis.advice}\n")


def main() -> None:
    problem = vertex_cover(EDGES)
    f = energy_function(problem, DEPTH)
    print(problem)

    weak_params, weak = multi_start(problem, DEPTH, starts=1, steps=20, seed=0)
    print(f"\nweak ceiling: one start, 20 steps -> ratio {problem.ratio(weak):.4f}")
    report("diagnosis against the weak ceiling",
           diagnose(weak_params, problem, DEPTH, ceiling=weak))
    print("Read that verdict with care: the search was one start of 20 steps,")
    print("so \"no better energy exists\" was not tested. The next run shows it.\n")

    x0 = np.random.default_rng(4).uniform(0, 2 * np.pi, (2, DEPTH))
    strong_params, strong = minimize(f, x0, steps=300, lr=0.05, method="adam")
    print(f"strong single run: 300 Adam steps -> ratio {problem.ratio(strong):.4f}")
    report("diagnosis against the same weak ceiling",
           diagnose(strong_params, problem, DEPTH, ceiling=weak))

    _, widened = multi_start(problem, DEPTH, starts=25, steps=200, seed=0)
    print(f"widened search: 25 starts -> ratio {problem.ratio(widened):.4f}")
    report("diagnosis against the widened ceiling",
           diagnose(strong_params, problem, DEPTH, ceiling=widened))

    print(
        "The ceiling is what the search found, not what the circuit can do."
        " A run that beats it is proof that the search was too small."
        " Widen the search before you conclude that a depth cannot do better."
    )


if __name__ == "__main__":
    main()
