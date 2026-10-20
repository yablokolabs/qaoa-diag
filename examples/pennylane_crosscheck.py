"""Cross-check the numpy simulator against PennyLane on the same angles.

The diagnostics are only worth trusting if the statevector code agrees with a
real quantum framework. This builds the same depth-3 circuit twice and compares
the energies and the measurement probabilities.

Run: python examples/pennylane_crosscheck.py
"""

import numpy as np

from qaoa_diag import energy, probabilities, vertex_cover

EDGES = [(0, 1), (1, 2), (0, 2), (2, 3)]
DEPTH = 3


def main() -> None:
    try:
        import pennylane as qml
    except ImportError:
        print("pennylane not installed; pip install pennylane to run the cross-check")
        return

    problem = vertex_cover(EDGES)
    n = problem.n

    coeffs, ops = [], []
    for i, j in EDGES:
        coeffs += [0.75, 0.75, 0.75]
        ops += [qml.PauliZ(i) @ qml.PauliZ(j), qml.PauliZ(i), qml.PauliZ(j)]
    for i in range(n):
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(i))
    cost_h = qml.Hamiltonian(coeffs, ops)
    mixer_h = qml.Hamiltonian([1.0] * n, [qml.PauliX(i) for i in range(n)])

    dev = qml.device("default.qubit", wires=n)

    def qaoa_layers(params):
        for wire in range(n):
            qml.Hadamard(wires=wire)
        for layer in range(DEPTH):
            qml.qaoa.cost_layer(params[0][layer], cost_h)
            qml.qaoa.mixer_layer(params[1][layer], mixer_h)

    @qml.qnode(dev)
    def expval(params):
        qaoa_layers(params)
        return qml.expval(cost_h)

    @qml.qnode(dev)
    def probs(params):
        qaoa_layers(params)
        return qml.probs(wires=range(n))

    rng = np.random.default_rng(11)
    print(f"{'run':>3}  {'numpy':>10}  {'pennylane':>10}  {'|diff|':>9}")
    worst_energy = 0.0
    for run in range(5):
        params = rng.uniform(0, 2 * np.pi, (2, DEPTH))
        ours = energy(params, problem, DEPTH)
        theirs = float(expval(params))
        worst_energy = max(worst_energy, abs(ours - theirs))
        print(f"{run:3d}  {ours:10.6f}  {theirs:10.6f}  {abs(ours - theirs):9.2e}")

    ours_probs = probabilities(params, problem, DEPTH)
    worst_prob = float(np.max(np.abs(ours_probs - np.asarray(probs(params)))))

    print(f"\nlargest |difference| in energy       {worst_energy:.2e}")
    print(f"largest |difference| in probability  {worst_prob:.2e}")
    if worst_energy < 1e-9 and worst_prob < 1e-9:
        print("\nBoth agree, so the diagnostics can run without a quantum framework.")
    else:
        print("\nThe two disagree: check the cost Hamiltonian and the qubit ordering.")


if __name__ == "__main__":
    main()
