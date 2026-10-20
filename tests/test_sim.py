"""The fast simulator is checked against two independent implementations.

1. a dense kronecker-product circuit (completely different code path),
2. PennyLane's own device, when it is installed.
"""

import numpy as np
import pytest

from qaoa_diag.problem import vertex_cover
from qaoa_diag.sim import energy, most_likely, probabilities, state, uniform_energy

GRAPH = [(0, 1), (1, 2), (0, 2), (2, 3)]
DEPTH = 4


def dense_energy(params, problem, depth):
    """Reference implementation: explicit matrices, no index tricks."""
    n = problem.n
    dim = 2**n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for layer in range(depth):
        psi = np.exp(-1j * params[0][layer] * problem.diag) * psi
        beta = params[1][layer]
        rx = np.array([[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]])
        for qubit in range(n):
            op = np.array([[1.0]])
            for other in range(n):
                op = np.kron(op, rx if other == qubit else np.eye(2))
            psi = op @ psi
    return float(np.real(psi.conj() @ (problem.diag * psi)))


def random_params(depth=DEPTH, seed=0):
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, (2, depth))


def test_matches_dense_matrix_simulation():
    problem = vertex_cover(GRAPH)
    for seed in range(3):
        params = random_params(seed=seed)
        assert energy(params, problem, DEPTH) == pytest.approx(
            dense_energy(params, problem, DEPTH), abs=1e-10
        )


def test_probabilities_are_normalised_and_consistent():
    problem = vertex_cover(GRAPH)
    params = random_params()
    probs = probabilities(params, problem, DEPTH)
    assert probs.shape == (2**problem.n,)
    assert probs.sum() == pytest.approx(1.0)
    assert energy(params, problem, DEPTH) == pytest.approx(float(probs @ problem.diag))
    bits, weight = most_likely(params, problem, DEPTH)
    assert weight == pytest.approx(probs.max())
    assert bits.shape == (problem.n,)


def test_depth_zero_is_the_uniform_superposition():
    problem = vertex_cover(GRAPH)
    params = np.zeros((2, DEPTH))
    assert energy(params, problem, DEPTH) == pytest.approx(uniform_energy(problem))
    assert np.allclose(np.abs(state(params, problem, DEPTH)) ** 2, 1 / 2**problem.n)


def test_matches_pennylane():
    qml = pytest.importorskip("pennylane")
    problem = vertex_cover(GRAPH)
    n = problem.n

    coeffs, ops = [], []
    for i, j in GRAPH:
        coeffs += [0.75, 0.75, 0.75]
        ops += [qml.PauliZ(i) @ qml.PauliZ(j), qml.PauliZ(i), qml.PauliZ(j)]
    for i in range(n):
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(i))
    cost_h = qml.Hamiltonian(coeffs, ops)
    mixer_h = qml.Hamiltonian([1.0] * n, [qml.PauliX(i) for i in range(n)])

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(params):
        for wire in range(n):
            qml.Hadamard(wires=wire)
        for layer in range(DEPTH):
            qml.qaoa.cost_layer(params[0][layer], cost_h)
            qml.qaoa.mixer_layer(params[1][layer], mixer_h)
        return qml.expval(cost_h)

    for seed in range(3):
        params = random_params(seed=seed)
        assert energy(params, problem, DEPTH) == pytest.approx(circuit(params), abs=1e-9)
