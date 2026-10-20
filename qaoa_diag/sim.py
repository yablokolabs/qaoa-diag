"""Statevector simulation of the depth-p QAOA circuit, numpy only.

The circuit is fixed by the statement-style variational ansatz

    |psi> = prod_{l=1..p} exp(-i beta_l H_mixer) exp(-i gamma_l H_cost) |+>^n

with ``H_cost`` diagonal (a :class:`~qaoa_diag.problem.Problem`) and
``H_mixer = sum_i X_i``.  Since exp(-i beta sum_i X_i) factorises into one
Rx(2 beta) per qubit and exp(-i gamma H_cost) is a diagonal phase, a full
evaluation costs O(p n 2**n) - microseconds for the small graphs this library
targets.

Parameter layout is ``params[0] = gammas`` (cost angles) and
``params[1] = betas`` (mixer angles), each of length ``depth``.
"""

from __future__ import annotations

import numpy as np

from .problem import Problem, bit_table

__all__ = ["state", "probabilities", "energy", "most_likely", "uniform_energy"]


def _apply_rx(psi: np.ndarray, beta: float, qubit: int, n: int) -> None:
    """Apply Rx(2*beta) to ``qubit`` in place, using index strides."""
    c, s = np.cos(beta), np.sin(beta)
    view = psi.reshape(2 ** (n - 1 - qubit), 2, 2**qubit)
    a0 = view[:, 0, :].copy()
    a1 = view[:, 1, :].copy()
    view[:, 0, :] = c * a0 - 1j * s * a1
    view[:, 1, :] = -1j * s * a0 + c * a1


def state(params, problem: Problem, depth: int) -> np.ndarray:
    """Full statevector at the end of the QAOA circuit."""
    params = np.asarray(params, dtype=float).reshape(2, depth)
    n = problem.n
    psi = np.full(2**n, 1 / np.sqrt(2**n), dtype=complex)
    for layer in range(depth):
        psi = psi * np.exp(-1j * params[0][layer] * problem.diag)
        for qubit in range(n):
            _apply_rx(psi, params[1][layer], qubit, n)
    return psi


def probabilities(params, problem: Problem, depth: int) -> np.ndarray:
    """Measurement probabilities over all 2**n basis states."""
    return np.abs(state(params, problem, depth)) ** 2


def energy(params, problem: Problem, depth: int) -> float:
    """Expectation value <psi| H_cost |psi>."""
    probs = probabilities(params, problem, depth)
    return float(probs @ problem.diag)


def uniform_energy(problem: Problem) -> float:
    """Energy of the depth-0 state |+>^n (the starting point of every run)."""
    return float(problem.diag.mean())


def most_likely(params, problem: Problem, depth: int) -> tuple[np.ndarray, float]:
    """Most probable bitstring and its probability."""
    probs = probabilities(params, problem, depth)
    index = int(np.argmax(probs))
    return bit_table(problem.n)[index], float(probs[index])
