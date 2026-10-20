"""Diagonal cost Hamiltonians for quadratic binary problems, with exact ground truth.

Every cost Hamiltonian here is diagonal in the computational Z basis, so it is
just a table of 2**n real numbers.  That makes the *exact* answer cheap to
compute for small graphs, which is the whole point: you cannot tell a stuck
optimizer from a limited ansatz until you know the true ground state and the
best energy your ansatz can reach at a given depth.

Bit convention: index k of the table encodes the bitstring whose qubit i sits at
bit position ``n - 1 - i`` (qubit 0 most significant).  This matches PennyLane's
``default.qubit`` convention, so tables can be compared with circuit results.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Problem",
    "bit_table",
    "spin_table",
    "qubo",
    "vertex_cover",
    "max_cut",
    "min_vertex_cover",
]


def bit_table(n: int) -> np.ndarray:
    """(2**n, n) array of bits, column i is qubit i, row index is the state."""
    idx = np.arange(2**n)
    return (idx[:, None] >> np.arange(n - 1, -1, -1)) & 1


def spin_table(n: int) -> np.ndarray:
    """(2**n, n) array of spin values z_i = 1 - 2 b_i in {+1, -1}."""
    return 1 - 2 * bit_table(n)


class Problem:
    """A diagonal cost Hamiltonian over ``n`` qubits."""

    def __init__(self, diag, name: str = ""):
        diag = np.asarray(diag, dtype=float).ravel()
        size = diag.size
        n = int(round(np.log2(size))) if size > 0 else 0
        if size == 0 or 2**n != size:
            raise ValueError(f"diagonal length must be a power of two, got {size}")
        self.diag = diag
        self.n = n
        self.name = name

    # -- exact reference quantities -----------------------------------------
    @property
    def ground(self) -> float:
        """Exact minimum eigenvalue (the true optimum)."""
        return float(self.diag.min())

    def ground_states(self, tol: float = 1e-9) -> np.ndarray:
        """Indices of the basis states achieving the ground energy."""
        return np.flatnonzero(self.diag <= self.ground + tol)

    def best_bits(self, tol: float = 1e-9) -> np.ndarray:
        """Bitstring(s) of the ground state(s)."""
        return bit_table(self.n)[self.ground_states(tol)]

    def value(self, bits) -> float:
        """Energy of a single bitstring."""
        bits = np.asarray(bits, dtype=int)
        index = int(bits.dot(2 ** np.arange(self.n - 1, -1, -1)))
        return float(self.diag[index])

    def ratio(self, energy: float) -> float:
        """Approximation ratio in [0, 1]: 1.0 at the exact optimum.

        A negative optimum (vertex cover, max cut, Ising) uses the usual
        ``energy / ground``.  A positive optimum, which most QUBO objectives
        have, uses ``ground / energy`` instead, because dividing by a positive
        ground would score a worse answer higher.  A zero optimum is 1.0 only at
        zero energy.  All three branches read 1.0 at the optimum and fall
        towards 0 as the energy moves away from it.
        """
        g = float(self.ground)
        e = float(energy)
        if g == 0:
            return 1.0 if abs(e) <= 1e-9 else 0.0
        if g < 0:
            return min(1.0, max(0.0, e / g))
        return min(1.0, g / e) if e > 0 else 0.0

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"<Problem{label}: n={self.n}, ground={self.ground:.6g}>"


def from_spin_terms(n: int, zpairs=(), zsingle=(), const: float = 0.0, name: str = "") -> Problem:
    """Build a Problem from H = const + sum c_ij Z_i Z_j + sum c_i Z_i."""
    z = spin_table(n)
    diag = np.full(2**n, float(const))
    for i, j, c in zpairs:
        diag += c * z[:, i] * z[:, j]
    for i, c in zsingle:
        diag += c * z[:, i]
    return Problem(diag, name=name)


def qubo(n: int, pairs=(), linear=(), const: float = 0.0, name: str = "qubo") -> Problem:
    """Build a Problem from a binary quadratic objective.

        H(x) = const + sum_(i,j) c_ij x_i x_j + sum_i c_i x_i,   x_i in {0, 1}

    Substituting x = (1 - z) / 2 turns this into the Ising form used everywhere
    else, so any QUBO (knapsack penalties, portfolio risk, scheduling) drops
    straight into the same diagnostics.
    """
    zpairs: dict[tuple[int, int], float] = {}
    zsingle: dict[int, float] = {}
    total = float(const)
    for i, j, c in pairs:
        key = (min(i, j), max(i, j))
        # c x_i x_j = c/4 (1 - z_i - z_j + z_i z_j)
        zpairs[key] = zpairs.get(key, 0.0) + c / 4
        zsingle[i] = zsingle.get(i, 0.0) - c / 4
        zsingle[j] = zsingle.get(j, 0.0) - c / 4
        total += c / 4
    for i, c in linear:
        zsingle[i] = zsingle.get(i, 0.0) - c / 2
        total += c / 2
    return from_spin_terms(
        n,
        zpairs=[(i, j, c) for (i, j), c in sorted(zpairs.items())],
        zsingle=sorted(zsingle.items()),
        const=total,
        name=name,
    )


def _num_vertices(edges) -> int:
    return max(max(edge) for edge in edges) + 1


def vertex_cover(edges) -> Problem:
    """Minimum Vertex Cover cost Hamiltonian.

    H = 3/4 sum_(i,j) in edges (Z_i Z_j + Z_i + Z_j) - sum_i Z_i

    In terms of a cover C this is H = 3*U + 2|C| - 3m/4 - n, where U counts the
    edges left uncovered, so the ground state is a minimum vertex cover.
    """
    edges = [tuple(map(int, e)) for e in edges]
    n = _num_vertices(edges)
    coeff: dict[int, float] = {}
    for i, j in edges:
        coeff[i] = coeff.get(i, 0.0) + 0.75
        coeff[j] = coeff.get(j, 0.0) + 0.75
    for i in range(n):
        coeff[i] = coeff.get(i, 0.0) - 1.0
    return from_spin_terms(
        n,
        zpairs=[(i, j, 0.75) for i, j in edges],
        zsingle=sorted(coeff.items()),
        name="vertex-cover",
    )


def max_cut(edges) -> Problem:
    """Max-Cut as a minimisation problem: H = sum_(i,j) (Z_i Z_j - 1) / 2.

    That equals minus the number of cut edges, so the ground energy is
    ``-maxcut`` and ``ratio`` reads as achieved cuts / maximum cuts.
    """
    edges = [tuple(map(int, e)) for e in edges]
    n = _num_vertices(edges)
    return from_spin_terms(
        n,
        zpairs=[(i, j, 0.5) for i, j in edges],
        const=-len(edges) / 2,
        name="max-cut",
    )


def min_vertex_cover(edges):
    """Exact minimum vertex cover by brute force (small graphs only)."""
    edges = [tuple(map(int, e)) for e in edges]
    n = _num_vertices(edges)
    for size in range(n + 1):
        for mask in range(2**n):
            chosen = {i for i in range(n) if (mask >> (n - 1 - i)) & 1}
            if len(chosen) != size:
                continue
            if all(i in chosen or j in chosen for i, j in edges):
                return chosen
    return set()
