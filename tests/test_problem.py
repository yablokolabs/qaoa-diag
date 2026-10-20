import numpy as np
import pytest

from qaoa_diag.problem import (
    Problem,
    bit_table,
    max_cut,
    min_vertex_cover,
    qubo,
    spin_table,
    vertex_cover,
)

GRAPH = [(0, 1), (1, 2), (0, 2), (2, 3)]


def cover_energy(edges, cover):
    """Independent evaluation of the statement's cost Hamiltonian for a cover."""
    n = max(max(e) for e in edges) + 1
    # a vertex in the cover has bit 1, hence spin z = -1
    z = [-1 if i in cover else 1 for i in range(n)]
    total = 0.0
    for i, j in edges:
        total += 0.75 * (z[i] * z[j] + z[i] + z[j])
    return total - sum(z)


def test_bit_and_spin_tables_agree_with_index_bits():
    n = 3
    bits = bit_table(n)
    z = spin_table(n)
    for index in range(2**n):
        expected = [(index >> (n - 1 - i)) & 1 for i in range(n)]
        assert bits[index].tolist() == expected
        assert z[index].tolist() == [1 - 2 * b for b in expected]


def test_vertex_cover_ground_is_a_minimum_cover():
    problem = vertex_cover(GRAPH)
    cover = min_vertex_cover(GRAPH)
    assert len(cover) == 2
    # the exact ground energy must equal the independent formula for that cover
    assert problem.ground == pytest.approx(cover_energy(GRAPH, cover))
    # every ground state is a valid cover of minimum size
    for bits in problem.best_bits():
        chosen = {i for i, b in enumerate(bits) if b == 1}
        assert all(i in chosen or j in chosen for i, j in GRAPH)
        assert len(chosen) == len(cover)


def test_vertex_cover_ground_matches_brute_force_over_all_covers():
    problem = vertex_cover(GRAPH)
    n = problem.n
    best = min(
        cover_energy(GRAPH, {i for i in range(n) if (mask >> i) & 1})
        for mask in range(2**n)
    )
    assert problem.ground == pytest.approx(best)


def test_max_cut_ground_is_minus_the_maximum_cut():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]  # odd cycle: max cut < m
    problem = max_cut(edges)
    best = np.inf
    for bits in bit_table(problem.n):
        z = 1 - 2 * bits
        cut = sum(0.5 * (1 - z[i] * z[j]) for i, j in edges)
        best = min(best, -cut)
    assert problem.ground == pytest.approx(best)
    assert problem.ground == pytest.approx(-4.0)
    assert problem.ratio(problem.ground) == pytest.approx(1.0)


def test_ratio_and_value_helpers():
    problem = vertex_cover(GRAPH)
    bits = problem.best_bits()[0]
    assert problem.value(bits) == pytest.approx(problem.ground)
    assert problem.ratio(problem.ground) == pytest.approx(1.0)


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        Problem(np.zeros(3))


def test_qubo_matches_direct_evaluation_of_the_objective():
    """The x = (1 - z) / 2 substitution is easy to get subtly wrong."""
    rng = np.random.default_rng(0)
    n = 5
    pairs = [(i, j, float(rng.normal())) for i in range(n) for j in range(i + 1, n)]
    linear = [(i, float(rng.normal())) for i in range(n)]
    problem = qubo(n, pairs=pairs, linear=linear, const=0.75, name="random-qubo")
    for bits in bit_table(n):
        direct = (
            0.75
            + sum(c * bits[i] * bits[j] for i, j, c in pairs)
            + sum(c * bits[i] for i, c in linear)
        )
        assert problem.value(bits) == pytest.approx(direct, abs=1e-12)

    single = qubo(2, pairs=[(0, 1, 1.0)])
    assert single.diag.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_ratio_is_a_fraction_of_the_optimum_for_any_sign():
    # positive optimum, as most QUBO objectives have
    positive = Problem(np.array([4.0, 2.0, 1.0, 8.0]))
    assert positive.ground == 1.0
    assert positive.ratio(positive.ground) == pytest.approx(1.0)
    assert positive.ratio(2.0) == pytest.approx(0.5)
    assert 0.0 <= positive.ratio(8.0) < 1.0
    # zero optimum, and an energy worse than the starting state
    zero = Problem(np.array([0.0, 3.0, 5.0, 9.0]))
    assert zero.ratio(0.0) == pytest.approx(1.0)
    assert zero.ratio(3.0) == pytest.approx(0.0)
    # negative optimum is unchanged, and the ratio is clamped to [0, 1]
    negative = vertex_cover(GRAPH)
    assert negative.ratio(negative.ground) == pytest.approx(1.0)
    assert negative.ratio(negative.ground / 2) == pytest.approx(0.5)
    assert negative.ratio(1.0) == pytest.approx(0.0)
