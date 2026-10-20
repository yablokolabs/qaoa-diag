import numpy as np
import pytest

from qaoa_diag.diagnose import diagnose
from qaoa_diag.optimize import budget_sweep, depth_frontier, energy_function, minimize, multi_start
from qaoa_diag.problem import vertex_cover

TRIANGLE_TAIL = [(0, 1), (1, 2), (0, 2), (2, 3)]
CYCLE = [(0, 1), (1, 2), (2, 3), (3, 0)]
DEPTH = 4


def fixed_start(problem, seed=13, depth=DEPTH):
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, (2, depth))


def test_zero_init_is_reported_as_a_stationary_point():
    problem = vertex_cover(TRIANGLE_TAIL)
    d = diagnose(np.zeros((2, DEPTH)), problem, DEPTH, starts=8)
    assert d.grad_norm == pytest.approx(0.0, abs=1e-12)
    assert d.verdict == "stationary-point"
    assert d.ratio == pytest.approx(problem.ratio(problem.diag.mean()))


def test_modest_budget_is_under_converged():
    problem = vertex_cover(TRIANGLE_TAIL)
    f = energy_function(problem, DEPTH)
    params, _ = minimize(f, fixed_start(problem), steps=3, lr=0.02)
    d = diagnose(params, problem, DEPTH, starts=16)
    assert d.verdict == "under-converged"
    assert d.ceiling < d.energy


def test_best_of_many_starts_hits_the_depth_ceiling():
    problem = vertex_cover(TRIANGLE_TAIL)
    best, value = multi_start(problem, DEPTH, starts=12, steps=200, seed=1)
    d = diagnose(best, problem, DEPTH, starts=12, steps=200, seed=1)
    # the ceiling at depth 4 is short of the optimum for this graph
    assert d.ceiling_ratio < 0.99
    assert d.verdict == "ansatz-limited"
    assert d.ratio == pytest.approx(d.ceiling_ratio, abs=1e-9)
    assert d.energy == pytest.approx(value, abs=1e-9)


def test_ceiling_reaches_ground_state_for_a_four_cycle():
    problem = vertex_cover(CYCLE)
    best, _ = multi_start(problem, DEPTH, starts=16, steps=250, seed=2)
    d = diagnose(best, problem, DEPTH, starts=16, steps=250, seed=2)
    assert d.ceiling_ratio == pytest.approx(1.0, abs=1e-6)
    assert d.verdict == "optimal"


def test_budget_sweep_is_monotone_in_the_budget_and_deterministic():
    problem = vertex_cover(TRIANGLE_TAIL)
    rows = budget_sweep(problem, DEPTH, budgets=(0, 25, 100))
    ratios = [row["ratio"] for row in rows]
    assert ratios[0] == pytest.approx(problem.ratio(problem.diag.mean()))
    assert ratios[1] > ratios[0]
    assert ratios[2] >= ratios[1]
    assert rows == budget_sweep(problem, DEPTH, budgets=(0, 25, 100))


def test_depth_frontier_is_non_decreasing():
    problem = vertex_cover(TRIANGLE_TAIL)
    frontier = depth_frontier(problem, depths=(1, 2, 3), starts=12, steps=250)
    ratios = [frontier[d]["ratio"] for d in (1, 2, 3)]
    assert ratios == sorted(ratios)
    for depth in (1, 2, 3):
        assert frontier[depth]["gap"] >= -1e-9
