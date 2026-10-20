"""Classical optimizers for QAOA angles, plus the multi-start reference budget.

Two budgets matter and they are easy to confuse:``minimize``      one run from one random start - what a submission usually does.
``multi_start``   many runs, best energy kept - the reference for "the best this
depth can do", which is what tells you whether a disappointing
ratio is an optimizer problem or an ansatz problem.

The gradient is a central finite difference.  It is exact enough for these tiny
parameter spaces; a run is microseconds.  ``lbfgs`` uses scipy and is optional -
the rest of the package needs numpy only.

A multi-start result is a *measured* ceiling: it says no run we tried did better,
not that nothing better exists.  That is why :func:`qaoa_diag.diagnose.diagnose`
compares a run against both the ceiling and the run's own energy.
"""

from __future__ import annotations

import numpy as np

from .problem import Problem
from .sim import energy, uniform_energy

__all__ = [
    "gradient",
    "adam",
    "gradient_descent",
    "lbfgs",
    "OPTIMIZERS",
    "minimize",
    "energy_function",
    "multi_start",
    "depth_frontier",
    "budget_sweep",
]


def gradient(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar function of a flat array."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus.flat[i] += h
        minus.flat[i] -= h
        g.flat[i] = (f(plus) - f(minus)) / (2 * h)
    return g


def adam(f, x0, steps: int, lr: float, b1: float = 0.9, b2: float = 0.999,
         eps: float = 1e-8) -> tuple[np.ndarray, float, list[float]]:
    """Adam on a flat parameter vector.  Returns (x, final f, history)."""
    x = np.array(x0, dtype=float).ravel()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [f(x)]
    for t in range(1, steps + 1):
        g = gradient(f, x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        history.append(f(x))
    return x, history[-1], history


def gradient_descent(f, x0, steps: int, lr: float) -> tuple[np.ndarray, float, list[float]]:
    """Plain gradient descent.  Returns (x, final f, history)."""
    x = np.array(x0, dtype=float).ravel()
    history = [f(x)]
    for _ in range(steps):
        x = x - lr * gradient(f, x)
        history.append(f(x))
    return x, history[-1], history


def lbfgs(f, x0, steps: int, lr: float = 0.0) -> tuple[np.ndarray, float, list[float]]:
    """L-BFGS-B from scipy, with a finite-difference gradient.

    Converges in far fewer function evaluations than Adam, which makes it the
    cheapest way to measure a depth ceiling.  ``steps`` caps the iterations and
    ``lr`` is accepted for signature compatibility and ignored.
    """
    try:
        from scipy.optimize import minimize as scipy_minimize
    except ImportError as exc:  # pragma: no cover - scipy is optional
        raise ImportError(
            "method='lbfgs' needs scipy (`pip install scipy`); use method='adam' "
            "to stay on numpy only"
        ) from exc

    x = np.array(x0, dtype=float).ravel()
    history = [f(x)]
    result = scipy_minimize(
        f,
        x,
        jac=lambda z: gradient(f, z),
        method="L-BFGS-B",
        options={"maxiter": int(steps)},
        callback=lambda z: history.append(float(f(z))),
    )
    history.append(float(result.fun))
    return result.x, float(result.fun), history


OPTIMIZERS = {"adam": adam, "gd": gradient_descent, "lbfgs": lbfgs}


def energy_function(problem: Problem, depth: int):
    """Cost function over flat parameter arrays for this problem and depth."""
    return lambda flat: energy(np.asarray(flat).reshape(2, depth), problem, depth)


def minimize(f, x0, steps: int = 50, lr: float = 0.02, method: str = "adam"):
    """One optimizer run.  Returns (params, energy) with the shape of ``x0``.

    ``method`` is one of ``adam``, ``gd`` or ``lbfgs``.
    """
    if method not in OPTIMIZERS:
        raise ValueError(f"unknown method {method!r}; pick one of {sorted(OPTIMIZERS)}")
    shape = np.shape(x0) or (np.size(x0),)
    x, value, _ = OPTIMIZERS[method](f, np.asarray(x0, dtype=float), steps, lr)
    return x.reshape(shape), value


def multi_start(problem: Problem, depth: int, starts: int = 20, steps: int = 200,
                lr: float = 0.05, seed: int = 0, method: str = "lbfgs"):
    """Best energy any of ``starts`` runs reaches - the measured depth ceiling.

    This is the reference a well-optimized single run should approach.  If a run
    sits well above it, the optimizer, not the ansatz, is the problem.

    Three starts in four cover the full angle range; the rest start small, the
    near-adiabatic regime that Adam and L-BFGS reach from different basins.  The
    returned energy is an upper bound on the true ceiling, so treat a run that
    beats it as evidence the search missed something, not as an error.
    """
    f = energy_function(problem, depth)
    rng = np.random.default_rng(seed)
    best_params, best_energy = None, np.inf
    for index in range(starts):
        if index % 4 == 3:
            scale = np.exp(rng.uniform(np.log(0.05), np.log(2 * np.pi)))
        else:
            scale = 2 * np.pi
        x0 = rng.uniform(0, scale, (2, depth))
        params, value = minimize(f, x0, steps=steps, lr=lr, method=method)
        if value < best_energy:
            best_params, best_energy = params, float(value)
    return best_params, best_energy


def depth_frontier(problem: Problem, depths=(1, 2, 3, 4), **kwargs) -> dict[int, dict]:
    """Best energy and ratio at each depth - shows the ansatz ceiling.

    If the frontier is still well above the ground energy at the largest depth,
    no amount of optimization will close the gap: only a deeper circuit (or a
    different cost Hamiltonian) can.
    """
    ground = problem.ground
    rows = {}
    for depth in depths:
        params, value = multi_start(problem, depth, **kwargs)
        rows[depth] = {
            "params": params,
            "energy": value,
            "ratio": problem.ratio(value),
            "gap": value - ground,
        }
    return rows


def budget_sweep(problem: Problem, depth: int, budgets=(0, 10, 25, 50, 100, 200, 400),
                 lr: float = 0.02, seed: int = 13, method: str = "adam") -> list[dict]:
    """Ratio as a function of the number of optimizer steps, from one fixed start.

    The distinguishing experiment: a single start improves with budget and then
    flattens against the ceiling, whereas a start pinned to a stationary point
    does not move at all.
    """
    f = energy_function(problem, depth)
    x0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, (2, depth))
    rows = []
    for steps in budgets:
        if steps == 0:
            value = uniform_energy(problem)
        else:
            _, value = minimize(f, x0, steps=steps, lr=lr, method=method)
        rows.append({"steps": steps, "energy": value, "ratio": problem.ratio(value)})
    return rows
