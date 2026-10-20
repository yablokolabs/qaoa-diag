"""Explain *why* a QAOA run fell short of the exact optimum.

A disappointing approximation ratio has three very different causes, and they
are routinely confused:

``optimal``          the run reached the true ground energy - nothing to fix.
``ansatz-limited``   the run reached the best energy this circuit *can* reach,
                     which is still above the ground energy.  More optimizer
                     steps or restarts cannot help; only a deeper circuit or a
                     different cost Hamiltonian can.
``stationary-point`` the parameters sit where the gradient vanishes - typically
                     a zero or symmetry-preserving initialisation, which is a
                     fixed point of every gradient method.
``under-converged``  the run stopped short of the ceiling: it needs a bigger
                     budget (steps, learning rate, or restarts).

The ceiling is obtained by :func:`qaoa_diag.optimize.multi_start`, so the verdict
is measured, not guessed.  A measured ceiling is a lower bound: when a run beats
it, the run is the better evidence and the verdict falls back to comparing the
run against the exact ground energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimize import energy_function, gradient, multi_start
from .problem import Problem
from .sim import energy

__all__ = ["Diagnosis", "diagnose", "VERDICTS"]

VERDICTS = ("optimal", "ansatz-limited", "stationary-point", "under-converged")

_ADVICE = {
    "optimal": "Nothing to fix: the run reached the exact ground energy.",
    "ansatz-limited": (
        "At the ceiling for this depth; more optimization cannot help. "
        "Increase the depth or change the cost Hamiltonian."
    ),
    "stationary-point": (
        "Gradient vanishes here, so the optimizer cannot move: this is the "
        "usual symptom of a zero or symmetric initialisation. Randomise the "
        "initial angles over the full range (0, 2*pi) or restart from elsewhere."
    ),
    "under-converged": (
        "The ceiling is reachable but this run stopped short of it: raise the "
        "step budget or the learning rate, or take the best of several starts."
    ),
}


@dataclass
class Diagnosis:
    """Outcome of :func:`diagnose`."""

    energy: float
    ground: float
    ratio: float
    ceiling: float
    ceiling_ratio: float
    grad_norm: float
    verdict: str
    advice: str

    def __str__(self) -> str:
        return (
            f"energy         {self.energy:.6f}\n"
            f"exact ground   {self.ground:.6f}\n"
            f"ratio          {self.ratio:.6f}\n"
            f"depth ceiling  {self.ceiling:.6f}  (ratio {self.ceiling_ratio:.6f})\n"
            f"|gradient|     {self.grad_norm:.3e}\n"
            f"verdict        {self.verdict}\n"
            f"advice         {self.advice}"
        )


def diagnose(
    params,
    problem: Problem,
    depth: int,
    starts: int = 20,
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
    tol: float = 1e-3,
    grad_tol: float = 1e-6,
    method: str = "lbfgs",
    ceiling: float | None = None,
) -> Diagnosis:
    """Classify a QAOA run against the exact ground energy and the depth ceiling.

    Pass ``ceiling`` when you already measured it (the energy
    :func:`~qaoa_diag.optimize.multi_start` returned) to skip recomputing it.
    """
    params = np.asarray(params, dtype=float).reshape(2, depth)
    value = energy(params, problem, depth)

    if ceiling is None:
        _, ceiling = multi_start(
            problem, depth, starts=starts, steps=steps, lr=lr, seed=seed, method=method
        )
    # ``value`` is itself achievable, so it bounds the ceiling from above.
    ceiling = float(min(ceiling, value))
    grad_norm = float(np.linalg.norm(gradient(energy_function(problem, depth), params.ravel())))

    scale = max(abs(problem.ground), 1e-12)
    if value - problem.ground <= tol * scale:
        verdict = "optimal"
    elif value - ceiling <= tol * scale:
        verdict = "ansatz-limited"
    elif grad_norm <= grad_tol:
        verdict = "stationary-point"
    else:
        verdict = "under-converged"

    return Diagnosis(
        energy=float(value),
        ground=problem.ground,
        ratio=problem.ratio(value),
        ceiling=ceiling,
        ceiling_ratio=problem.ratio(ceiling),
        grad_norm=grad_norm,
        verdict=verdict,
        advice=_ADVICE[verdict],
    )
