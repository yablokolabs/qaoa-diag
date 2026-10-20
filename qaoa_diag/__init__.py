"""Diagnostics for small-graph QAOA: exact ground truth, depth ceilings, verdicts."""

from .diagnose import Diagnosis, diagnose
from .optimize import (
    budget_sweep,
    depth_frontier,
    energy_function,
    minimize,
    multi_start,
)
from .problem import Problem, max_cut, min_vertex_cover, qubo, vertex_cover
from .sim import energy, most_likely, probabilities, state, uniform_energy

__version__ = "0.1.0"

__all__ = [
    "Problem",
    "qubo",
    "vertex_cover",
    "max_cut",
    "min_vertex_cover",
    "state",
    "probabilities",
    "energy",
    "uniform_energy",
    "most_likely",
    "minimize",
    "multi_start",
    "depth_frontier",
    "budget_sweep",
    "energy_function",
    "Diagnosis",
    "diagnose",
]
