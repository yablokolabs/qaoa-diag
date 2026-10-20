# Use cases

Eight situations where a ratio alone is not enough, and the shortest call that resolves each.

## 1. You are grading a submission or reading a published number

A reference ratio can come from a run that stopped early. Measure what the depth can reach before you mark anything wrong.

```python
from qaoa_diag import multi_start, vertex_cover

problem = vertex_cover([(0, 1), (1, 2), (0, 2), (2, 3)])
params, ceiling = multi_start(problem, depth=4, starts=25)
print(problem.ground, problem.ratio(ceiling))
```

Read: if a submission scores below `problem.ratio(ceiling)`, it is under-converged and loses a fair comparison. If it scores above, the reference was the weak one. Run `examples/vertex_cover_trap.py` for the full case.

## 2. You are deciding the circuit depth

Extra layers cost hardware time. Find the depth where the ceiling stops moving.

```python
from qaoa_diag import depth_frontier, vertex_cover

problem = vertex_cover([(i, (i + 1) % 6) for i in range(6)])
for depth, row in depth_frontier(problem, depths=(1, 2, 3, 4, 5), starts=12).items():
    print(depth, row["ratio"], row["gap"])
```

Read: the first depth where the gain from an extra layer drops below your budget threshold. `examples/depth_frontier.py` and `examples/max_cut_frontier.py` print the whole table.

## 3. Your ratio is 0.0 or far below what the depth should reach

Check the gradient at the final angles. Zero gradient means the optimizer never started.

```python
import numpy as np
from qaoa_diag import diagnose, vertex_cover

problem = vertex_cover([(0, 1), (1, 2), (0, 2), (2, 3)])
d = diagnose(np.zeros((2, 4)), problem, depth=4)
print(d.verdict, d.grad_norm)
```

Read: `stationary-point` with `grad_norm` near zero. All-zero angles are a fixed point of every gradient method for this ansatz. Randomize over `(0, 2*pi)`. `examples/zero_init_trap.py` shows gradient descent frozen while Adam escapes.

## 4. You wrote a QUBO and want to check it before running QAOA

The ground state of a badly posed penalty is the answer to the wrong problem, and depth cannot fix that.

```python
from qaoa_diag import qubo

problem = qubo(4, pairs=[(0, 1, 2.0)], linear=[(0, -1.0)], const=0.5)
best = problem.best_bits()[0]
print(best, problem.value(best), problem.ground)
```

Read: decode `best_bits` and check the constraints by hand. If the exact optimum violates them, raise the penalty weight. `examples/knapsack_penalty.py` sweeps the weight and shows the feasible answer moving to the bottom of the spectrum.

## 5. You want to compare two models of the same problem

Same graph, different Hamiltonian, different ceiling. The comparison is meaningful only when both are measured the same way.

```python
from qaoa_diag import max_cut, multi_start, vertex_cover

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 0), (4, 2)]
for build in (vertex_cover, max_cut):
    problem = build(edges)
    _, ceiling = multi_start(problem, depth=4, starts=20)
    print(problem.name, problem.ground, problem.ratio(ceiling))
```

Read: compare `ratio`, not energy. The two Hamiltonians have different scales and different ground energies.

## 6. You are about to report one number from one seed

One seed is not a result. Forty seeds of the same run spread from 0.24 to 0.80 on a four-vertex graph.

```bash
python examples/seed_study.py
```

Read: report the ceiling and the share of runs that reached it, or the median with a spread. A single seed tells you about the optimizer, not the algorithm.

## 7. You are teaching how QAOA fails

The three failure modes are concrete and reproducible on a four-qubit graph.

```bash
python examples/zero_init_trap.py
python examples/budget_sweep.py
python examples/depth_frontier.py
```

Read: the same circuit, three different stories. Initialization, budget, and depth each produce a low ratio, and each needs a different fix.

## 8. You do not trust the simulator

Compare against PennyLane on random angles. Both use qubit 0 as the most significant bit, so energies and probabilities must match to machine precision.

```bash
python examples/pennylane_crosscheck.py
```

Read: differences near 1e-15 in both energy and probability mean the numpy path is correct and the diagnostics need no quantum framework installed.

## Command line

```bash
python -m qaoa_diag --edges 0-1,1-2,2-3,3-0 --depth 4 --steps 50
python -m qaoa_diag --edges 0-1,1-2,2-3,3-0 --problem max-cut --depth 3 --frontier --sweep
```

The report prints the exact ground energy, the measured ceiling, the gradient norm, and the verdict with advice.
