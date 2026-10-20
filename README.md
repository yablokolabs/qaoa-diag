# qaoa-diag

A QAOA approximation ratio below 1 has four different causes. This library tells you which one you have, in one call.

```bash
pip install -e .
python examples/quickstart.py
```

```python
from qaoa_diag import diagnose, multi_start, vertex_cover

problem = vertex_cover([(0, 1), (1, 2), (2, 3), (3, 0)])
params, ceiling = multi_start(problem, depth=4)
print(diagnose(params, problem, depth=4, ceiling=ceiling).verdict)
```

## The four causes

| Verdict | What happened | What to do |
| --- | --- | --- |
| `optimal` | Reached the exact ground energy | Nothing |
| `ansatz-limited` | Hit the best energy the search found for this depth, still above the exact optimum | Add layers or change the cost Hamiltonian |
| `stationary-point` | Gradient is zero, usually from all-zero starting angles | Randomize the angles over `(0, 2*pi)` |
| `under-converged` | Stopped short of an energy this depth did reach | Raise steps, learning rate, or restarts |

The ratio on its own cannot separate these. A verdict can, because it compares the run against two things it computes itself: the exact ground energy for graphs small enough to diagonalize, and the best energy this depth reached over many random starts.

## What it computes

1. **Exact ground truth.** The cost Hamiltonian is diagonal, so the whole spectrum is a table of `2**n` numbers. `problem.ground` is the true optimum, not an estimate.
2. **Measured depth ceiling.** `multi_start` runs the optimizer from many starts and keeps the best energy. That is the reference a well-optimized single run should approach.
3. **Verdict.** `diagnose` compares the run to both and reports which is binding, plus the gradient norm at the final angles.

## Use cases

1. **Grading a submission or a reference value.** Check whether the published number is the optimum or an under-converged run before you compare against it. See `examples/vertex_cover_trap.py`.
2. **Choosing a depth before hardware time.** `depth_frontier` shows where extra layers stop paying. See `examples/max_cut_frontier.py`.
3. **Debugging a ratio of 0.0.** A zero or symmetric initialization is a stationary point, so every gradient method stays there. See `examples/zero_init_trap.py`.
4. **Validating a QUBO before optimizing it.** A penalty weight that is too small puts an infeasible answer at the bottom of the spectrum, and no circuit depth repairs that. See `examples/knapsack_penalty.py`.
5. **Reporting results you can defend.** One seed is not a result: 40 seeds of the same run spread from 0.24 to 0.80. See `examples/seed_study.py`.

Full write-up with code for each: [docs/use_cases.md](docs/use_cases.md).

## Real output

```
$ python -m qaoa_diag --edges 0-1,1-2,0-2,2-3 --depth 4 --steps 50
<Problem 'vertex-cover': n=4, ground=-3>

energy         -1.701496
exact ground   -3.000000
ratio          0.567165
depth ceiling  -2.694950  (ratio 0.898317)
|gradient|     1.334e+00
verdict        under-converged
advice         The ceiling is reachable but this run stopped short of it: raise the
               step budget or the learning rate, or take the best of several starts.
```

A batch over several graphs at depth 4, 10 starts each:

```
graph           n   m   ground  ceiling   ratio  verdict
path-4          4   3   -2.250   -2.250   1.000  optimal
cycle-4         4   4   -3.000   -3.000   1.000  optimal
cycle-6         6   6   -4.500   -4.171   0.927  ansatz-limited
triangle+tail   4   4   -3.000   -2.695   0.898  ansatz-limited
complete-4      4   6   -2.500   -2.500   1.000  optimal
grid-2x3        6   7   -5.250   -4.339   0.827  ansatz-limited
```

## Examples

Start here:

| Script | Shows |
| --- | --- |
| `quickstart.py` | Build a problem, optimize, read the verdict |
| `budget_sweep.py` | Ratio against optimizer steps, and where it flattens |
| `depth_frontier.py` | Best energy at depths 1 to 5 for one graph |

Diagnosing a run:

| Script | Shows |
| --- | --- |
| `zero_init_trap.py` | A ratio of 0.0 from initialization, fixed by random starts |
| `vertex_cover_trap.py` | A reference value that is not the optimum |
| `beats_the_ceiling.py` | What a measured ceiling does and does not prove |
| `seed_study.py` | Spread over 40 seeds versus the ceiling |
| `decode_bitstring.py` | Turn the state into an answer, then check the answer |

Other problems as Hamiltonians:

| Script | Shows |
| --- | --- |
| `max_cut_frontier.py` | Max-Cut on an odd cycle, which cannot be cut edge for edge |
| `random_ising.py` | Frustrated couplings, where depth stops helping |
| `knapsack_penalty.py` | How the penalty weight decides the answer |
| `custom_qubo.py` | A packing problem written as a QUBO |
| `number_partition.py` | Exact ground truth on a quartic penalty landscape |
| `graph_coloring.py` | Constraint satisfaction, decoded and checked |
| `portfolio_qubo.py` | Risk and return over four assets |

Reporting and cross-checks:

| Script | Shows |
| --- | --- |
| `batch_graphs.py` | One table over seven graphs |
| `markdown_report.py` | Markdown table for a README or PR |
| `pennylane_crosscheck.py` | The numpy simulator agrees with PennyLane to 1e-15 |

## API

| Name | Purpose |
| --- | --- |
| `vertex_cover(edges)`, `max_cut(edges)`, `qubo(n, pairs, linear, const)` | Build a diagonal cost Hamiltonian |
| `problem.ground`, `.best_bits()`, `.ratio(energy)` | Exact reference values |
| `energy(params, problem, depth)` | Expectation value of the cost Hamiltonian |
| `probabilities(...)`, `most_likely(...)` | Measurement distribution, top bitstring |
| `minimize(f, x0, steps, lr, method)` | One optimizer run (`adam`, `gd`, or `lbfgs`) |
| `multi_start(problem, depth, starts)` | The measured depth ceiling |
| `diagnose(params, problem, depth)` | The verdict, with the ceiling and gradient norm |
| `depth_frontier(...)`, `budget_sweep(...)` | The two sweep experiments |

Runs on numpy alone. `method="lbfgs"` uses scipy when it is installed.

## Limits

1. **Small problems only.** The exact table is `2**n`, so up to about 12 qubits for comfortable runs. Above that, the ground truth is the expensive part, not the simulation.
2. **One ansatz.** Cost layer plus `sum_i X_i` mixer, angles `(gammas, betas)` per layer. Other mixers and constrained ansatze are not wired in.
3. **A ceiling is a measurement.** It says no run we tried did better. Widen the search before you claim a depth cannot do it: `beats_the_ceiling.py` shows a ceiling moving from 0.57 to 0.92 with more starts.
4. **No noise model.** Statevector simulation, infinite shots. Shot noise changes optimizer behaviour and is not modelled here.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

18 tests, about a minute. They pin the algebra of every Hamiltonian builder, the bit ordering against PennyLane, the stationary-point and under-converged verdicts, and the sign-aware ratio.

## License

MIT.
