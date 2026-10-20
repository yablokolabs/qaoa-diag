"""Turn the optimized state into an answer, then check the answer.

A ratio in [0, 1] tells you how well the algorithm did. It does not tell you
*which* vertex cover it found. Read the most likely bitstring, then check that it
is a real cover, of the right size.

Run: python examples/decode_bitstring.py
"""

import numpy as np

from qaoa_diag import energy_function, minimize, min_vertex_cover, most_likely, probabilities, vertex_cover

edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
depth = 4
problem = vertex_cover(edges)

params0 = np.random.default_rng(0).uniform(0, 2 * np.pi, (2, depth))
params, _ = minimize(energy_function(problem, depth), params0, steps=200, lr=0.05)

probs = probabilities(params, problem, depth)
order = np.argsort(probs)[::-1][:4]

print(problem)
print(f"exact minimum cover: {sorted(min_vertex_cover(edges))}\n")
print("top bitstrings by probability:")
for index in order:
    bits = [(index >> (problem.n - 1 - i)) & 1 for i in range(problem.n)]
    chosen = [i for i, b in enumerate(bits) if b]
    covers = all(i in chosen or j in chosen for i, j in edges)
    print(f"  {''.join(map(str, bits))}  p={probs[index]:.4f}  cover={covers}  vertices={chosen}")

bits, weight = most_likely(params, problem, depth)
chosen = [i for i, b in enumerate(bits) if b]
print(f"\nmost likely: {''.join(map(str, bits))} (p={weight:.4f}), energy {problem.value(bits):.4f}")
print(f"valid cover: {all(i in chosen or j in chosen for i, j in edges)}")
