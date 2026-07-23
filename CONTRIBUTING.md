# Contributing

Keep changes deterministic, standard-library-only at runtime, and independently
runnable on Python 3.11 or newer.

1. Add or update a focused test under `tests/`, including a negative case.
2. Run `python -m unittest discover -s tests -v` from the repository root.
3. Run `python -m compileall -q src tests examples`.
4. Exercise any changed CLI path with `python -m trajectory_lab` after adding
   `src` to `PYTHONPATH`.
5. Update the README, architecture document, and changelog for public changes.

Do not add hidden randomness, wall-clock tie breakers, generated caches, or
machine-specific paths. New algorithmic claims must state the graph/model in
which they hold and include deterministic regression fixtures.

