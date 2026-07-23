# trajectory-lab

[English](README.md) | [简体中文](README.zh-CN.md)

`trajectory-lab` is an independent Python 3.11+ toolkit for deterministic,
CPU-only 2D path planning, post-processing, time parameterization, and
validation on occupancy grids. It uses only the Python standard library and
does not import or require any sibling repository.

It includes:

- rectangular occupancy grids and human-readable ASCII map fixtures;
- deterministic optimal A* and Dijkstra search with configurable diagonal
  motion and explicit corner-cutting policy;
- heading-aware state-lattice planning with bounded discrete curvature;
- exact line-segment versus occupied-cell collision tests;
- greedy line-of-sight shortcutting and collision-guarded,
  curvature-aware smoothing;
- forward/backward velocity and time parameterization under speed,
  longitudinal-acceleration, and lateral-acceleration limits; and
- structured validation reports plus `plan` and `validate` CLI commands.

The package is designed for education, simulation, and reproducible planning
experiments. It is not a certified robotics safety component.

## Quickstart on Windows

No installation is needed for source-tree use:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
python -m trajectory_lab plan `
  --map examples\maps\warehouse.txt `
  --algorithm astar --shortcut `
  --output path.csv --report report.json
python -m trajectory_lab validate `
  --map tests\fixtures\smoothing_collision.txt `
  --path examples\collision_path.csv --report -
```

Alternatively, install the console command without runtime dependencies:

```powershell
python -m pip install --no-deps --no-build-isolation -e .
trajectory-lab --help
```

The first command writes CSV columns `x,y,arc_length,speed,time,curvature`.
Exit status is `0` for a valid result, `2` for invalid input, `3` when no path
exists, and `4` for a completed validation that found an unsafe/infeasible
trajectory.

## ASCII maps

Maps are rectangular text files:

- `#` is occupied;
- `.` or a space is free;
- `S` and `G` are optional free start and goal cells.

Cell `(0, 0)` is the top-left character. Grid/world `x` increases right and
`y` increases down. A command can override fixture endpoints with
`--start X,Y --goal X,Y`.

## Path CSV input

`validate --path` reads a header before mapping any row. Header names are
trimmed, case-sensitive, non-blank, and unique after trimming; the file must
contain exactly one `x` and one `y` column. Additional uniquely named columns
are deliberately accepted and ignored so a CSV produced by `plan` can be
validated directly. Every data row must nevertheless have exactly the same
number of fields as the header: missing fields, trailing/extra fields, blank
rows, malformed CSV, and non-finite coordinates are invalid input and map to
exit status `2`.

## Python API example

```python
from trajectory_lab import (
    MotionLimits, astar, cells_to_points, load_ascii_scenario,
    parameterize_velocity, validate_trajectory,
)

scenario = load_ascii_scenario("examples/maps/warehouse.txt")
search = astar(scenario.grid, scenario.start, scenario.goal)
path = cells_to_points(scenario.grid, search.path)
profile = parameterize_velocity(path, MotionLimits(4.0, 1.5, 2.0))
report = validate_trajectory(scenario.grid, path, profile=profile)
assert report.valid
```

See [`examples/plan_example.py`](examples/plan_example.py) for a complete
script and [`docs/architecture.md`](docs/architecture.md) for algorithmic and
safety boundaries.

## Honest scope and determinism

A* and Dijkstra are optimal for the configured grid neighbor graph and edge
costs. The state-lattice planner is resolution-complete only with respect to
its finite primitive/state set; it is not continuous-space Hybrid A*. The
grid planners reject finite inputs whose derived heuristic, edge, accumulated,
or queue-priority cost cannot be represented, instead of misreporting the
topology as unreachable. State-lattice geometry/costs and velocity/time
parameterization use the same fail-closed finite-arithmetic boundary. The
state-lattice planner and public feasibility checker also share a
one-ULP-conservative turning-chord predicate: verification tolerance can never
authorize a chord shorter than `min_turn_radius * heading_change`, regardless
of grid scale. The timing API treats only exact zero curvature as unconstrained
by lateral acceleration. Every positive finite curvature, however small, uses
checked lateral-speed arithmetic, and its representable speed ceiling is
reduced when necessary so
`speed**2 * curvature <= max_lateral_acceleration`; an
unrepresentable ceiling is rejected as invalid input. Forward/backward speed
passes also verify the acceleration represented by each candidate speed and
step an upward-rounded boundary down to a safe float neighbor. Endpoint
requests that need more than the configured acceleration are rejected at every
scale. Validation upper-bound checks use no fixed absolute allowance: a
derived value may equal the caller's bound or its immediate upward
representable neighbor to cover one-ULP calculation rounding, but any larger
value is reported as a violation. The smoother is a local
iterative method and may retain a curvature violation when
obstacles leave no feasible update--the validation report makes that explicit.
Velocity limits apply to the returned piecewise segment profile and do not
model actuator lag or vehicle dynamics.

Tie-breaking, neighbor ordering, primitive ordering, smoothing order, and
iteration counts are fixed. Identical inputs on a compatible Python runtime
produce identical paths and serialized outputs. The toolkit uses no random
numbers, clocks, network calls, cloud services, GPU, or hardware.
