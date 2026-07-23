# Architecture and algorithm design

## Package boundaries

`grid` owns immutable occupancy data and ASCII parsing. `search` implements the
cell graph, while `hybrid` owns heading-state primitives. `collision` provides
geometry checks shared by `postprocess` and `validation`. `timing` converts a
polyline to a segment profile. `cli` composes these modules but contains no
planning algorithm.

All core functions are synchronous, CPU-only, and free of global mutable state.
The standard library is the only runtime dependency.

## Deterministic grid search

Cardinal edges cost one resolution unit and diagonal edges cost `sqrt(2)`
units. A* uses Manhattan distance for cardinal graphs and the consistent octile
heuristic for diagonal graphs. Dijkstra uses the same implementation with a
zero heuristic. Edge, heuristic, accumulated, and frontier-priority costs pass
through the shared finite-arithmetic boundary; an unrepresentable derived cost
is invalid input rather than evidence that a free-cell topology is unreachable.
Heap entries include cost and cell coordinates, update only on
strict cost improvement, and traverse a fixed neighbor order. This makes tie
resolution independent of hash-table iteration.

By default, a diagonal is rejected unless both adjacent cardinal cells are
free. Explicit corner cutting changes the search graph but does not weaken the
continuous validator.

## Heading-aware lattice

Lattice states are `(x_cell, y_cell, heading_bin)`. Each expansion considers a
straight primitive, then one-bin clockwise and counter-clockwise primitives.
A turn must satisfy the hard minimum
`chord_length >= min_turn_radius * heading_change`. Primitive generation and
`lattice_is_kinematically_feasible` use the same predicate, comparing a
one-ULP-down conservative bound of the represented cell-center chord against
that minimum. This replaces any scale-independent absolute allowance; the
public verification tolerance can absorb only secondary curvature-division
rounding and cannot authorize a shorter chord. Each complete accepted chord is
collision checked. A Euclidean heuristic orders states. The result is complete
only over this finite lattice and should not be represented as continuous
Hybrid A*. Radius scaling, primitive steps, chord lengths, accumulated costs,
and priorities are checked for finite representability before entering the
frontier.

## Collision and post-processing

Collision checking does not sample a segment. It enumerates potentially
intersected occupied cells and applies a slab line-versus-closed-box test,
returning the earliest contact. Treating boundaries as occupied is a deliberate
conservative policy that catches diagonal corner contact.

Shortcutting greedily selects the farthest visible future point. Smoothing uses
fixed-order data/smoothness updates. A curvature guard bisects toward the local
neighbor midpoint; the default collision guard rejects updates whose adjacent
segments touch occupancy. Because obstacle and curvature constraints can
conflict, callers should always inspect the final validation report.

## Timing and validation

Polyline curvature is computed from each three-point circumcircle. The local
speed ceiling is `max_speed` only at exact zero curvature. Every positive
finite curvature uses checked division and square-root arithmetic for
`min(max_speed, sqrt(max_lateral_acceleration / curvature))`. If floating-point
rounding would make the resulting ceiling violate
`speed**2 * curvature <= max_lateral_acceleration`, the ceiling is reduced to a
lower representable value before the speed passes. A forward pass limits
acceleration; a backward pass limits deceleration. Each represented reachable
speed is checked against the acceleration implied by its endpoint kinematics.
If square-root rounding would put that acceleration more than one ULP above
the configured limit, the pass uses the next lower representable speed.
Explicit endpoint speeds undergo the same check in both directions. Segment
time follows constant acceleration between nonzero vertex speeds. A
zero-to-zero two-point segment receives an explicit triangular or trapezoidal
sub-profile so total time and peak speed remain physical. Point distances,
lateral-speed ceilings, speed-pass energy terms, segment durations, total arc
length, and elapsed time all use checked finite arithmetic; an unrepresentable
derived profile is invalid input and is never emitted with JSON `Infinity`
values.

Validation independently checks map bounds, zero-length segments, continuous
collision, optional curvature limits, profile geometry, time monotonicity,
peak speed, longitudinal acceleration, and vertex lateral acceleration. Issues
carry stable codes and segment/cell context. Comparisons of calculated values
with caller-supplied upper bounds admit only the exact bound and its immediate
upward representable neighbor, covering unavoidable one-ULP calculation
rounding without a scale-independent tolerance. World-to-cell conversion checks
the coordinate difference, resolution division, and floor conversion before
returning a cell. Collision and trajectory validation use the same checked
finite-geometry boundary, so individually finite inputs whose intermediate
geometry overflows are rejected as malformed rather than leaking arithmetic
exceptions. Non-finite metrics in otherwise structurally inspectable profiles
are reported as JSON `null`, never non-standard `Infinity`.

## Failure model

Malformed domains raise `InvalidInputError`; valid but unreachable searches
raise `NoPathError`. The CLI maps these to distinct status codes. Successful
validation can still return an invalid report, allowing automated callers to
distinguish bad input from a genuine safety finding. In particular, a finite
point whose grid conversion is representable but out of bounds is a validation
finding, while a finite point/resolution/origin combination whose subtraction,
division, distance, or integer conversion is not representable is invalid
input.

The validation CSV schema is checked before row mapping. Trimmed header names
are case-sensitive, non-blank, and unique, with exactly one `x` and one `y`.
Extra unique columns are intentionally ignored for compatibility with planner
output, but every row must have exactly the header width.
