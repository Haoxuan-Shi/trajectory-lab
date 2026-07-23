# Changelog

All notable changes are documented here. The format follows Keep a Changelog,
and versions use semantic versioning.

## [Unreleased]

### Fixed

- Replace fixed absolute endpoint-feasibility and validation allowances with
  representable-neighbor comparisons. Reachable speeds are rounded down when
  their represented segment acceleration would exceed the caller limit by
  more than one ULP, and tiny-scale curvature, speed, longitudinal-, and
  lateral-acceleration violations are no longer certified.
- Remove the scale-independent `1e-12` turning-chord allowance and share a
  one-ULP-conservative hard minimum-distance predicate between state-lattice
  planning and public feasibility verification, so tolerance never approves a
  chord below `min_turn_radius * heading_change` at any grid scale.
- Apply the lateral speed ceiling to every strictly positive finite curvature
  instead of treating values at or below `1e-15` as zero, and conservatively
  round checked ceilings so returned vertex speeds obey their profile's
  lateral-acceleration bound.
- Reject duplicate or blank validation-CSV headers before row mapping, require
  exactly one `x` and `y`, and reject short or extra-field rows while retaining
  deliberate support for ignored extra unique columns.
- Map overflow and unrepresentable finite intermediates in world/cell
  conversion, collision geometry, and trajectory validation to
  `InvalidInputError` and CLI status `2` instead of leaking arithmetic
  exceptions.
- Reject unrepresentable A*/Dijkstra edge, heuristic, accumulated, and
  priority costs as invalid input instead of misclassifying a reachable grid
  as having no path.
- Reject unrepresentable state-lattice radius/primitive arithmetic and all
  non-finite velocity/time derivations with stable `InvalidInputError` output.

## [0.1.0] - 2026-07-23

### Added

- Occupancy grids, deterministic A*/Dijkstra, and heading-aware state lattice.
- Continuous segment collision checks, shortcutting, and guarded smoothing.
- Curvature-, speed-, acceleration-, and lateral-acceleration-aware timing.
- Structured validation, ASCII/CSV fixtures, CLI, tests, and cross-platform CI.
