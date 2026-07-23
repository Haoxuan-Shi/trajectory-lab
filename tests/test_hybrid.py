from __future__ import annotations

import math
import unittest

import _path  # noqa: F401

from trajectory_lab import (
    InvalidInputError,
    LatticeResult,
    LatticeState,
    NoPathError,
    OccupancyGrid,
    lattice_is_kinematically_feasible,
    state_lattice_plan,
)


class HybridPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = OccupancyGrid.from_rows([[False] * 13 for _ in range(13)])

    def test_straight_lattice_path(self) -> None:
        result = state_lattice_plan(
            self.grid, (1, 6), (10, 6), start_heading=0.0, goal_heading=0.0
        )
        self.assertEqual(result.states[0].heading_bin, 0)
        self.assertEqual(result.states[-1].cell, (10, 6))
        self.assertTrue(all(state.heading_bin == 0 for state in result.states))
        self.assertTrue(lattice_is_kinematically_feasible(result))

    def test_turning_path_respects_discrete_curvature_bound(self) -> None:
        result = state_lattice_plan(
            self.grid,
            (2, 10),
            (10, 2),
            start_heading=0.0,
            goal_heading=-math.pi / 4.0,
            heading_bins=8,
            min_turn_radius=3.0,
        )
        self.assertEqual(result.states[-1].heading_bin, 7)
        self.assertTrue(lattice_is_kinematically_feasible(result))
        self.assertEqual(result, state_lattice_plan(
            self.grid,
            (2, 10),
            (10, 2),
            start_heading=0.0,
            goal_heading=-math.pi / 4.0,
            heading_bins=8,
            min_turn_radius=3.0,
        ))

    def test_tiny_scale_turn_cannot_bypass_required_chord_length(self) -> None:
        grid = OccupancyGrid.from_rows(
            [[False] * 3 for _ in range(3)],
            resolution=1e-15,
        )

        with self.assertRaises(NoPathError):
            state_lattice_plan(
                grid,
                (0, 0),
                (1, 1),
                heading_bins=8,
                min_turn_radius=2e-15,
            )

    def test_turning_profiles_are_strictly_feasible_across_grid_scales(
        self,
    ) -> None:
        for resolution in (1e-15, 1.0, 1e15):
            with self.subTest(resolution=resolution):
                grid = OccupancyGrid.from_rows(
                    [[False] * 13 for _ in range(13)],
                    resolution=resolution,
                )
                result = state_lattice_plan(
                    grid,
                    (2, 10),
                    (10, 2),
                    start_heading=0.0,
                    goal_heading=-math.pi / 4.0,
                    heading_bins=8,
                    min_turn_radius=3.0 * resolution,
                    turn_penalty=0.0,
                )

                self.assertTrue(
                    lattice_is_kinematically_feasible(result, tolerance=0.0)
                )
                angle_step = 2.0 * math.pi / result.heading_bins
                for first, second, first_point, second_point in zip(
                    result.states,
                    result.states[1:],
                    result.path,
                    result.path[1:],
                ):
                    difference = (
                        second.heading_bin - first.heading_bin
                    ) % result.heading_bins
                    difference = min(
                        difference, result.heading_bins - difference
                    )
                    required_distance = (
                        result.min_turn_radius * difference * angle_step
                    )
                    self.assertGreaterEqual(
                        math.dist(first_point, second_point),
                        required_distance,
                    )

    def test_feasibility_never_accepts_an_ulp_short_turn_at_any_scale(
        self,
    ) -> None:
        states = (LatticeState(0, 0, 0), LatticeState(1, 1, 1))
        for resolution in (1e-15, 1.0, 1e15):
            with self.subTest(resolution=resolution):
                radius = 2.0 * resolution
                required_distance = radius * (2.0 * math.pi / 8)
                short_distance = math.nextafter(required_distance, 0.0)
                short_result = LatticeResult(
                    states,
                    ((0.0, 0.0), (short_distance, 0.0)),
                    short_distance,
                    0,
                    8,
                    radius,
                )

                self.assertFalse(lattice_is_kinematically_feasible(short_result))
                self.assertFalse(
                    lattice_is_kinematically_feasible(
                        short_result,
                        tolerance=0.0,
                    )
                )

                sufficient_distance = math.nextafter(
                    required_distance, math.inf
                )
                sufficient_result = LatticeResult(
                    states,
                    ((0.0, 0.0), (sufficient_distance, 0.0)),
                    sufficient_distance,
                    0,
                    8,
                    radius,
                )
                self.assertTrue(
                    lattice_is_kinematically_feasible(
                        sufficient_result,
                        tolerance=0.0,
                    )
                )

    def test_invalid_lattice_parameters_are_rejected(self) -> None:
        with self.assertRaises(InvalidInputError):
            state_lattice_plan(self.grid, (0, 0), (2, 0), heading_bins=3)
        with self.assertRaises(InvalidInputError):
            state_lattice_plan(self.grid, (0, 0), (2, 0), min_turn_radius=0.0)
        with self.assertRaises(InvalidInputError):
            state_lattice_plan(self.grid, (-1, 0), (2, 0))

    def test_unrepresentable_finite_turn_radius_is_invalid_input(self) -> None:
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        grid = OccupancyGrid.from_rows([[False, False], [False, False]])
        with self.assertRaisesRegex(
            InvalidInputError,
            "derived state-lattice geometry or cost is not representable",
        ):
            state_lattice_plan(
                grid,
                (0, 0),
                (1, 1),
                heading_bins=4,
                min_turn_radius=maximum,
            )


if __name__ == "__main__":
    unittest.main()
