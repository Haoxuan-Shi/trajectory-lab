from __future__ import annotations

import math
from pathlib import Path
import unittest

import _path  # noqa: F401

from trajectory_lab import (
    InvalidInputError,
    OccupancyGrid,
    curvature_aware_smooth,
    first_path_collision,
    load_ascii_scenario,
    path_is_collision_free,
    polyline_curvatures,
    segment_collision,
    shortcut_path,
    validate_trajectory,
)


FIXTURES = Path(__file__).parent / "fixtures"


class PostprocessCollisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = load_ascii_scenario(FIXTURES / "smoothing_collision.txt").grid
        self.detour = ((0.5, 2.5), (3.5, 0.5), (6.5, 2.5))

    def test_continuous_check_detects_obstacle_between_endpoints(self) -> None:
        hit = first_path_collision(self.grid, ((0.5, 2.5), (6.5, 2.5)))
        self.assertIsNotNone(hit)
        self.assertEqual(hit.cell, (3, 2))
        self.assertEqual(hit.segment_index, 0)

    def test_closed_boundary_contact_at_endpoint_is_collision(self) -> None:
        grid = OccupancyGrid.from_rows([[True, False]])
        path = ((1.5, 0.5), (1.0, 0.5))

        hit = segment_collision(grid, *path)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.cell, (0, 0))
        self.assertEqual(hit.fraction, 1.0)
        self.assertFalse(validate_trajectory(grid, path).valid)

    def test_nextafter_scale_entry_into_occupied_cell_is_collision(self) -> None:
        grid = OccupancyGrid.from_rows(
            [
                [False, False],
                [False, True],
            ]
        )
        path = (
            (math.nextafter(1.0, 0.0), 0.5),
            (math.nextafter(1.0, math.inf), 1.5),
        )

        hit = segment_collision(grid, *path)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.cell, (1, 1))
        self.assertLess(hit.fraction, 1.0)
        report = validate_trajectory(grid, path)
        self.assertFalse(report.valid)
        self.assertIn("collision", {issue.code for issue in report.issues})

    def test_one_ulp_corner_miss_is_collision_free(self) -> None:
        grid = OccupancyGrid.from_rows(
            [
                [False, False, False, False, False],
                [False, True, False, False, False],
                [False, False, False, False, False],
            ]
        )

        hit = segment_collision(
            grid,
            (0.0, 0.0),
            (4.0, math.nextafter(2.0, 0.0)),
        )

        self.assertIsNone(hit)

    def test_representable_diagonal_clearance_is_collision_free(self) -> None:
        grid = OccupancyGrid.from_rows(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ]
        )
        clearance = 2.0 - 2.0**-50

        hit = segment_collision(
            grid,
            (0.0, clearance),
            (clearance, 0.0),
        )

        self.assertIsNone(hit)

    def test_exact_corner_contact_remains_collision(self) -> None:
        grid = OccupancyGrid.from_rows(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ]
        )

        hit = segment_collision(grid, (0.0, 2.0), (2.0, 0.0))

        self.assertIsNotNone(hit)
        self.assertEqual(hit.cell, (1, 1))
        self.assertEqual(hit.point, (1.0, 1.0))

    def test_occupied_segment_endpoints_never_validate(self) -> None:
        grid = OccupancyGrid.from_rows([[False, True]])
        for path in (
            ((0.5, 0.5), (1.5, 0.5)),
            ((1.5, 0.5), (0.5, 0.5)),
        ):
            with self.subTest(path=path):
                report = validate_trajectory(grid, path)
                self.assertFalse(report.valid)
                self.assertIn("collision", {issue.code for issue in report.issues})

    def test_collision_rejects_unrepresentable_finite_world_geometry(self) -> None:
        minimum_subnormal = math.nextafter(0.0, 1.0)
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        cases = (
            (
                OccupancyGrid.from_rows(
                    [[False]], resolution=minimum_subnormal
                ),
                (1.0, 0.0),
                (2.0, 0.0),
            ),
            (
                OccupancyGrid.from_rows([[False]]),
                (-maximum, 0.5),
                (maximum, 0.5),
            ),
        )
        for grid, start, end in cases:
            with self.subTest(grid=grid, start=start, end=end):
                with self.assertRaises(InvalidInputError):
                    segment_collision(grid, start, end)

    def test_line_of_sight_shortcut_keeps_required_detour(self) -> None:
        shortened = shortcut_path(self.grid, self.detour)
        self.assertEqual(shortened, self.detour)
        self.assertTrue(path_is_collision_free(self.grid, shortened))

    def test_collision_after_unguarded_smoothing_is_reported(self) -> None:
        unsafe = curvature_aware_smooth(
            self.grid,
            self.detour,
            data_weight=0.01,
            smooth_weight=0.49,
            iterations=80,
            preserve_collision_free=False,
        )
        report = validate_trajectory(self.grid, unsafe)
        self.assertFalse(report.valid)
        self.assertIn("collision", {issue.code for issue in report.issues})

    def test_collision_guard_preserves_feasibility_and_determinism(self) -> None:
        first = curvature_aware_smooth(
            self.grid,
            self.detour,
            data_weight=0.01,
            smooth_weight=0.49,
            iterations=80,
        )
        second = curvature_aware_smooth(
            self.grid,
            self.detour,
            data_weight=0.01,
            smooth_weight=0.49,
            iterations=80,
        )
        self.assertEqual(first, second)
        self.assertTrue(path_is_collision_free(self.grid, first))

    def test_curvature_limiter_reduces_a_sharp_corner(self) -> None:
        open_grid = OccupancyGrid.from_rows([[False] * 10 for _ in range(10)])
        path = ((1.0, 1.0), (1.0, 6.0), (2.0, 6.0), (7.0, 6.0))
        original_max = max(polyline_curvatures(path))
        smoothed = curvature_aware_smooth(
            open_grid,
            path,
            max_curvature=0.35,
            iterations=200,
        )
        self.assertLess(max(polyline_curvatures(smoothed)), original_max)


if __name__ == "__main__":
    unittest.main()
