from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import unittest

import _path  # noqa: F401

from trajectory_lab import (
    InvalidInputError,
    MotionLimits,
    OccupancyGrid,
    load_ascii_scenario,
    parameterize_velocity,
    validate_trajectory,
)


FIXTURES = Path(__file__).parent / "fixtures"


class TimingValidationTests(unittest.TestCase):
    def test_parameterization_obeys_speed_acceleration_and_lateral_limits(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 12 for _ in range(12)])
        path = ((1.5, 1.5), (4.5, 1.5), (6.5, 3.5), (9.5, 3.5))
        limits = MotionLimits(3.0, 1.25, 1.0)
        profile = parameterize_velocity(path, limits, start_speed=0.0, end_speed=0.0)
        report = validate_trajectory(grid, path, profile=profile)
        self.assertTrue(report.valid, report.issues)
        self.assertLessEqual(
            report.max_speed, math.nextafter(limits.max_speed, math.inf)
        )
        self.assertLessEqual(
            report.max_abs_acceleration,
            math.nextafter(limits.max_acceleration, math.inf),
        )
        self.assertLessEqual(
            report.max_lateral_acceleration,
            math.nextafter(limits.max_lateral_acceleration, math.inf),
        )
        self.assertGreater(profile.total_time, 0.0)
        self.assertTrue(
            all(
                later.time > earlier.time
                for earlier, later in zip(profile.points, profile.points[1:])
            )
        )

    def test_parameterization_limits_tiny_positive_curvature(self) -> None:
        path = ((0.0, 0.0), (1e8, 0.0), (2e8, 1.0))
        limits = MotionLimits(
            max_speed=1e9,
            max_acceleration=1e10,
            max_lateral_acceleration=2.5,
        )

        profile = parameterize_velocity(path, limits)

        self.assertGreater(profile.points[1].curvature, 0.0)
        self.assertLess(profile.points[1].speed, limits.max_speed)
        for index, point in enumerate(profile.points):
            lateral_acceleration = point.speed * point.speed * point.curvature
            self.assertLessEqual(
                lateral_acceleration,
                profile.limits.max_lateral_acceleration,
                f"point {index} exceeds its profile's lateral bound",
            )

    def test_parameterization_checks_subnormal_positive_curvature_arithmetic(
        self,
    ) -> None:
        minimum_subnormal = math.nextafter(0.0, 1.0)
        path = ((0.0, 0.0), (1.0, 0.0), (2.0, minimum_subnormal))

        with self.assertRaisesRegex(
            InvalidInputError,
            "derived trajectory timing is not representable",
        ):
            parameterize_velocity(path)

    def test_parameterization_rejects_tiny_endpoint_acceleration_violation(
        self,
    ) -> None:
        path = ((0.0, 0.0), (2e-12, 0.0))
        limits = MotionLimits(
            max_speed=1.0,
            max_acceleration=1e-12,
            max_lateral_acceleration=1.0,
        )

        with self.assertRaisesRegex(
            InvalidInputError,
            "path is too short to reach end_speed",
        ):
            parameterize_velocity(path, limits, end_speed=5e-11)

    def test_parameterization_rejects_tiny_start_deceleration_violation(
        self,
    ) -> None:
        path = ((0.0, 0.0), (2e-12, 0.0))
        limits = MotionLimits(
            max_speed=1.0,
            max_acceleration=1e-12,
            max_lateral_acceleration=1.0,
        )

        with self.assertRaisesRegex(
            InvalidInputError,
            "path is too short to decelerate from start_speed",
        ):
            parameterize_velocity(path, limits, start_speed=5e-11)

    def test_endpoint_speed_boundary_does_not_hide_multi_ulp_acceleration(
        self,
    ) -> None:
        path = ((0.0, 0.0), (1.0, 0.0))
        limits = MotionLimits(3.0, 2.0, 1.0)

        boundary = parameterize_velocity(path, limits, end_speed=2.0)
        self.assertEqual(boundary.segments[0].max_abs_acceleration, 2.0)

        with self.assertRaisesRegex(
            InvalidInputError,
            "path is too short to reach end_speed",
        ):
            parameterize_velocity(
                path,
                limits,
                end_speed=math.nextafter(2.0, math.inf),
            )

    def test_reachable_speed_uses_safe_representable_boundary_neighbor(
        self,
    ) -> None:
        start_speed = 1.239776611328125e-6
        distance = 1.239776611328125e-6
        acceleration_limit = 1.7
        limits = MotionLimits(1.0, acceleration_limit, 1.0)
        path = ((0.0, 0.0), (distance, 0.0))
        rounded_reachable = math.sqrt(
            start_speed * start_speed
            + 2.0 * acceleration_limit * distance
        )
        safe_neighbor = math.nextafter(rounded_reachable, 0.0)

        profile = parameterize_velocity(
            path,
            limits,
            start_speed=start_speed,
            end_speed=safe_neighbor,
        )
        self.assertLessEqual(
            profile.segments[0].max_abs_acceleration,
            math.nextafter(acceleration_limit, math.inf),
        )

        with self.assertRaisesRegex(
            InvalidInputError,
            "path is too short to reach end_speed",
        ):
            parameterize_velocity(
                path,
                limits,
                start_speed=start_speed,
                end_speed=rounded_reachable,
            )

    def test_validation_derives_lateral_acceleration_from_path_geometry(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 4 for _ in range(4)])
        path = ((0.5, 0.5), (1.5, 0.5), (1.5, 1.5))
        profile = parameterize_velocity(
            path,
            MotionLimits(2.0, 10.0, 100.0),
            start_speed=0.0,
            end_speed=0.0,
        )
        forged_points = tuple(
            replace(point, curvature=0.0) for point in profile.points
        )
        forged = replace(profile, points=forged_points)

        report = validate_trajectory(
            grid,
            path,
            profile=forged,
            limits=MotionLimits(2.0, 10.0, 1.0),
        )

        self.assertIn(
            "lateral_acceleration_limit", {issue.code for issue in report.issues}
        )
        self.assertGreater(report.max_lateral_acceleration, 1.0)

    def test_validation_rejects_tiny_curvature_above_caller_limit(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 3])
        path = ((0.5, 0.5), (1.5, 0.5), (2.5, 0.5000000005))

        report = validate_trajectory(grid, path, max_curvature=1e-12)

        self.assertGreater(report.max_curvature, 1e-12)
        self.assertIn("curvature_limit", {issue.code for issue in report.issues})

    def test_curvature_limit_allows_only_one_ulp_of_calculation_rounding(
        self,
    ) -> None:
        grid = OccupancyGrid.from_rows([[False] * 3])
        path = ((0.5, 0.5), (1.5, 0.5), (2.5, 0.5000000005))
        measured = validate_trajectory(grid, path).max_curvature
        one_ulp_below = math.nextafter(measured, 0.0)
        two_ulps_below = math.nextafter(one_ulp_below, 0.0)

        for name, limit in (
            ("exact boundary", measured),
            ("one-ULP calculation allowance", one_ulp_below),
        ):
            with self.subTest(name=name):
                report = validate_trajectory(grid, path, max_curvature=limit)
                self.assertNotIn(
                    "curvature_limit",
                    {issue.code for issue in report.issues},
                )

        report = validate_trajectory(grid, path, max_curvature=two_ulps_below)
        self.assertIn("curvature_limit", {issue.code for issue in report.issues})

    def test_validation_rejects_tiny_endpoint_acceleration_above_limit(
        self,
    ) -> None:
        grid = OccupancyGrid.from_rows([[False] * 3], resolution=1e-12)
        path = ((0.0, 0.0), (2e-12, 0.0))
        profile = parameterize_velocity(
            path,
            MotionLimits(1.0, 1.0, 1.0),
            end_speed=5e-11,
        )
        caller_limits = MotionLimits(1.0, 1e-12, 1.0)

        report = validate_trajectory(
            grid,
            path,
            profile=profile,
            limits=caller_limits,
        )

        self.assertGreater(
            report.max_abs_acceleration,
            caller_limits.max_acceleration,
        )
        self.assertIn("acceleration_limit", {issue.code for issue in report.issues})

    def test_acceleration_limit_allows_only_one_ulp_of_calculation_rounding(
        self,
    ) -> None:
        grid = OccupancyGrid.from_rows([[False] * 2])
        path = ((0.0, 0.0), (1.0, 0.0))
        profile = parameterize_velocity(
            path,
            MotionLimits(3.0, 2.0, 1.0),
            end_speed=2.0,
        )
        measured = profile.segments[0].max_abs_acceleration
        one_ulp_below = math.nextafter(measured, 0.0)
        two_ulps_below = math.nextafter(one_ulp_below, 0.0)

        for name, limit in (
            ("exact boundary", measured),
            ("one-ULP calculation allowance", one_ulp_below),
        ):
            with self.subTest(name=name):
                report = validate_trajectory(
                    grid,
                    path,
                    profile=profile,
                    limits=MotionLimits(3.0, limit, 1.0),
                )
                self.assertNotIn(
                    "acceleration_limit",
                    {issue.code for issue in report.issues},
                )

        report = validate_trajectory(
            grid,
            path,
            profile=profile,
            limits=MotionLimits(3.0, two_ulps_below, 1.0),
        )
        self.assertIn("acceleration_limit", {issue.code for issue in report.issues})

    def test_validation_rejects_timed_curvature_that_disagrees_with_geometry(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 4 for _ in range(4)])
        path = ((0.5, 0.5), (1.5, 0.5), (1.5, 1.5))
        profile = parameterize_velocity(path, MotionLimits(2.0, 10.0, 100.0))
        forged = replace(
            profile,
            points=tuple(replace(point, curvature=0.0) for point in profile.points),
        )

        report = validate_trajectory(grid, path, profile=forged)

        self.assertIn("profile_curvature", {issue.code for issue in report.issues})

    def test_validation_binds_each_timed_point_to_geometric_arc_length(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 6 for _ in range(3)])
        path = ((0.5, 0.5), (2.5, 0.5), (5.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_points = list(profile.points)
        forged_points[1] = replace(
            forged_points[1], arc_length=forged_points[1].arc_length + 0.25
        )

        report = validate_trajectory(
            grid, path, profile=replace(profile, points=tuple(forged_points))
        )

        self.assertIn("profile_arc_length", {issue.code for issue in report.issues})

    def test_validation_binds_profile_total_length_to_path_geometry(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))

        report = validate_trajectory(
            grid, path, profile=replace(profile, total_length=profile.total_length + 1.0)
        )

        self.assertIn("profile_total_length", {issue.code for issue in report.issues})

    def test_validation_binds_segment_indices_to_sequence_order(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_segments = list(profile.segments)
        forged_segments[0] = replace(forged_segments[0], index=7)

        report = validate_trajectory(
            grid, path, profile=replace(profile, segments=tuple(forged_segments))
        )

        self.assertIn("segment_index", {issue.code for issue in report.issues})

    def test_validation_binds_segment_lengths_to_path_geometry(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_segments = list(profile.segments)
        forged_segments[1] = replace(forged_segments[1], length=0.25)

        report = validate_trajectory(
            grid, path, profile=replace(profile, segments=tuple(forged_segments))
        )

        self.assertIn("segment_length", {issue.code for issue in report.issues})

    def test_validation_binds_segment_endpoint_speeds_to_timed_points(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_segments = list(profile.segments)
        forged_segments[0] = replace(forged_segments[0], start_speed=0.5)

        report = validate_trajectory(
            grid, path, profile=replace(profile, segments=tuple(forged_segments))
        )

        self.assertIn("segment_speed", {issue.code for issue in report.issues})

    def test_validation_binds_segment_duration_to_timed_point_delta(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_segments = list(profile.segments)
        forged_segments[0] = replace(
            forged_segments[0], duration=forged_segments[0].duration + 0.5
        )

        report = validate_trajectory(
            grid, path, profile=replace(profile, segments=tuple(forged_segments))
        )

        self.assertIn("segment_duration", {issue.code for issue in report.issues})

    def test_validation_binds_total_time_to_points_and_segments(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))

        report = validate_trajectory(
            grid, path, profile=replace(profile, total_time=profile.total_time + 1.0)
        )

        self.assertIn("profile_total_time", {issue.code for issue in report.issues})

    def test_validation_rejects_malformed_profile_fields_without_crashing(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_points = list(profile.points)
        forged_points[0] = replace(forged_points[0], speed="fast")

        report = validate_trajectory(
            grid, path, profile=replace(profile, points=tuple(forged_points))
        )

        self.assertFalse(report.valid)
        self.assertIn("malformed_profile", {issue.code for issue in report.issues})

    def test_validation_rejects_malformed_point_time_without_crashing(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_points = list(profile.points)
        forged_points[0] = replace(forged_points[0], time="later")

        report = validate_trajectory(
            grid, path, profile=replace(profile, points=tuple(forged_points))
        )

        self.assertFalse(report.valid)
        self.assertIn("malformed_profile", {issue.code for issue in report.issues})

    def test_validation_rejects_negative_segment_fields(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_segments = list(profile.segments)
        forged_segments[0] = replace(forged_segments[0], peak_speed=-0.5)

        report = validate_trajectory(
            grid, path, profile=replace(profile, segments=tuple(forged_segments))
        )

        self.assertFalse(report.valid)
        self.assertIn("negative_profile", {issue.code for issue in report.issues})

    def test_validation_rejects_peak_speed_inconsistent_with_segment_kinematics(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5])
        path = ((0.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(10.0, 2.0, 2.0))
        forged = replace(
            profile,
            segments=(replace(profile.segments[0], peak_speed=2.0),),
        )

        report = validate_trajectory(grid, path, profile=forged)

        self.assertFalse(report.valid)
        self.assertIn("segment_kinematics", {issue.code for issue in report.issues})

    def test_validation_rejects_malformed_segment_fields_without_crashing(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_segments = list(profile.segments)
        forged_segments[0] = replace(forged_segments[0], duration="soon")

        report = validate_trajectory(
            grid, path, profile=replace(profile, segments=tuple(forged_segments))
        )

        self.assertFalse(report.valid)
        self.assertIn("malformed_profile", {issue.code for issue in report.issues})

    def test_validation_rejects_malformed_segment_profile_label(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5])
        path = ((0.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(10.0, 2.0, 2.0))
        forged = replace(
            profile,
            segments=(replace(profile.segments[0], profile=[]),),
        )

        report = validate_trajectory(grid, path, profile=forged)

        self.assertFalse(report.valid)
        self.assertIn("malformed_profile", {issue.code for issue in report.issues})

    def test_validation_rejects_malformed_profile_totals_without_crashing(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))

        report = validate_trajectory(
            grid, path, profile=replace(profile, total_length="four")
        )

        self.assertFalse(report.valid)
        self.assertIn("malformed_profile", {issue.code for issue in report.issues})

    def test_validation_rejects_extra_segments_without_index_error(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged = replace(profile, segments=profile.segments + (profile.segments[-1],))

        report = validate_trajectory(grid, path, profile=forged)

        self.assertFalse(report.valid)
        self.assertIn("profile_size", {issue.code for issue in report.issues})

    def test_validation_rejects_malformed_profile_structure_without_crashing(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))

        report = validate_trajectory(
            grid, path, profile=replace(profile, points=("not-a-timed-point",))
        )

        self.assertFalse(report.valid)
        self.assertIn("malformed_profile", {issue.code for issue in report.issues})

    def test_validation_rejects_non_finite_fields_without_crashing(self) -> None:
        grid = OccupancyGrid.from_rows([[False] * 5 for _ in range(2)])
        path = ((0.5, 0.5), (2.5, 0.5), (4.5, 0.5))
        profile = parameterize_velocity(path, MotionLimits(3.0, 2.0, 2.0))
        forged_points = list(profile.points)
        forged_points[1] = replace(forged_points[1], speed=float("inf"))
        forged_segments = list(profile.segments)
        forged_segments[0] = replace(forged_segments[0], length=float("nan"))
        forged = replace(
            profile,
            points=tuple(forged_points),
            segments=tuple(forged_segments),
            total_time=float("inf"),
        )

        report = validate_trajectory(grid, path, profile=forged)

        self.assertFalse(report.valid)
        self.assertIn("non_finite_profile", {issue.code for issue in report.issues})

    def test_two_point_zero_boundary_speeds_use_triangular_profile(self) -> None:
        profile = parameterize_velocity(
            ((0.0, 0.0), (4.0, 0.0)), MotionLimits(10.0, 2.0, 2.0)
        )
        self.assertEqual(profile.segments[0].profile, "triangular")
        self.assertAlmostEqual(profile.segments[0].peak_speed, 8.0 ** 0.5)
        self.assertAlmostEqual(profile.segments[0].max_abs_acceleration, 2.0)
        self.assertGreater(profile.total_time, 0.0)

    def test_collision_report_identifies_segment_and_cell(self) -> None:
        grid = load_ascii_scenario(FIXTURES / "smoothing_collision.txt").grid
        report = validate_trajectory(grid, ((0.5, 2.5), (6.5, 2.5)))
        self.assertFalse(report.valid)
        collision = next(issue for issue in report.issues if issue.code == "collision")
        self.assertEqual(collision.segment_index, 0)
        self.assertEqual(collision.cell, (3, 2))

    def test_parameterization_is_exactly_deterministic(self) -> None:
        path = ((0.0, 0.0), (2.0, 0.0), (3.0, 1.0), (5.0, 1.0))
        limits = MotionLimits(4.0, 1.5, 1.25)
        first = parameterize_velocity(path, limits)
        second = parameterize_velocity(path, limits)
        self.assertEqual(first, second)

    def test_invalid_limits_geometry_and_boundary_states_are_rejected(self) -> None:
        with self.assertRaises(InvalidInputError):
            MotionLimits(max_speed=0.0)
        with self.assertRaises(InvalidInputError):
            parameterize_velocity(((0.0, 0.0), (0.0, 0.0)))
        with self.assertRaises(InvalidInputError):
            parameterize_velocity(
                ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0))
            )
        with self.assertRaises(InvalidInputError):
            parameterize_velocity(
                ((0.0, 0.0), (0.1, 0.0)),
                MotionLimits(10.0, 1.0, 2.0),
                start_speed=5.0,
                end_speed=0.0,
            )
        with self.assertRaises(InvalidInputError):
            validate_trajectory(
                OccupancyGrid.from_ascii(".."),
                ((0.5, 0.5), (1.5, 0.5)),
                max_curvature=0.0,
            )
        with self.assertRaises(InvalidInputError):
            validate_trajectory(
                OccupancyGrid.from_ascii(".."),
                ((0.5, 0.5), (1.5, 0.5)),
                max_curvature="sharp",
            )
        with self.assertRaises(InvalidInputError):
            validate_trajectory(
                OccupancyGrid.from_ascii(".."),
                ((10**400, 0.5), (1.5, 0.5)),
            )

    def test_validation_rejects_unrepresentable_finite_world_geometry(self) -> None:
        minimum_subnormal = math.nextafter(0.0, 1.0)
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        cases = (
            (
                OccupancyGrid.from_ascii("..", resolution=minimum_subnormal),
                ((0.5, 0.0), (1.5, 0.0)),
            ),
            (
                OccupancyGrid.from_ascii(".."),
                ((-maximum, 0.5), (maximum, 0.5)),
            ),
        )
        for grid, path in cases:
            with self.subTest(grid=grid, path=path):
                with self.assertRaises(InvalidInputError):
                    validate_trajectory(grid, path)

    def test_parameterization_rejects_unrepresentable_finite_arithmetic(self) -> None:
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        cases = (
            (
                "point-distance overflow",
                ((-maximum, 0.0), (maximum, 0.0)),
                MotionLimits(),
            ),
            (
                "speed-pass overflow",
                ((0.0, 0.0), (maximum, 0.0)),
                MotionLimits(
                    max_speed=maximum,
                    max_acceleration=maximum,
                    max_lateral_acceleration=maximum,
                ),
            ),
        )
        for name, path, limits in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    InvalidInputError,
                    "derived trajectory timing is not representable",
                ):
                    parameterize_velocity(path, limits)

    def test_motion_limits_reject_boolean_scalars(self) -> None:
        for name in (
            "max_speed",
            "max_acceleration",
            "max_lateral_acceleration",
        ):
            with self.subTest(name=name):
                with self.assertRaises(InvalidInputError):
                    MotionLimits(**{name: True})

    def test_boundary_speeds_reject_boolean_scalars(self) -> None:
        path = ((0.0, 0.0), (2.0, 0.0))
        for name in ("start_speed", "end_speed"):
            for value in (False, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(InvalidInputError):
                        parameterize_velocity(path, **{name: value})

    def test_out_of_bounds_and_degenerate_geometry_are_reported(self) -> None:
        grid = OccupancyGrid.from_ascii("...\n...")
        outside = validate_trajectory(grid, ((0.5, 0.5), (4.5, 0.5)))
        self.assertIn("out_of_bounds", {issue.code for issue in outside.issues})
        reversal = validate_trajectory(
            grid, ((0.5, 0.5), (1.5, 0.5), (0.5, 0.5))
        )
        self.assertIn(
            "undefined_curvature", {issue.code for issue in reversal.issues}
        )
        self.assertIsNone(reversal.to_dict()["metrics"]["max_curvature"])


if __name__ == "__main__":
    unittest.main()
