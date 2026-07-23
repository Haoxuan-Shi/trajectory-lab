from __future__ import annotations

import math
from pathlib import Path
import unittest

import _path  # noqa: F401

from trajectory_lab import (
    InvalidInputError,
    NoPathError,
    OccupancyGrid,
    astar,
    dijkstra,
    load_ascii_scenario,
    parse_ascii_scenario,
)


FIXTURES = Path(__file__).parent / "fixtures"


class SearchTests(unittest.TestCase):
    def test_grid_numeric_metadata_requires_finite_non_boolean_reals(self) -> None:
        grid = OccupancyGrid(1, 1, (False,), resolution=2, origin=(-1, 2.5))
        self.assertEqual(grid.resolution, 2)
        self.assertEqual(grid.origin, (-1, 2.5))

        for resolution in (True, "1.0", 1.0 + 0.0j):
            with self.subTest(resolution=resolution):
                with self.assertRaises(InvalidInputError):
                    OccupancyGrid(1, 1, (False,), resolution=resolution)

        for origin in (
            (True, 0.0),
            (0.0, False),
            ("0.0", 0.0),
            (0.0, 0.0 + 0.0j),
            [0.0, 0.0],
        ):
            with self.subTest(origin=origin):
                with self.assertRaises(InvalidInputError):
                    OccupancyGrid(1, 1, (False,), origin=origin)

    def test_load_rejects_invalid_grid_numeric_metadata(self) -> None:
        for options in (
            {"resolution": True},
            {"resolution": "1.0"},
            {"origin": (False, 0.0)},
            {"origin": [0.0, 0.0]},
        ):
            with self.subTest(options=options):
                with self.assertRaises(InvalidInputError):
                    load_ascii_scenario(FIXTURES / "simple_open.txt", **options)

    def test_world_to_cell_rejects_unrepresentable_finite_intermediates(self) -> None:
        minimum_subnormal = math.nextafter(0.0, 1.0)
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        cases = (
            (
                "minimum subnormal x resolution",
                OccupancyGrid(1, 1, (False,), resolution=minimum_subnormal),
                (1.0, 0.0),
            ),
            (
                "minimum subnormal y resolution",
                OccupancyGrid(1, 1, (False,), resolution=minimum_subnormal),
                (0.0, 1.0),
            ),
            (
                "positive point and negative origin",
                OccupancyGrid(1, 1, (False,), origin=(-maximum, 0.0)),
                (maximum, 0.0),
            ),
            (
                "negative point and positive origin",
                OccupancyGrid(1, 1, (False,), origin=(maximum, 0.0)),
                (-maximum, 0.0),
            ),
        )
        for name, grid, point in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    InvalidInputError, "cannot be mapped to a grid cell"
                ):
                    grid.world_to_cell(point)

    def test_astar_and_dijkstra_are_optimal_on_simple_case(self) -> None:
        scenario = load_ascii_scenario(FIXTURES / "simple_open.txt")
        self.assertIsNotNone(scenario.start)
        self.assertIsNotNone(scenario.goal)
        a_star = astar(scenario.grid, scenario.start, scenario.goal, diagonal=False)
        uniform = dijkstra(scenario.grid, scenario.start, scenario.goal, diagonal=False)
        self.assertEqual(a_star.path, ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)))
        self.assertEqual(a_star.path, uniform.path)
        self.assertAlmostEqual(a_star.cost, 4.0)
        self.assertAlmostEqual(uniform.cost, 4.0)

    def test_search_rejects_unrepresentable_finite_path_costs(self) -> None:
        grid = OccupancyGrid(3, 3, (False,) * 9, resolution=5e307)
        for planner in (astar, dijkstra):
            with self.subTest(planner=planner.__name__):
                with self.assertRaisesRegex(
                    InvalidInputError,
                    "derived search cost is not representable as a finite number",
                ):
                    planner(grid, (0, 0), (2, 2), diagonal=False)

    def test_equal_cost_detour_is_deterministic(self) -> None:
        scenario = load_ascii_scenario(FIXTURES / "detour.txt")
        expected = astar(scenario.grid, scenario.start, scenario.goal, diagonal=False)
        for _ in range(20):
            self.assertEqual(
                astar(scenario.grid, scenario.start, scenario.goal, diagonal=False),
                expected,
            )
        self.assertAlmostEqual(expected.cost, 6.0)
        self.assertEqual(
            dijkstra(scenario.grid, scenario.start, scenario.goal, diagonal=False).cost,
            expected.cost,
        )

    def test_unreachable_goal_raises(self) -> None:
        scenario = load_ascii_scenario(FIXTURES / "blocked.txt")
        with self.assertRaises(NoPathError):
            astar(scenario.grid, scenario.start, scenario.goal)
        with self.assertRaises(NoPathError):
            dijkstra(scenario.grid, scenario.start, scenario.goal)

    def test_corner_cutting_is_disabled_by_default(self) -> None:
        scenario = load_ascii_scenario(FIXTURES / "corner.txt")
        with self.assertRaises(NoPathError):
            astar(scenario.grid, scenario.start, scenario.goal)
        result = astar(
            scenario.grid,
            scenario.start,
            scenario.goal,
            allow_corner_cutting=True,
        )
        self.assertEqual(result.path, ((0, 0), (1, 1)))
        self.assertAlmostEqual(result.cost, math.sqrt(2.0))

    def test_invalid_maps_and_endpoints_are_rejected(self) -> None:
        with self.assertRaises(InvalidInputError):
            parse_ascii_scenario("..\n...")
        with self.assertRaises(InvalidInputError):
            parse_ascii_scenario("S?G")
        grid = OccupancyGrid.from_ascii("#.")
        with self.assertRaises(InvalidInputError):
            astar(grid, (0, 0), (1, 0))
        with self.assertRaises(InvalidInputError):
            astar(grid, (1, 0), (2, 0))
        with self.assertRaises(InvalidInputError):
            OccupancyGrid.from_rows([[False, False]], resolution=1e308)


if __name__ == "__main__":
    unittest.main()
