from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import _path  # noqa: F401

from trajectory_lab import InvalidInputError
from trajectory_lab.cli import _read_path_csv, build_parser, main


FIXTURES = Path(__file__).parent / "fixtures"


class CliTests(unittest.TestCase):
    def test_path_csv_rejects_duplicate_and_blank_headers(self) -> None:
        cases = {
            "duplicate x": "x,x,y\n0.5,9,0.5\n1.5,9,0.5\n",
            "duplicate y": "x,y,y\n0.5,0.5,9\n1.5,0.5,9\n",
            "duplicate extra": (
                "x,y,label,label\n"
                "0.5,0.5,start,duplicate\n"
                "1.5,0.5,finish,duplicate\n"
            ),
            "duplicate after trimming": (
                "x, x ,y\n0.5,9,0.5\n1.5,9,0.5\n"
            ),
            "blank": "x,,y\n0.5,ignored,0.5\n1.5,ignored,0.5\n",
            "whitespace blank": "x,   ,y\n0.5,ignored,0.5\n1.5,ignored,0.5\n",
        }
        for name, contents in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path_csv = Path(directory) / "path.csv"
                path_csv.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(InvalidInputError, "path CSV header"):
                    _read_path_csv(str(path_csv))

    def test_path_csv_requires_exactly_one_x_and_y(self) -> None:
        cases = {
            "missing x": "longitude,y\n0.5,0.5\n1.5,0.5\n",
            "missing y": "x,latitude\n0.5,0.5\n1.5,0.5\n",
        }
        for name, contents in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path_csv = Path(directory) / "path.csv"
                path_csv.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(
                    InvalidInputError, "exactly one x and one y"
                ):
                    _read_path_csv(str(path_csv))

    def test_path_csv_accepts_and_ignores_extra_unique_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_csv = root / "path.csv"
            report_json = root / "report.json"
            path_csv.write_text(
                "label,y,x,confidence\n"
                "start,0.5,0.5,0.8\n"
                "finish,0.5,1.5,0.9\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _read_path_csv(str(path_csv)),
                ((0.5, 0.5), (1.5, 0.5)),
            )
            self.assertEqual(
                main(
                    [
                        "validate",
                        "--map",
                        str(FIXTURES / "simple_open.txt"),
                        "--path",
                        str(path_csv),
                        "--report",
                        str(report_json),
                    ]
                ),
                0,
            )

    def test_path_csv_rejects_ragged_rows_and_extra_fields(self) -> None:
        cases = {
            "missing field": (
                "x,y,label\n0.5,0.5,start\n1.5,0.5\n",
                r"row 3 has 2 fields; expected 3",
            ),
            "extra field": (
                "x,y\n0.5,0.5\n1.5,0.5,unexpected\n",
                r"row 3 has 3 fields; expected 2",
            ),
            "blank row": (
                "x,y\n0.5,0.5\n\n1.5,0.5\n",
                r"row 3 has 0 fields; expected 2",
            ),
        }
        for name, (contents, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path_csv = Path(directory) / "path.csv"
                path_csv.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(InvalidInputError, message):
                    _read_path_csv(str(path_csv))

    def test_csv_shape_errors_map_to_cli_exit_2(self) -> None:
        cases = {
            "duplicate header": "x,x,y\n0.5,9,0.5\n1.5,9,0.5\n",
            "blank header": "x,,y\n0.5,note,0.5\n1.5,note,0.5\n",
            "ragged row": "x,y,label\n0.5,0.5,start\n1.5,0.5\n",
            "extra field": "x,y\n0.5,0.5\n1.5,0.5,unexpected\n",
        }
        for name, contents in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path_csv = root / "path.csv"
                path_csv.write_text(contents, encoding="utf-8")
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = main(
                        [
                            "validate",
                            "--map",
                            str(FIXTURES / "simple_open.txt"),
                            "--path",
                            str(path_csv),
                            "--report",
                            str(root / "report.json"),
                        ]
                    )
                self.assertEqual(exit_code, 2)
                self.assertIn("trajectory-lab: invalid input:", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_extreme_finite_validation_inputs_map_to_cli_exit_2(self) -> None:
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        cases = {
            "minimum subnormal resolution": (
                "x,y\n0.5,0.5\n1.5,0.5\n",
                "5e-324",
            ),
            "overflowed finite point difference": (
                f"x,y\n{-maximum},0.5\n{maximum},0.5\n",
                "1",
            ),
        }
        for name, (contents, resolution) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path_csv = root / "path.csv"
                path_csv.write_text(contents, encoding="utf-8")
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = main(
                        [
                            "validate",
                            "--map",
                            str(FIXTURES / "simple_open.txt"),
                            "--path",
                            str(path_csv),
                            "--resolution",
                            resolution,
                            "--report",
                            str(root / "report.json"),
                        ]
                    )
                self.assertEqual(exit_code, 2)
                self.assertIn("trajectory-lab: invalid input:", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_unrepresentable_search_cost_maps_to_cli_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = root / "extreme.txt"
            scenario.write_text("S..\n...\n..G\n", encoding="utf-8")
            for algorithm in ("astar", "dijkstra"):
                with self.subTest(algorithm=algorithm):
                    output = root / f"{algorithm}.csv"
                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        exit_code = main(
                            [
                                "plan",
                                "--map",
                                str(scenario),
                                "--algorithm",
                                algorithm,
                                "--no-diagonal",
                                "--resolution",
                                "5e307",
                                "--output",
                                str(output),
                            ]
                        )
                    self.assertEqual(exit_code, 2)
                    self.assertIn(
                        "derived search cost is not representable",
                        stderr.getvalue(),
                    )
                    self.assertNotIn("Traceback", stderr.getvalue())
                    self.assertFalse(output.exists())

    def test_plan_writes_deterministic_csv_and_valid_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_csv = root / "first.csv"
            second_csv = root / "second.csv"
            first_json = root / "first.json"
            second_json = root / "second.json"
            common = [
                "plan",
                "--map",
                str(FIXTURES / "simple_open.txt"),
                "--algorithm",
                "astar",
                "--no-diagonal",
                "--shortcut",
            ]
            self.assertEqual(
                main(common + ["--output", str(first_csv), "--report", str(first_json)]),
                0,
            )
            self.assertEqual(
                main(common + ["--output", str(second_csv), "--report", str(second_json)]),
                0,
            )
            self.assertEqual(first_csv.read_bytes(), second_csv.read_bytes())
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            payload = json.loads(first_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["planner"]["algorithm"], "astar")

    def test_unreachable_plan_has_distinct_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exit_code = main(
                [
                    "plan",
                    "--map",
                    str(FIXTURES / "blocked.txt"),
                    "--output",
                    str(Path(directory) / "path.csv"),
                ]
            )
            self.assertEqual(exit_code, 3)

    def test_dijkstra_and_lattice_cli_paths_are_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for algorithm in ("dijkstra", "lattice"):
                with self.subTest(algorithm=algorithm):
                    output = root / f"{algorithm}.csv"
                    self.assertEqual(
                        main(
                            [
                                "plan",
                                "--map",
                                str(FIXTURES / "simple_open.txt"),
                                "--algorithm",
                                algorithm,
                                "--output",
                                str(output),
                            ]
                        ),
                        0,
                    )
                    self.assertTrue(output.read_text(encoding="utf-8").startswith("x,y,"))

    def test_validate_reports_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path_csv = Path(directory) / "path.csv"
            report_json = Path(directory) / "report.json"
            path_csv.write_text("x,y\n0.5,2.5\n6.5,2.5\n", encoding="utf-8")
            exit_code = main(
                [
                    "validate",
                    "--map",
                    str(FIXTURES / "smoothing_collision.txt"),
                    "--path",
                    str(path_csv),
                    "--report",
                    str(report_json),
                ]
            )
            self.assertEqual(exit_code, 4)
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertFalse(payload["valid"])
            self.assertIn("collision", {issue["code"] for issue in payload["issues"]})

    def test_invalid_map_returns_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid_map = Path(directory) / "bad.txt"
            invalid_map.write_text("S?G\n", encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "plan",
                        "--map",
                        str(invalid_map),
                        "--output",
                        str(Path(directory) / "path.csv"),
                    ]
                ),
                2,
            )

    def test_cli_rejects_boolean_text_for_numeric_scalars(self) -> None:
        parser = build_parser()
        for option in (
            "--resolution",
            "--max-speed",
            "--max-acceleration",
            "--max-lateral-acceleration",
            "--start-speed",
            "--end-speed",
        ):
            with self.subTest(option=option):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parser.parse_args(
                            [
                                "plan",
                                "--map",
                                str(FIXTURES / "simple_open.txt"),
                                option,
                                "true",
                            ]
                        )
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
