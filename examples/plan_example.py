"""Minimal library example; run with the repository's src directory on PYTHONPATH."""

from pathlib import Path

from trajectory_lab import (
    MotionLimits,
    astar,
    cells_to_points,
    load_ascii_scenario,
    parameterize_velocity,
    shortcut_path,
    validate_trajectory,
)


def main() -> None:
    map_path = Path(__file__).parent / "maps" / "warehouse.txt"
    scenario = load_ascii_scenario(map_path)
    result = astar(scenario.grid, scenario.start, scenario.goal)
    path = cells_to_points(scenario.grid, result.path)
    path = shortcut_path(scenario.grid, path)
    profile = parameterize_velocity(path, MotionLimits(4.0, 1.5, 2.0))
    report = validate_trajectory(scenario.grid, path, profile=profile)
    print(
        f"points={len(path)} length={profile.total_length:.3f} "
        f"time={profile.total_time:.3f} valid={report.valid}"
    )


if __name__ == "__main__":
    main()

