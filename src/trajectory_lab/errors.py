"""Public exception hierarchy for trajectory-lab."""


class TrajectoryLabError(Exception):
    """Base class for package-specific failures."""


class InvalidInputError(TrajectoryLabError, ValueError):
    """Raised when a public input violates a documented precondition."""


class NoPathError(TrajectoryLabError, RuntimeError):
    """Raised when a valid planning problem has no reachable solution."""

