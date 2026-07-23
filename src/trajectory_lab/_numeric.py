"""Shared predicates for validating public numeric inputs."""

from __future__ import annotations

import math
from numbers import Real

from .errors import InvalidInputError


def is_real(value: object) -> bool:
    """Return whether value is a non-boolean real scalar."""

    return isinstance(value, Real) and not isinstance(value, bool)


def is_finite_real(value: object) -> bool:
    """Return whether value is a finite, non-boolean real scalar."""

    if not is_real(value):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def exceeds_upper_bound(value: float, bound: float) -> bool:
    """Return whether ``value`` is over ``bound`` beyond one float neighbor.

    Derived binary64 calculations can round an exact boundary result to the
    next representable value.  This predicate admits only that single upward
    neighbor; unlike a fixed absolute tolerance, its allowance scales with the
    represented caller bound.
    """

    return value > math.nextafter(bound, math.inf)


def checked_finite_difference(
    left: object, right: object, *, message: str
) -> float:
    """Subtract two validated scalars or raise a stable domain error."""

    try:
        result = left - right  # type: ignore[operator]
    except (ArithmeticError, TypeError, ValueError) as error:
        raise InvalidInputError(message) from error
    return _finite_float_result(result, message)


def checked_finite_division(
    numerator: object, denominator: object, *, message: str
) -> float:
    """Divide two validated scalars or raise a stable domain error."""

    try:
        result = numerator / denominator  # type: ignore[operator]
    except (ArithmeticError, TypeError, ValueError) as error:
        raise InvalidInputError(message) from error
    return _finite_float_result(result, message)


def checked_finite_sum(left: object, right: object, *, message: str) -> float:
    """Add two validated scalars or raise a stable domain error."""

    try:
        result = left + right  # type: ignore[operator]
    except (ArithmeticError, TypeError, ValueError) as error:
        raise InvalidInputError(message) from error
    return _finite_float_result(result, message)


def checked_finite_product(
    left: object, right: object, *, message: str
) -> float:
    """Multiply two validated scalars or raise a stable domain error."""

    try:
        result = left * right  # type: ignore[operator]
    except (ArithmeticError, TypeError, ValueError) as error:
        raise InvalidInputError(message) from error
    return _finite_float_result(result, message)


def checked_floor(value: object, *, message: str) -> int:
    """Convert a finite scalar to its floor without leaking conversion errors."""

    if not is_finite_real(value):
        raise InvalidInputError(message)
    try:
        result = math.floor(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise InvalidInputError(message) from error
    if type(result) is not int:
        raise InvalidInputError(message)
    return result


def checked_ceil(value: object, *, message: str) -> int:
    """Convert a finite scalar to its ceiling without leaking conversion errors."""

    if not is_finite_real(value):
        raise InvalidInputError(message)
    try:
        result = math.ceil(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise InvalidInputError(message) from error
    if type(result) is not int:
        raise InvalidInputError(message)
    return result


def checked_point_distance(
    first: tuple[object, object],
    second: tuple[object, object],
    *,
    message: str,
) -> float:
    """Return a finite Euclidean distance with checked coordinate differences."""

    dx = checked_finite_difference(second[0], first[0], message=message)
    dy = checked_finite_difference(second[1], first[1], message=message)
    try:
        distance = math.hypot(dx, dy)
    except (ArithmeticError, TypeError, ValueError) as error:
        raise InvalidInputError(message) from error
    return _finite_float_result(distance, message)


def _finite_float_result(value: object, message: str) -> float:
    if not is_finite_real(value):
        raise InvalidInputError(message)
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise InvalidInputError(message) from error
