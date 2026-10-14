"""Small numeric validators shared by public scientific functions.

Python's ``int`` and ``float`` constructors accept booleans and silently
truncate fractional values. Scientific parameters use these helpers so an
invalid caller value fails before it can change a calculation or simulation.
"""

from __future__ import annotations

from numbers import Integral, Rational, Real

import numpy as np


__all__ = [
    "contains_boolean",
    "finite_integer",
    "finite_positive_float",
    "nonnegative_integer",
    "positive_integer",
    "strict_boolean",
]


def contains_boolean(value: object) -> bool:
    """Return whether a scalar or array-like contains a boolean element."""

    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.dtype.kind == "b":
        return True
    if array.dtype.kind == "O":
        items = array.reshape(-1)
    elif isinstance(value, np.ndarray):
        return False
    else:
        try:
            items = np.asarray(value, dtype=object).reshape(-1)
        except (TypeError, ValueError):
            return False
    return any(isinstance(item, (bool, np.bool_)) for item in items)


def strict_boolean(value: object, *, name: str) -> bool:
    """Return one explicit boolean without accepting truthy substitutes."""

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def finite_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an exact finite integer within optional inclusive bounds."""

    if contains_boolean(value) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite integer")
    if isinstance(value, Integral):
        integer = int(value)
    elif isinstance(value, Rational):
        if value.denominator != 1:
            raise ValueError(f"{name} must be a finite integer")
        integer = int(value.numerator)
    else:
        numeric = float(value)
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValueError(f"{name} must be a finite integer")
        integer = int(numeric)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and integer > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return integer


def nonnegative_integer(value: object, *, name: str) -> int:
    """Return an exact integer greater than or equal to zero."""

    try:
        return finite_integer(value, name=name, minimum=0)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite non-negative integer") from error




def positive_integer(value: object, *, name: str) -> int:
    """Return an exact integer greater than zero."""

    try:
        return finite_integer(value, name=name, minimum=1)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite positive integer") from error


def finite_positive_float(value: object, *, name: str) -> float:
    """Return a finite real scalar greater than zero."""

    if contains_boolean(value) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric
