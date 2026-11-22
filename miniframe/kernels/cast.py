"""Strict casts between numeric types, between text encodings, and from String to numbers."""

import re
from collections.abc import Iterator
from decimal import Decimal
from math import isfinite
from typing import cast

from ..column import Column
from ..dtype import INT64_MAX, INT64_MIN, Boolean, Categorical, DType, Int64, String
from ..errors import ComputeError, SchemaError

_INTEGER = re.compile(r"[+-]?[0-9]+")
_FLOAT = re.compile(r"[+-]?([0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _to_int(value: str | float) -> int:
    number: Decimal | float
    if isinstance(value, str):
        if not _INTEGER.fullmatch(value):
            raise ValueError("expected a decimal integer")
        number = Decimal(value)
    else:
        if not value.is_integer():
            raise ValueError("expected a finite integral value")
        number = value
    if not INT64_MIN <= number <= INT64_MAX:
        raise ValueError("outside Int64 range")
    return int(number)


def _to_float(value: str | int) -> float:
    if isinstance(value, int):
        result = float(value)
        if result != value:
            raise ValueError("integer loses precision in Float64")
        return result
    match = _FLOAT.fullmatch(value)
    if match is None:
        raise ValueError("expected a decimal number")
    result = float(value)
    if not isfinite(result) or (result == 0.0 and match[1].strip("0.")):
        raise ValueError("outside Float64 range")
    return result


def cast_column(column: Column, dtype: DType) -> Column:
    """Copy a column into the target dtype. Impossible conversions raise ComputeError."""
    if not isinstance(column, Column):
        raise TypeError("expected a Column")
    if not isinstance(dtype, DType):
        raise SchemaError(f"Unsupported column dtype: {dtype!r}")
    source = column.dtype
    if source is dtype:
        return column.slice(0, len(column))
    if source is Boolean and dtype is Int64:
        values = (column.get(i) for i in range(len(column)))
        return Column((None if value is None else int(value) for value in values), Int64)
    if {source, dtype} == {String, Categorical}:
        try:
            return Column((column.get(i) for i in range(len(column))), dtype)
        except SchemaError as error:
            raise ComputeError(str(error)) from error
    if not (dtype.is_numeric and (source.is_numeric or source is String)):
        raise ComputeError(f"cannot cast {source.value} to {dtype.value}")

    def converted() -> Iterator[int | float | None]:
        for i in range(len(column)):
            value = column.get(i)
            try:
                if value is None:
                    yield None
                elif dtype is Int64:
                    yield _to_int(cast(str | float, value))
                else:
                    yield _to_float(cast(str | int, value))
            except ValueError as error:
                raise ComputeError(
                    f"{source.value} to {dtype.value} at position {i}: {error}"
                ) from error

    return Column(converted(), dtype)
