"""Nullable arithmetic promotes mixed operands to Float64. Division always returns Float64."""

import operator
from array import array
from collections.abc import Callable, Iterable
from itertools import repeat
from math import copysign, inf, isnan, nan
from typing import cast

from ..bitmap import Bitmap
from ..column import Column
from ..dtype import INT64_MAX, INT64_MIN, DType, Float64, Int64
from ..errors import ComputeError, SchemaError

type _Number = int | float
type _Operand = Column | int | float | None
type _Operation = Callable[[_Number, _Number], _Number]


def _checked_int64(value: int) -> int:
    if not INT64_MIN <= value <= INT64_MAX:
        raise ComputeError("Int64 overflow")
    return value


def _numeric_dtype(operand: _Operand) -> DType | None:
    if isinstance(operand, Column):
        if not operand.dtype.is_numeric:
            raise SchemaError(f"expected numeric dtype, got {operand.dtype.value}")
        return operand.dtype
    if type(operand) is int:
        _checked_int64(operand)
        return Int64
    if type(operand) is float:
        return Float64
    if operand is None:
        return None
    raise TypeError(f"expected a numeric Column, int, float or None, got {type(operand).__name__}")


def _values(operand: Column | _Number, length: int) -> Iterable[_Number]:
    if isinstance(operand, Column):
        return cast("array[int] | array[float]", operand._values)
    return repeat(operand, length)


def _divide(left: _Number, right: _Number) -> float:
    numerator, divisor = float(left), float(right)
    if divisor != 0.0:
        return numerator / divisor
    if numerator == 0.0 or isnan(numerator):
        return nan
    return copysign(inf, numerator) * copysign(1.0, divisor)


def _binary(left: _Operand, right: _Operand, operation: _Operation) -> Column:
    column = left if isinstance(left, Column) else right
    if not isinstance(column, Column):
        raise TypeError("at least one operand must be a Column")
    left_dtype, right_dtype = _numeric_dtype(left), _numeric_dtype(right)
    dtype = Float64 if operation is _divide or Float64 in (left_dtype, right_dtype) else Int64
    length = len(column)
    typecode = "q" if dtype is Int64 else "d"
    if left is None or right is None:
        return column._with_storage(array(typecode, [0]) * length, Bitmap(length), dtype=dtype)

    if isinstance(left, Column) and isinstance(right, Column):
        validity = left._validity & right._validity
    else:
        validity = column._validity.slice(0, length)
    values = array(typecode, [0]) * length
    pairs = zip(_values(left, length), _values(right, length), strict=True)
    for i, (a, b) in enumerate(pairs):
        if validity.get(i):
            result = operation(a, b)
            values[i] = _checked_int64(cast(int, result)) if dtype is Int64 else result
    return column._with_storage(values, validity, dtype=dtype)


def add(left: _Operand, right: _Operand) -> Column:
    """Add numeric operands, propagating nulls and checking Int64 overflow."""
    return _binary(left, right, operator.add)


def subtract(left: _Operand, right: _Operand) -> Column:
    """Subtract the right operand from the left, propagating nulls."""
    return _binary(left, right, operator.sub)


def multiply(left: _Operand, right: _Operand) -> Column:
    """Multiply numeric operands, propagating nulls and checking Int64 overflow."""
    return _binary(left, right, operator.mul)


def divide(left: _Operand, right: _Operand) -> Column:
    """Return the Float64 quotient with IEEE signed-zero and infinity semantics."""
    return _binary(left, right, _divide)


def negate(column: Column) -> Column:
    """Negate known values, preserving dtype and checking Int64 overflow."""
    if not isinstance(column, Column):
        raise TypeError("expected a Column")
    dtype = _numeric_dtype(column)
    validity = column._validity.slice(0, len(column))
    values = array("q" if dtype is Int64 else "d", [0]) * len(column)
    for i, value in enumerate(_values(column, len(column))):
        if validity.get(i):
            values[i] = _checked_int64(-cast(int, value)) if dtype is Int64 else -value
    return column._with_storage(values, validity)
