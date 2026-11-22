"""Nullable comparisons over numeric values, Unicode text, and Boolean equality."""

import operator
from array import array
from collections.abc import Callable

from ..bitmap import Bitmap
from ..column import Column, _Scalar
from ..dtype import Boolean, Categorical, DType, Float64, Int64, String
from ..errors import ComputeError, SchemaError

type _Operand = Column | _Scalar


def _operand_dtype(operand: _Operand) -> DType:
    if isinstance(operand, Column):
        return operand.dtype
    if type(operand) is int:
        if not Int64.accepts(operand):
            raise ComputeError("Int64 overflow")
        return Int64
    if type(operand) is float:
        return Float64
    if type(operand) is bool:
        return Boolean
    if type(operand) is str:
        return String
    raise SchemaError(f"unsupported comparison operand: {type(operand).__name__}")


def _reader(operand: _Operand) -> Callable[[int], _Scalar]:
    if isinstance(operand, Column):
        if isinstance(operand._values, array):
            return operand._values.__getitem__
        return operand.get
    return lambda _: operand


def _compare(left: _Operand, right: _Operand, op: Callable[[_Scalar, _Scalar], bool]) -> Column:
    column = left if isinstance(left, Column) else right
    if not isinstance(column, Column):
        raise TypeError("at least one operand must be a Column")
    length = len(column)
    if left is None or right is None:
        return column._with_storage(Bitmap(length), Bitmap(length), dtype=Boolean)
    a, b = _operand_dtype(left), _operand_dtype(right)
    if not (a is b or (a.is_numeric and b.is_numeric) or {a, b} == {String, Categorical}):
        raise SchemaError(f"cannot compare {a.value} with {b.value}")
    if a is Boolean and op not in (operator.eq, operator.ne):
        raise SchemaError("Boolean supports only equality and inequality")

    if isinstance(left, Column) and isinstance(right, Column):
        validity = left._validity & right._validity
    else:
        validity = column._validity.slice(0, length)
    left_at, right_at = _reader(left), _reader(right)
    values = Bitmap.from_bools(
        validity.get(i) and op(left_at(i), right_at(i)) for i in range(length)
    )
    return column._with_storage(values, validity, dtype=Boolean)


def eq(left: _Operand, right: _Operand) -> Column:
    """Compare for equality, propagating nulls."""
    return _compare(left, right, operator.eq)


def ne(left: _Operand, right: _Operand) -> Column:
    """Compare for inequality, propagating nulls."""
    return _compare(left, right, operator.ne)


def lt(left: _Operand, right: _Operand) -> Column:
    """Compare numeric or text operands using strict less-than."""
    return _compare(left, right, operator.lt)


def le(left: _Operand, right: _Operand) -> Column:
    """Compare numeric or text operands using less-than-or-equal."""
    return _compare(left, right, operator.le)


def gt(left: _Operand, right: _Operand) -> Column:
    """Compare numeric or text operands using strict greater-than."""
    return _compare(left, right, operator.gt)


def ge(left: _Operand, right: _Operand) -> Column:
    """Compare numeric or text operands using greater-than-or-equal."""
    return _compare(left, right, operator.ge)
