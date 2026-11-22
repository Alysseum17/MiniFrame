"""Internal expression execution over a mapping of equal-height columns."""

from array import array
from collections.abc import Mapping
from itertools import repeat
from typing import cast

from .bitmap import Bitmap
from .column import Column, _Scalar
from .dtype import Boolean, DType, Float64, Int64
from .errors import ColumnNotFoundError, SchemaError, ShapeError
from .expr import (
    Expr,
    Scalar,
    _Aggregate,
    _AggregateOp,
    _Alias,
    _Binary,
    _BinaryOp,
    _Cast,
    _ColumnRef,
    _Cumulative,
    _Fill,
    _Literal,
    _name,
    _Over,
    _Rank,
    _Rolling,
    _Shift,
    _Unary,
)
from .kernels import aggregate, arithmetic, compare, logic
from .kernels.cast import cast_column
from .ops import window

type Value = Column | Scalar

_ARITHMETIC = {
    "add": arithmetic.add,
    "sub": arithmetic.subtract,
    "mul": arithmetic.multiply,
    "div": arithmetic.divide,
}
_COMPARISON = {
    "eq": compare.eq,
    "ne": compare.ne,
    "lt": compare.lt,
    "le": compare.le,
    "gt": compare.gt,
    "ge": compare.ge,
}
_REDUCTIONS = {
    "sum": (aggregate.aggregate_sum, None),
    "mean": (aggregate.aggregate_mean, Float64),
    "min": (aggregate.aggregate_min, None),
    "max": (aggregate.aggregate_max, None),
    "count": (aggregate.aggregate_count, Int64),
    "len": (aggregate.aggregate_len, Int64),
    "n_unique": (aggregate.aggregate_n_unique, Int64),
    "std": (aggregate.aggregate_std, Float64),
    "var": (aggregate.aggregate_var, Float64),
    "first": (aggregate.aggregate_first, None),
    "last": (aggregate.aggregate_last, None),
}


def _check_height(height: int) -> None:
    if type(height) is not int:
        raise TypeError("height must be an int")
    if height < 0:
        raise ShapeError("height must be nonnegative")


def broadcast_scalar(scalar: Scalar, *, height: int) -> Column:
    """Repeat a typed scalar to the requested height, including zero."""
    _check_height(height)
    if scalar.dtype is None:
        raise SchemaError("an untyped null needs a dtype to broadcast")
    unit = Column((scalar.value,), scalar.dtype)
    if height == 1:
        return unit
    values = unit._values
    if isinstance(values, Bitmap):
        values = Bitmap(height, scalar.value is True)
    elif isinstance(values, array):
        values = values * height
    else:
        return unit.take(repeat(0, height))
    return unit._with_storage(values, Bitmap(height, scalar.value is not None))


def _as_column(value: Value, height: int, null_dtype: DType | None = None) -> Column:
    if isinstance(value, Column):
        return value
    if value.dtype is None:
        value = Scalar(None, null_dtype)
    return broadcast_scalar(value, height=height)


def _scalar_at_zero(column: Column) -> Scalar:
    return Scalar(column.get(0), column.dtype)


def _kernel_operand(value: Value, height: int) -> Column | _Scalar:
    if isinstance(value, Column):
        return value
    if value.value is None and value.dtype is not None:
        return broadcast_scalar(value, height=height)
    return value.value


def _numeric_operand(value: Column | _Scalar) -> Column | int | float | None:
    if isinstance(value, Column) or value is None or type(value) in (int, float):
        return cast(Column | int | float | None, value)
    raise SchemaError(f"expected a numeric operand, got {type(value).__name__}")


def _binary(op: _BinaryOp, left: Value, right: Value, height: int) -> Value:
    scalar_result = isinstance(left, Scalar) and isinstance(right, Scalar)
    length = 1 if scalar_result else height
    if op in ("and", "or"):
        kernel = logic.and_kleene if op == "and" else logic.or_kleene
        result = kernel(_as_column(left, length, Boolean), _as_column(right, length, Boolean))
    else:
        if scalar_result:
            if left.dtype is not None:
                left = _as_column(left, 1)
            elif right.dtype is not None:
                right = _as_column(right, 1)
            elif op in _COMPARISON:
                return Scalar(None, Boolean)
            else:
                raise SchemaError("arithmetic on two untyped nulls requires a numeric dtype")
        a, b = _kernel_operand(left, length), _kernel_operand(right, length)
        if op in _ARITHMETIC:
            result = _ARITHMETIC[op](_numeric_operand(a), _numeric_operand(b))
        else:
            result = _COMPARISON[op](a, b)
    return _scalar_at_zero(result) if scalar_result else result


def _elementwise(expr: _Unary | _Fill | _Cast, value: Value) -> Value:
    if isinstance(expr, _Cast):
        if value.dtype is None:
            return Scalar(None, expr.dtype)
        result = cast_column(_as_column(value, 1), expr.dtype)
    elif isinstance(expr, _Fill):
        if value.dtype is None:
            return expr.replacement
        result = _as_column(value, 1).fill_null(expr.replacement.value)
    elif expr.op in ("is_null", "is_not_null"):
        is_null = expr.op == "is_null"
        if isinstance(value, Scalar):
            return Scalar((value.value is None) == is_null, Boolean)
        return value._null_mask(is_null=is_null)
    elif expr.op == "neg":
        result = arithmetic.negate(_as_column(value, 1))
    else:
        result = logic.not_kleene(_as_column(value, 1, Boolean))
    return _scalar_at_zero(result) if isinstance(value, Scalar) else result


def _reduce(op: _AggregateOp, value: Value, height: int) -> Scalar:
    column = _as_column(value, height)
    kernel, dtype = _REDUCTIONS[op]
    return Scalar(kernel(column), column.dtype if dtype is None else dtype)


def evaluate_expr(expr: Expr, columns: Mapping[str, Column], *, height: int) -> Value:
    """Evaluate in left-to-right dependency order. Scalar reduction inputs span the context."""
    _check_height(height)
    if not isinstance(columns, Mapping):
        raise TypeError("expected a mapping of names to Columns")
    for name, column in columns.items():
        _name(name)
        if not isinstance(column, Column):
            raise TypeError(f"expected a Column for {name!r}")
        if len(column) != height:
            raise ShapeError(f"column {name!r} has length {len(column)}, expected {height}")

    def visit(node: Expr, in_partition: bool = False) -> Value:
        if in_partition and isinstance(node, (_Aggregate, _Over)):
            raise ShapeError("nested aggregation or over inside a grouped window")
        match node:
            case _ColumnRef(name):
                if name not in columns:
                    raise ColumnNotFoundError(f"column not found: {name!r}")
                return columns[name]
            case _Literal(scalar):
                return scalar
            case _Alias(operand, _):
                return visit(operand, in_partition)
            case _Binary(op, left, right):
                return _binary(op, visit(left, in_partition), visit(right, in_partition), height)
            case _Rolling(op, operand, size, minimum):
                return window.rolling(
                    _as_column(visit(operand, in_partition), height), op, size, minimum
                )
            case _Cumulative(op, operand):
                return window.cumulative(_as_column(visit(operand, in_partition), height), op)
            case _Rank(operand, method, descending):
                return window.rank(
                    _as_column(visit(operand, in_partition), height), method, descending
                )
            case _Shift(operand, n):
                return window.shift(_as_column(visit(operand, in_partition), height), n)
            case _Over(reduction, keys):
                partitions = tuple(_as_column(visit(key, True), height) for key in keys)
                source = _as_column(visit(reduction.operand, True), height)
                return window.over(source, reduction.op, partitions)
            case _Aggregate(op, operand):
                return _reduce(op, visit(operand, in_partition), height)
            case _Unary() | _Fill() | _Cast():
                return _elementwise(node, visit(node.operand, in_partition))
            case _:
                raise TypeError("expected an expression built with col() or lit()")

    return visit(expr)
