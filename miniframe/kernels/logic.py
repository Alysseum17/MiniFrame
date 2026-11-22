from typing import cast

from ..bitmap import Bitmap
from ..column import Column
from ..dtype import Boolean
from ..errors import SchemaError


def _boolean_buffers(column: Column) -> tuple[Bitmap, Bitmap]:
    if not isinstance(column, Column):
        raise TypeError(f"expected a Column, got {type(column).__name__}")
    if column.dtype is not Boolean:
        raise SchemaError(f"expected Boolean dtype, got {column.dtype.value}")
    return cast(Bitmap, column._values), column._validity


def and_kleene(left: Column, right: Column) -> Column:
    """Return the Boolean conjunction. A known False determines the result."""
    left_values, left_valid = _boolean_buffers(left)
    right_values, right_valid = _boolean_buffers(right)
    values = left_values & right_values
    false = (left_valid & ~left_values) | (right_valid & ~right_values)
    return left._with_storage(values, values | false)


def or_kleene(left: Column, right: Column) -> Column:
    """Return the Boolean disjunction. A known True determines the result."""
    left_values, left_valid = _boolean_buffers(left)
    right_values, right_valid = _boolean_buffers(right)
    values = left_values | right_values
    validity = values | (left_valid & right_valid)
    return left._with_storage(values, validity)


def not_kleene(column: Column) -> Column:
    """Invert each known Boolean and preserve nulls."""
    values, validity = _boolean_buffers(column)
    return column._with_storage(~values & validity, validity.slice(0, len(column)))
