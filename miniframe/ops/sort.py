from array import array
from collections.abc import Sequence
from math import isnan

from ..column import Column, _Scalar
from ..errors import ShapeError

type _Key = tuple[Sequence[_Scalar], bool, bool]


def _flags(value: bool | Sequence[bool], count: int, name: str) -> tuple[bool, ...]:
    if type(value) is bool:
        return (value,) * count
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a bool or a sequence of bools")
    if len(value) != count:
        raise ShapeError(f"{name} requires one flag per sort key")
    if any(type(flag) is not bool for flag in value):
        raise TypeError(f"{name} flags must be bools")
    return tuple(value)


def _compare_rows(keys: list[_Key], left: int, right: int) -> int:
    for values, descending, nulls_last in keys:
        a, b = values[left], values[right]
        if a == b:
            continue
        if a is None:
            return 1 if nulls_last else -1
        if b is None:
            return -1 if nulls_last else 1
        if isinstance(a, float) and isnan(a):
            if isinstance(b, float) and isnan(b):
                continue
            order = 1
        elif isinstance(b, float) and isnan(b):
            order = -1
        else:
            order = -1 if a < b else 1
        return -order if descending else order
    return 0


def _key_values(column: Column) -> Sequence[_Scalar]:
    if column.null_count == 0 and isinstance(column._values, array):
        return column._values
    return column.to_list()


def argsort(
    columns: Sequence[Column],
    *,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = False,
) -> list[int]:
    """Return a stable lexicographic permutation for equally long key columns.

    Flags may be scalar or per key. Null placement is independent of direction.
    NaNs tie above all other numeric values in ascending order.
    """
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes, bytearray)):
        raise TypeError("sort keys must be a sequence of Columns")
    if not columns:
        raise ValueError("at least one sort key is required")
    if any(not isinstance(column, Column) for column in columns):
        raise TypeError("sort keys must be Columns")
    length = len(columns[0])
    if any(len(column) != length for column in columns):
        raise ShapeError("sort keys must have equal lengths")
    directions = _flags(descending, len(columns), "descending")
    placements = _flags(nulls_last, len(columns), "nulls_last")
    keys = [
        (_key_values(column), direction, placement)
        for column, direction, placement in zip(columns, directions, placements, strict=True)
    ]

    source = list(range(length))
    destination = [0] * length
    width = 1
    while width < length:
        for start in range(0, length, 2 * width):
            middle = min(start + width, length)
            stop = min(middle + width, length)
            left, right = start, middle
            for out in range(start, stop):
                if right == stop or (
                    left < middle and _compare_rows(keys, source[left], source[right]) <= 0
                ):
                    destination[out] = source[left]
                    left += 1
                else:
                    destination[out] = source[right]
                    right += 1
        source, destination = destination, source
        width *= 2
    return source
