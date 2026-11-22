"""Height-preserving operations on nullable columns."""

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from decimal import localcontext
from math import inf, isfinite, isnan, nan
from typing import Literal, cast

from ..column import Column, _Scalar
from ..dtype import Boolean, Float64, Int64
from ..errors import SchemaError, ShapeError
from ..kernels.aggregate import (
    _WELFORD_CONTEXT,
    _AggregateOp,
    _distinct_key,
    _float_order,
    _FloatTotal,
)
from ..kernels.arithmetic import _checked_int64
from .groupby import _reduction, group_ids
from .join import _gather
from .sort import argsort

type CumulativeOp = Literal["sum", "min", "max"]
type RollingOp = CumulativeOp | Literal["mean"]
type RankMethod = Literal["average", "min", "max", "dense", "ordinal"]


def periods(size: int, minimum: int | None) -> int:
    if type(size) is not int or (minimum is not None and type(minimum) is not int):
        raise TypeError("window_size and min_periods must be built-in integers")
    if size < 1:
        raise ValueError("window_size must be positive")
    if minimum is None:
        return size
    if not 0 <= minimum <= size:
        raise ValueError("min_periods must be between zero and window_size")
    return minimum


def validate_rank(method: RankMethod, descending: bool) -> None:
    if method not in ("average", "min", "max", "dense", "ordinal"):
        raise ValueError(f"unsupported ranking method: {method!r}")
    if type(descending) is not bool:
        raise TypeError("descending must be a bool")


def validate_shift(n: int) -> None:
    if type(n) is not int:
        raise TypeError("shift offset must be a built-in integer")


@dataclass(slots=True)
class _SlidingTotal:
    """Exact binary-rational finite sum with reversible counts for nonfinite observations."""

    finite: _FloatTotal = field(default_factory=_FloatTotal)
    count: int = 0
    nans: int = 0
    positive_inf: int = 0
    negative_inf: int = 0

    def update(self, value: int | float, direction: int) -> None:
        self.count += direction
        if type(value) is int:
            self.finite.total += value * direction
        elif isfinite(value):
            self.finite.update(value * direction)
        elif isnan(value):
            self.nans += direction
        elif value > 0:
            self.positive_inf += direction
        else:
            self.negative_inf += direction

    def floating_result(self, divisor: int) -> float:
        if self.nans or (self.positive_inf and self.negative_inf):
            return nan
        if self.positive_inf:
            return inf
        if self.negative_inf:
            return -inf
        return self.finite.divided_by(divisor)


def _rolling_total(column: Column, size: int, minimum: int, mean: bool) -> Iterator[_Scalar]:
    state = _SlidingTotal()
    for i in range(len(column)):
        if i >= size:
            leaving = column.get(i - size)
            if leaving is not None:
                state.update(cast(int | float, leaving), -1)
        entering = column.get(i)
        if entering is not None:
            state.update(cast(int | float, entering), 1)
        if state.count < minimum or (mean and not state.count):
            yield None
        elif column.dtype is Int64 and not mean:
            yield _checked_int64(state.finite.total)
        else:
            yield state.floating_result(state.count if mean else 1)


def _dominates(value: int | float | str, previous: int | float | str, maximum: bool) -> bool:
    if isinstance(value, float):
        a, b = _float_order(value), _float_order(cast(float, previous))
        return a > b if maximum else a < b
    return value > previous if maximum else value < previous


def _rolling_extreme(column: Column, size: int, minimum: int, maximum: bool) -> Iterator[_Scalar]:
    candidates: deque[int] = deque()
    count = 0
    for i in range(len(column)):
        if i >= size:
            count -= column._is_valid(i - size)
            if candidates and candidates[0] == i - size:
                candidates.popleft()
        value = column.get(i)
        if value is not None:
            count += 1
            while candidates:
                previous = column.get(candidates[-1])
                assert previous is not None
                if not _dominates(value, previous, maximum):
                    break
                candidates.pop()
            candidates.append(i)
        yield column.get(candidates[0]) if candidates and count >= minimum else None


def rolling(column: Column, op: RollingOp, size: int, minimum: int | None) -> Column:
    threshold = periods(size, minimum)
    if op in ("min", "max"):
        if column.dtype is Boolean:
            raise SchemaError("Boolean does not support ordering")
        return Column(_rolling_extreme(column, size, threshold, op == "max"), column.dtype)
    if not column.dtype.is_numeric:
        raise SchemaError(f"rolling_{op} requires a numeric operand")
    mean = op == "mean"
    return Column(_rolling_total(column, size, threshold, mean), Float64 if mean else column.dtype)


def cumulative(column: Column, op: CumulativeOp) -> Column:
    dtype, factory = _reduction(op, column.dtype)
    state = factory()

    def prefixes() -> Iterator[_Scalar]:
        for i in range(len(column)):
            value = column.get(i)
            if value is None:
                yield None
            else:
                state.update(value)
                yield state.finish()

    return Column(prefixes(), dtype)


def rank(column: Column, method: RankMethod, descending: bool) -> Column:
    validate_rank(method, descending)
    order = argsort((column,), descending=descending, nulls_last=True)
    valid = len(column) - column.null_count
    ranks: list[int | float | None] = [None] * len(column)
    start = dense = 0
    while start < valid:
        key = _distinct_key(column.get(order[start]))
        end = start + 1
        while end < valid and _distinct_key(column.get(order[end])) == key:
            end += 1
        dense += 1
        if method == "average":
            tied = (start + 1 + end) / 2
        elif method == "min":
            tied = start + 1
        elif method == "max":
            tied = end
        else:
            tied = dense
        for position in range(start, end):
            ranks[order[position]] = position + 1 if method == "ordinal" else tied
        start = end
    return Column(ranks, Float64 if method == "average" else Int64)


def shift(column: Column, n: int) -> Column:
    validate_shift(n)
    length = len(column)
    return _gather(column, [i - n if 0 <= i - n < length else None for i in range(length)])


def over(column: Column, op: _AggregateOp, keys: Sequence[Column]) -> Column:
    if not keys or any(len(key) != len(column) for key in keys):
        raise ShapeError("over requires grouping keys of the operand height")
    dtype, factory = _reduction(op, column.dtype)
    states = []
    indices = []
    with localcontext(_WELFORD_CONTEXT):
        for row, group in enumerate(group_ids(keys, len(column))):
            if group == len(states):
                states.append(factory())
            states[group].update(column.get(row))
            indices.append(group)
        reduced = Column((state.finish() for state in states), dtype)
    return reduced.take(indices)
