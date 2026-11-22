"""Stable hash grouping and streaming reductions over prepared columns."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from decimal import localcontext
from functools import partial
from typing import Protocol, cast

from ..column import Column, _Scalar
from ..dtype import Boolean, DType, Float64, Int64
from ..errors import SchemaError
from ..kernels.aggregate import (
    _WELFORD_CONTEXT,
    _AggregateOp,
    _distinct_key,
    _float_order,
    _FloatTotal,
    _Welford,
)
from ..kernels.arithmetic import _checked_int64


class _Accumulator(Protocol):
    def update(self, value: _Scalar) -> None: ...
    def finish(self) -> _Scalar: ...


@dataclass(slots=True)
class _IntegerTotal:
    mean: bool
    total: int = 0
    count: int = 0

    def update(self, value: _Scalar) -> None:
        if value is not None:
            total = self.total + cast(int, value)
            self.total = total if self.mean else _checked_int64(total)
            self.count += 1

    def finish(self) -> int | float | None:
        if self.mean:
            return self.total / self.count if self.count else None
        return self.total


@dataclass(slots=True)
class _FloatReduction:
    mean: bool
    total: _FloatTotal = field(default_factory=_FloatTotal)
    count: int = 0

    def update(self, value: _Scalar) -> None:
        if value is not None:
            self.total.update(cast(float, value))
            self.count += 1

    def finish(self) -> float | None:
        if self.mean and self.count == 0:
            return None
        return self.total.divided_by(self.count if self.mean else 1)


@dataclass(slots=True)
class _Moments:
    std: bool
    state: _Welford = field(default_factory=_Welford)

    def update(self, value: _Scalar) -> None:
        if value is not None:
            self.state.update(cast(int | float, value))

    def finish(self) -> float | None:
        variance = self.state.sample_variance()
        if variance is None:
            return None
        return float(variance.sqrt() if self.std else variance)


@dataclass(slots=True)
class _Count:
    include_null: bool
    count: int = 0

    def update(self, value: _Scalar) -> None:
        if self.include_null or value is not None:
            self.count += 1

    def finish(self) -> int:
        return self.count


@dataclass(slots=True)
class _Distinct:
    seen: set[object] = field(default_factory=set)

    def update(self, value: _Scalar) -> None:
        self.seen.add(_distinct_key(value))

    def finish(self) -> int:
        return len(self.seen)


@dataclass(slots=True)
class _Endpoint:
    first: bool
    initialized: bool = False
    value: _Scalar = None

    def update(self, value: _Scalar) -> None:
        if not self.first or not self.initialized:
            self.value = value
            self.initialized = True

    def finish(self) -> _Scalar:
        return self.value


@dataclass(slots=True)
class _Extreme:
    maximum: bool
    value: _Scalar = None

    def update(self, value: _Scalar) -> None:
        if value is None:
            return
        current = self.value
        if current is None:
            self.value = value
            return
        if isinstance(value, float):
            incoming, previous = _float_order(value), _float_order(cast(float, current))
            replace = incoming > previous if self.maximum else incoming < previous
        else:
            replace = value > current if self.maximum else value < current
        if replace:
            self.value = value

    def finish(self) -> _Scalar:
        return self.value


def _reduction(op: _AggregateOp, dtype: DType) -> tuple[DType, Callable[[], _Accumulator]]:
    if op in ("sum", "mean", "var", "std") and not dtype.is_numeric:
        raise SchemaError(f"{op} requires a numeric operand")
    if op in ("sum", "mean"):
        factory = _IntegerTotal if dtype is Int64 else _FloatReduction
        return (Float64 if op == "mean" else dtype), partial(factory, op == "mean")
    if op in ("var", "std"):
        return Float64, partial(_Moments, op == "std")
    if op in ("min", "max"):
        if dtype is Boolean:
            raise SchemaError("Boolean does not support ordering")
        return dtype, partial(_Extreme, op == "max")
    if op in ("first", "last"):
        return dtype, partial(_Endpoint, op == "first")
    if op == "n_unique":
        return Int64, _Distinct
    return Int64, partial(_Count, op == "len")


def group_ids(keys: Sequence[Column], height: int) -> Iterator[int]:
    groups: dict[tuple[object, ...], int] = {}
    for row in range(height):
        key = tuple(_distinct_key(column.get(row)) for column in keys)
        yield groups.setdefault(key, len(groups))


def hash_aggregate(
    keys: dict[str, Column], operands: dict[str, tuple[_AggregateOp, Column]], height: int
) -> dict[str, Column]:
    reductions = [_reduction(op, column.dtype) for op, column in operands.values()]
    inputs = [column for _, column in operands.values()]
    first_rows: list[int] = []
    states: list[list[_Accumulator]] = []
    with localcontext(_WELFORD_CONTEXT):
        for row, group in enumerate(group_ids(tuple(keys.values()), height)):
            if group == len(states):
                first_rows.append(row)
                states.append([factory() for _, factory in reductions])
            for state, column in zip(states[group], inputs, strict=True):
                state.update(column.get(row))

        result = {name: column.take(first_rows) for name, column in keys.items()}
        for index, (name, (dtype, _)) in enumerate(zip(operands, reductions, strict=True)):
            result[name] = Column((group_states[index].finish() for group_states in states), dtype)
    return result
