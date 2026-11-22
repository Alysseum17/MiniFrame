"""Scalar reductions over nullable columns."""

from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
from math import inf, isfinite, isnan
from typing import Literal, cast

from ..column import Column, _Scalar
from ..dtype import Boolean, Float64, Int64
from ..errors import SchemaError
from .arithmetic import _checked_int64

type _AggregateOp = Literal[
    "sum", "mean", "min", "max", "count", "len", "n_unique", "std", "var", "first", "last"
]
type _Number = int | float
type _Value = int | float | bool | str

_NAN_KEY = object()
_WELFORD_CONTEXT = Context(prec=50)


def _require_column(column: Column) -> None:
    if not isinstance(column, Column):
        raise TypeError("expected a Column")


def _valid_values(column: Column) -> Iterator[_Value]:
    values = column._values
    if isinstance(values, array):
        if column.null_count == 0:
            yield from values
        else:
            for i, value in enumerate(values):
                if column._validity.get(i):
                    yield value
    else:
        for i in range(len(column)):
            value = column.get(i)
            if value is not None:
                yield value


def _numeric_values(column: Column) -> Iterator[_Number]:
    _require_column(column)
    if not column.dtype.is_numeric:
        raise SchemaError(f"expected numeric dtype, got {column.dtype.value}")
    return cast(Iterator[_Number], _valid_values(column))


@dataclass(slots=True)
class _FloatTotal:
    total: int = 0
    scale: int = 1
    nonfinite: float = 0.0

    def update(self, value: float) -> None:
        if not isfinite(value):
            self.nonfinite += value
            return
        numerator, denominator = value.as_integer_ratio()
        if denominator > self.scale:
            self.total *= denominator // self.scale
            self.scale = denominator
        self.total += numerator * (self.scale // denominator)

    def divided_by(self, divisor: int) -> float:
        if self.nonfinite != 0.0:
            return self.nonfinite
        try:
            return self.total / (self.scale * divisor)
        except OverflowError:
            return inf if self.total > 0 else -inf


def _float_total(values: Iterable[float], divisor: int) -> float:
    state = _FloatTotal()
    for value in values:
        state.update(value)
    return state.divided_by(divisor)


def aggregate_sum(column: Column) -> int | float:
    """Skip nulls and return a typed zero for empty input. Check every Int64 prefix."""
    values = _numeric_values(column)
    if column.dtype is Float64:
        return _float_total(cast(Iterable[float], values), 1)
    total = 0
    for value in cast(Iterable[int], values):
        total = _checked_int64(total + value)
    return total


def aggregate_mean(column: Column) -> float | None:
    """Return the mean of the valid numeric values, or None when there are none."""
    values = _numeric_values(column)
    count = len(column) - column.null_count
    if count == 0:
        return None
    if column.dtype is Int64:
        return sum(cast(Iterable[int], values)) / count
    return _float_total(cast(Iterable[float], values), count)


@dataclass(slots=True)
class _Welford:
    """Online sample moments updated and finalized in the extended-precision context."""

    count: int = 0
    mean: Decimal = Decimal(0)
    m2: Decimal = Decimal(0)

    def update(self, value: _Number) -> None:
        self.count += 1
        if not isfinite(value):
            self.m2 = Decimal("NaN")
            return
        x = +Decimal.from_float(value)
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def sample_variance(self) -> Decimal | None:
        if self.count < 2:
            return None
        assert self.m2.is_nan() or self.m2 >= 0, "negative Welford sum of squared deviations"
        return self.m2 / (self.count - 1)


def _sample_variance(column: Column) -> Decimal | None:
    values = _numeric_values(column)
    if len(column) - column.null_count < 2:
        return None
    with localcontext(_WELFORD_CONTEXT):
        state = _Welford()
        for value in values:
            state.update(value)
        return state.sample_variance()


def aggregate_var(column: Column) -> float | None:
    """Sample variance (ddof=1). Any nonfinite observation in a sample yields NaN."""
    variance = _sample_variance(column)
    return None if variance is None else float(variance)


def aggregate_std(column: Column) -> float | None:
    """Sample standard deviation, taking the square root before rounding to Float64."""
    variance = _sample_variance(column)
    return None if variance is None else float(variance.sqrt(context=_WELFORD_CONTEXT))


def _float_order(value: float) -> tuple[bool, float]:
    return isnan(value), value


def _extreme(column: Column, *, maximum: bool) -> _Scalar:
    _require_column(column)
    if column.dtype is Boolean:
        raise SchemaError("Boolean does not support ordering")
    reduce = max if maximum else min
    values = _valid_values(column)
    if column.dtype is Float64:
        return reduce(cast(Iterator[float], values), key=_float_order, default=None)
    return reduce(values, default=None)


def aggregate_min(column: Column) -> _Scalar:
    """Minimum valid numeric or text value, ordering NaN above all numbers."""
    return _extreme(column, maximum=False)


def aggregate_max(column: Column) -> _Scalar:
    """Maximum valid numeric or text value, ordering NaN above all numbers."""
    return _extreme(column, maximum=True)


def aggregate_count(column: Column) -> int:
    """Count non-null positions using column metadata."""
    _require_column(column)
    return len(column) - column.null_count


def aggregate_len(column: Column) -> int:
    """Count all positions, including nulls."""
    _require_column(column)
    return len(column)


def _distinct_key(value: _Scalar) -> object:
    return _NAN_KEY if isinstance(value, float) and isnan(value) else value


def _distinct_indices(columns: tuple[Column, ...], height: int) -> list[int]:
    if not columns:
        return [0] if height else []
    seen = set()
    indices = []
    for i in range(height):
        key = tuple(_distinct_key(column.get(i)) for column in columns)
        if key not in seen:
            seen.add(key)
            indices.append(i)
    return indices


def aggregate_n_unique(column: Column) -> int:
    """Count distinct logical values, grouping nulls, NaNs, and signed zeros respectively."""
    _require_column(column)
    distinct = {_distinct_key(value) for value in _valid_values(column)}
    return len(distinct) + (column.null_count > 0)


def aggregate_first(column: Column) -> _Scalar:
    """Return the first position, including null, or None for an empty column."""
    _require_column(column)
    return column.get(0) if len(column) else None


def aggregate_last(column: Column) -> _Scalar:
    """Return the last position, including null, or None for an empty column."""
    _require_column(column)
    return column.get(len(column) - 1) if len(column) else None
