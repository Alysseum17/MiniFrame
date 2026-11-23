from collections.abc import Callable
from decimal import localcontext
from itertools import accumulate
from math import copysign, fsum, inf, isclose, isnan, nan, sqrt, ulp
from statistics import mean, stdev, variance
from sys import float_info

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from miniframe import Boolean, Categorical, ComputeError, DType, Float64, Int64, SchemaError, String
from miniframe.column import Column
from miniframe.dtype import INT64_MAX, INT64_MIN
from miniframe.kernels.aggregate import (
    aggregate_count,
    aggregate_first,
    aggregate_last,
    aggregate_len,
    aggregate_max,
    aggregate_mean,
    aggregate_min,
    aggregate_n_unique,
    aggregate_std,
    aggregate_sum,
    aggregate_var,
)

NUMERIC = (aggregate_sum, aggregate_mean, aggregate_var, aggregate_std)
EXTREMA = (aggregate_min, aggregate_max)
GENERAL = (aggregate_count, aggregate_len, aggregate_n_unique, aggregate_first, aggregate_last)
TOLERANCE = 8 * ulp(1.0)


@pytest.mark.parametrize("kernel", (*NUMERIC, *EXTREMA, *GENERAL))
def test_kernels_require_columns(kernel: Callable) -> None:
    with pytest.raises(TypeError, match="^expected a Column$"):
        kernel([])


@pytest.mark.parametrize("dtype", list(DType))
@pytest.mark.parametrize("length", [0, 1, 9])
def test_empty_and_all_null(dtype: DType, length: int) -> None:
    column = Column([None] * length, dtype)
    assert tuple(f(column) for f in GENERAL) == (0, length, int(length > 0), None, None)
    if dtype.is_numeric:
        total = aggregate_sum(column)
        assert type(total) is (int if dtype is Int64 else float)
        assert (total, aggregate_mean(column), aggregate_var(column), aggregate_std(column)) == (
            0,
            None,
            None,
            None,
        )
    else:
        for kernel in NUMERIC:
            with pytest.raises(SchemaError):
                kernel(column)
    for kernel in EXTREMA:
        if dtype is Boolean:
            with pytest.raises(SchemaError):
                kernel(column)
        else:
            assert kernel(column) is None


@pytest.mark.parametrize(
    ("dtype", "value"),
    [(Int64, -3), (Float64, -0.0), (Boolean, False), (String, ""), (Categorical, "🙂")],
)
def test_single_valid_value(dtype: DType, value: object) -> None:
    column = Column([value], dtype)
    assert tuple(f(column) for f in GENERAL) == (1, 1, 1, value, value)
    if dtype is not Boolean:
        assert tuple(f(column) for f in EXTREMA) == (value, value)
    if dtype.is_numeric:
        padded = Column([None, value, None], dtype)
        assert tuple(f(padded) for f in NUMERIC) == (value, value, None, None)


@pytest.mark.parametrize("dtype", [String, Categorical])
def test_text_aggregates_after_slice(dtype: DType) -> None:
    column = Column(["unused", "🙂", None, "Київ", "", "🙂", None], dtype).slice(1, 6)
    assert tuple(f(column) for f in EXTREMA) == ("", "🙂")
    assert tuple(f(column) for f in GENERAL) == (4, 6, 4, "🙂", None)
    assert column.to_list() == ["🙂", None, "Київ", "", "🙂", None]
    column.check_invariants()


@pytest.mark.parametrize("special", [inf, -inf, nan])
def test_nonfinite_samples(special: float) -> None:
    singleton = Column([None, special], Float64)
    assert aggregate_var(singleton) is aggregate_std(singleton) is None
    column = Column([None, 1.0, special, special], Float64)
    assert aggregate_count(column) == 3
    assert aggregate_n_unique(column) == 3
    assert isnan(aggregate_std(column)) and isnan(aggregate_var(column))
    for kernel in (aggregate_sum, aggregate_mean):
        result = kernel(column)
        if isnan(special):
            assert isnan(result)
        else:
            assert result == special
        assert isnan(kernel(Column([-inf, inf], Float64)))


def test_nan_order_distinct_keys_and_endpoints() -> None:
    values = [float("nan"), -0.0, 0.0, -inf, None, inf, float("nan")]
    for ordered in (values, list(reversed(values))):
        column = Column(ordered, Float64)
        assert aggregate_min(column) == -inf
        assert isnan(aggregate_max(column))
        assert aggregate_n_unique(column) == 5
        assert isnan(aggregate_first(column)) and isnan(aggregate_last(column))
    assert isnan(aggregate_min(Column([nan, None, nan], Float64)))
    zeros = Column([-0.0, None, 0.0], Float64)
    assert [copysign(1.0, f(zeros)) for f in (aggregate_first, aggregate_last)] == [-1.0, 1.0]


@pytest.mark.parametrize("bound,step", [(INT64_MAX, 1), (INT64_MIN, -1)])
def test_int64_intermediate_overflow(bound: int, step: int) -> None:
    column = Column([bound, None, step, -step], Int64)
    with pytest.raises(ComputeError, match="^Int64 overflow$"):
        aggregate_sum(column)
    assert aggregate_mean(column) == bound / 3
    assert aggregate_sum(Column([bound, -step, step], Int64)) == bound


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (Int64, [INT64_MAX - 1, INT64_MAX]),
        (Int64, [INT64_MIN, INT64_MIN + 1, INT64_MIN + 2]),
        (Float64, [1_000_000_000.001, 1_000_000_000.002, 1_000_000_000.003, 1_000_000_000.004]),
        (Float64, [1e300] * 10),
        (Int64, list(range(-5000, 5001))),
    ],
    ids=["adjacent-max", "adjacent-min", "small-spread", "constant", "large-sample"],
)
def test_welford_against_exact_rational_oracle(dtype: DType, values: list[int | float]) -> None:
    column = Column(values, dtype)
    with localcontext(prec=3):
        assert isclose(aggregate_var(column), variance(values), rel_tol=TOLERANCE)
        assert isclose(aggregate_std(column), stdev(values), rel_tol=TOLERANCE)


def test_float64_extremes() -> None:
    largest = float_info.max
    for sign in (1, -1):
        value = sign * largest
        column = Column([value, None, value], Float64)
        assert aggregate_sum(column) == sign * inf
        assert aggregate_mean(column) == value
        assert aggregate_var(column) == aggregate_std(column) == 0.0
        cancellation = Column([value, 1.0, -value], Float64)
        assert aggregate_sum(cancellation) == 1.0
        assert aggregate_mean(cancellation) == 1 / 3
    for magnitude in (1e200, 1e-200):
        column = Column([-magnitude, magnitude], Float64)
        assert aggregate_var(column) == (inf if magnitude > 1 else 0.0)
        assert isclose(aggregate_std(column), sqrt(2) * magnitude, rel_tol=TOLERANCE)


@pytest.mark.parametrize(
    ("dtype", "scalars"),
    [
        (Int64, st.integers(INT64_MIN, INT64_MAX) | st.sampled_from([INT64_MIN, INT64_MAX])),
        (Float64, st.floats(-1e150, 1e150)),
    ],
)
@given(data=st.data())
@settings(derandomize=True)
def test_nullable_numeric_reductions(
    dtype: DType, scalars: st.SearchStrategy, data: st.DataObject
) -> None:
    values = data.draw(st.lists(st.none() | scalars))
    valid = [v for v in values if v is not None]
    overflow = dtype is Int64 and any(not INT64_MIN <= t <= INT64_MAX for t in accumulate(valid))
    for rows in (values, [None, *values, None]):
        column = Column(rows, dtype)
        snapshot = column._values.tobytes(), column._validity.to_bytes()
        length = aggregate_len(column)
        assert aggregate_count(column) + column.null_count == length == len(rows)
        assert 0 <= aggregate_n_unique(column) == len(set(rows)) <= length
        assert aggregate_first(column) == (rows[0] if rows else None)
        assert aggregate_last(column) == (rows[-1] if rows else None)
        assert aggregate_min(column) == min(valid, default=None)
        assert aggregate_max(column) == max(valid, default=None)
        if overflow:
            with pytest.raises(ComputeError):
                aggregate_sum(column)
        elif dtype is Int64:
            assert aggregate_sum(column) == sum(valid)
        else:
            assert aggregate_sum(column) == fsum(valid)
        assert aggregate_mean(column) == (float(mean(valid)) if valid else None)
        for kernel, reference in ((aggregate_var, variance), (aggregate_std, stdev)):
            result = kernel(column)
            if len(valid) < 2:
                assert result is None
            else:
                assert result >= 0.0
                assert isclose(result, reference(valid), rel_tol=TOLERANCE)
        assert (column._values.tobytes(), column._validity.to_bytes()) == snapshot
        column.check_invariants()


@given(st.lists(st.none() | st.floats() | st.just(float("nan"))))
@settings(derandomize=True)
def test_float_distinct_count(values: list[float | None]) -> None:
    ordinary = {v for v in values if v is not None and not isnan(v)}
    expected = len(ordinary) + (None in values) + any(v is not None and isnan(v) for v in values)
    assert aggregate_n_unique(Column(values, Float64)) == expected


@pytest.mark.parametrize("dtype", [Int64, Float64])
def test_numeric_reductions_against_arrow(dtype: DType) -> None:
    convert = int if dtype is Int64 else float
    rows = [None, convert(5), convert(-2), convert(0), convert(5), None]
    column = Column(rows, dtype)
    arrow = pa.array(rows)
    assert aggregate_sum(column) == pc.sum(arrow).as_py()
    assert aggregate_mean(column) == pc.mean(arrow).as_py()
    assert aggregate_min(column) == pc.min(arrow).as_py()
    assert aggregate_max(column) == pc.max(arrow).as_py()
    assert aggregate_count(column) == pc.count(arrow).as_py()
    assert aggregate_n_unique(column) == pc.count_distinct(arrow, mode="all").as_py()
    assert isclose(aggregate_var(column), pc.variance(arrow, ddof=1).as_py(), rel_tol=TOLERANCE)
    assert isclose(aggregate_std(column), pc.stddev(arrow, ddof=1).as_py(), rel_tol=TOLERANCE)
