import operator
from collections.abc import Callable
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from math import copysign, inf, isnan, nan

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from hypothesis import given
from hypothesis import strategies as st

from miniframe.column import Column
from miniframe.dtype import (
    INT64_MAX,
    INT64_MIN,
    Boolean,
    Categorical,
    DType,
    Float64,
    Int64,
    String,
)
from miniframe.errors import ComputeError, SchemaError, ShapeError
from miniframe.kernels.arithmetic import add, divide, multiply, negate, subtract

type Number = int | float
type Operand = Column | Number | None
type Kernel = Callable[[Operand, Operand], Column]

binary_kernels = (add, subtract, multiply, divide)


def assert_result(column: Column, dtype: DType, expected: list[Number | None]) -> None:
    column.check_invariants()
    assert column.dtype is dtype
    assert len(column) == len(expected)
    assert column.null_count == expected.count(None)
    for actual, want in zip(column.to_list(), expected, strict=True):
        if want is None:
            assert actual is None
        elif isnan(want):
            assert isnan(actual)
        else:
            assert actual == want
            if want == 0.0:
                assert copysign(1.0, actual) == copysign(1.0, want)


def snapshot(column: Column) -> tuple[bytes, bytes]:
    return column._values.tobytes(), column._validity.to_bytes()


@pytest.mark.parametrize(
    ("kernel", "forward", "reverse"),
    [(add, 9, 9), (subtract, 3, -3), (multiply, 18, 18), (divide, 2.0, 0.5)],
)
def test_promotion_and_operand_order(kernel: Kernel, forward: Number, reverse: Number) -> None:
    integers = Column([6, None, 6], Int64)
    floats = Column([3.0, 3.0, None], Float64)
    assert_result(kernel(integers, floats), Float64, [forward, None, None])
    assert_result(kernel(floats, integers), Float64, [reverse, None, None])
    for scalar in (3, 3.0):
        dtype = Float64 if type(scalar) is float or kernel is divide else Int64
        assert_result(kernel(integers, scalar), dtype, [forward, None, forward])
        assert_result(kernel(scalar, integers), dtype, [reverse, None, reverse])


@pytest.mark.parametrize("dtype", [Int64, Float64])
@pytest.mark.parametrize("length", [0, 9])
def test_empty_and_all_null(dtype: DType, length: int) -> None:
    nulls = Column([None] * length, dtype)
    extreme = INT64_MIN if dtype is Int64 else -inf
    known = Column([extreme] * length, dtype)
    assert_result(negate(nulls), dtype, [None] * length)
    for kernel in binary_kernels:
        result_dtype = Float64 if kernel is divide else dtype
        for left, right in ((nulls, known), (known, nulls), (None, known), (known, None)):
            assert_result(kernel(left, right), result_dtype, [None] * length)


def test_broadcast_results_own_their_storage() -> None:
    column = Column([None, 2, 5], Int64)
    before = snapshot(column)
    for kernel in binary_kernels:
        for scalar in (2, None):
            for left, right in ((column, scalar), (scalar, column)):
                result = kernel(left, right)
                result.check_invariants()
                assert result._values is not column._values
                assert result._validity._data is not column._validity._data
    assert snapshot(column) == before


def test_ieee_division() -> None:
    numerators = Column([1.0, -1.0, 0.0, -0.0, inf, -inf, nan, None], Float64)
    for zero in (0.0, -0.0):
        signed_inf = copysign(inf, zero)
        expected = [signed_inf, -signed_inf, nan, nan, signed_inf, -signed_inf, nan, None]
        assert_result(divide(numerators, zero), Float64, expected)
    divisors = Column([2.0, -2.0, inf, -inf, nan, None], Float64)
    assert_result(divide(-0.0, divisors), Float64, [-0.0, 0.0, -0.0, 0.0, nan, None])
    assert_result(divide(inf, divisors), Float64, [inf, -inf, nan, nan, nan, None])
    assert_result(negate(numerators), Float64, [-1.0, 1.0, -0.0, 0.0, -inf, inf, nan, None])


def test_division_converts_integers_before_computing() -> None:
    rounded = 2**53 + 1
    assert_result(divide(Column([rounded], Int64), 3), Float64, [float(rounded) / 3.0])


@pytest.mark.parametrize(
    ("kernel", "left", "right"),
    [
        (add, INT64_MAX, 1),
        (add, INT64_MIN, -1),
        (subtract, INT64_MIN, 1),
        (subtract, INT64_MAX, -1),
        (multiply, INT64_MAX, 2),
        (multiply, INT64_MIN, 2),
    ],
)
def test_int64_overflow(kernel: Kernel, left: int, right: int) -> None:
    column = Column([None, 0, left], Int64)
    before = snapshot(column)
    with pytest.raises(ComputeError, match="^Int64 overflow$"):
        kernel(column, right)
    assert snapshot(column) == before


def test_int64_limits() -> None:
    limits = Column([INT64_MIN, None, INT64_MAX], Int64)
    for kernel, identity in ((add, 0), (subtract, 0), (multiply, 1)):
        assert_result(kernel(limits, identity), Int64, [INT64_MIN, None, INT64_MAX])
    with pytest.raises(ComputeError, match="^Int64 overflow$"):
        negate(limits)
    assert_result(
        negate(Column([-INT64_MAX, None, INT64_MAX], Int64)), Int64, [INT64_MAX, None, -INT64_MAX]
    )
    assert_result(subtract(Column([None], Int64), INT64_MIN), Int64, [None])


@pytest.mark.parametrize("kernel", binary_kernels)
def test_binary_validation(kernel: Kernel) -> None:
    empty = Column([], Float64)
    with pytest.raises(TypeError):
        kernel(1, 2.0)
    with pytest.raises(ShapeError):
        kernel(Column([1], Int64), Column([2, 3], Int64))
    for invalid in (True, "4", object()):
        for left, right in ((invalid, empty), (empty, invalid)):
            with pytest.raises(TypeError):
                kernel(left, right)
    for oversized in (INT64_MIN - 1, INT64_MAX + 1):
        for left, right in ((oversized, empty), (empty, oversized)):
            with pytest.raises(ComputeError):
                kernel(left, right)


@pytest.mark.parametrize("dtype", [Boolean, String, Categorical])
def test_unsupported_dtype(dtype: DType) -> None:
    bad, good = Column([], dtype), Column([], Int64)
    with pytest.raises(SchemaError):
        negate(bad)
    for kernel in binary_kernels:
        for left, right in ((bad, good), (good, bad)):
            with pytest.raises(SchemaError):
                kernel(left, right)


def test_negate_requires_column() -> None:
    with pytest.raises(TypeError):
        negate(1)


def scalar_result(kernel: Kernel, left: Number | None, right: Number | None) -> Number | None:
    if left is None or right is None:
        return None
    if kernel is divide:
        a, b = float(left), float(right)
        if b == 0.0:
            with localcontext() as context:
                context.traps[DivisionByZero] = context.traps[InvalidOperation] = False
                return float(Decimal.from_float(a) / Decimal.from_float(b))
        return a / b
    return {add: operator.add, subtract: operator.sub, multiply: operator.mul}[kernel](left, right)


def numbers(dtype: DType) -> st.SearchStrategy[Number | None]:
    if dtype is Float64:
        return st.none() | st.floats()
    return st.none() | st.integers(INT64_MIN, INT64_MAX) | st.sampled_from([INT64_MIN, INT64_MAX])


@given(data=st.data())
def test_binary_kernels_against_scalar_reference(data: st.DataObject) -> None:
    numeric_dtype = st.sampled_from((Int64, Float64))
    left_dtype, right_dtype = data.draw(st.tuples(numeric_dtype, numeric_dtype))
    rows = data.draw(st.lists(st.tuples(numbers(left_dtype), numbers(right_dtype)), max_size=40))
    a, b = [x for x, _ in rows], [y for _, y in rows]
    left, right = Column(a, left_dtype), Column(b, right_dtype)
    inputs = snapshot(left), snapshot(right)
    for kernel in binary_kernels:
        dtype = Float64 if kernel is divide or Float64 in (left_dtype, right_dtype) else Int64
        expected = [scalar_result(kernel, x, y) for x, y in rows]
        if dtype is Int64 and any(v is not None and not -(2**63) <= v < 2**63 for v in expected):
            with pytest.raises(ComputeError):
                kernel(left, right)
        else:
            result = kernel(left, right)
            assert_result(result, dtype, expected)
            for source in (left, right):
                assert result._values is not source._values
                assert result._validity._data is not source._validity._data
        assert (snapshot(left), snapshot(right)) == inputs


@given(data=st.data())
def test_negation_and_double_negation(data: st.DataObject) -> None:
    dtype = data.draw(st.sampled_from((Int64, Float64)))
    values = data.draw(st.lists(numbers(dtype), max_size=40))
    column = Column(values, dtype)
    before = snapshot(column)
    if dtype is Int64 and INT64_MIN in values:
        with pytest.raises(ComputeError):
            negate(column)
    else:
        result = negate(column)
        assert_result(result, dtype, [None if x is None else -x for x in values])
        assert_result(negate(result), dtype, values)
        assert result._values is not column._values
        assert result._validity._data is not column._validity._data
    assert snapshot(column) == before


@given(st.lists(st.tuples(st.integers(-10000, 10000) | st.none(), st.integers(1, 10000))))
def test_finite_arithmetic_against_arrow(rows: list[tuple[int | None, int]]) -> None:
    a, b = [x for x, _ in rows], [y for _, y in rows]
    left, right = Column(a, Int64), Column(b, Int64)
    x, y = pa.array(a, type=pa.int64()), pa.array(b, type=pa.int64())
    for kernel, arrow in ((add, pc.add), (subtract, pc.subtract), (multiply, pc.multiply)):
        assert_result(kernel(left, right), Int64, arrow(x, y).to_pylist())
    expected = pc.divide(x.cast(pa.float64()), y.cast(pa.float64()))
    assert_result(divide(left, right), Float64, expected.to_pylist())
