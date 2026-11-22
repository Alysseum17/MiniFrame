from array import array
from itertools import product
from math import copysign, inf, isnan, nan, nextafter

import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st

from miniframe.bitmap import Bitmap
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
from miniframe.errors import ComputeError, SchemaError
from miniframe.kernels.cast import cast_column

type Scalar = int | float | bool | str | None

samples = {Int64: -2, Float64: -2.0, Boolean: True, String: "-2", Categorical: "-2"}
conversions = {
    (Int64, Float64): -2.0,
    (Float64, Int64): -2,
    (String, Int64): -2,
    (String, Float64): -2.0,
    (String, Categorical): "-2",
    (Categorical, String): "-2",
}


def buffers(column: Column) -> tuple[bytearray | array, ...]:
    values = column._values
    validity = column._validity._data
    if isinstance(values, Bitmap):
        return validity, values._data
    if isinstance(values, array):
        return validity, values
    if column.dtype is String:
        return validity, values.offsets, values.data
    return validity, values.codes


def assert_result(column: Column, dtype: DType, expected: list[Scalar]) -> None:
    column.check_invariants()
    assert column.dtype is dtype
    assert column.null_count == expected.count(None)
    assert len(column) == len(expected)
    for actual, value in zip(column.to_list(), expected, strict=True):
        if isinstance(value, float) and isnan(value):
            assert isnan(actual)
        else:
            assert actual == value
            if isinstance(value, float) and value == 0.0:
                assert copysign(1.0, actual) == copysign(1.0, value)


@pytest.mark.parametrize(("source", "target"), list(product(DType, repeat=2)))
def test_conversion_matrix_and_ownership(source: DType, target: DType) -> None:
    for values in ([], [None], [None, samples[source]]):
        column = Column(values, source)
        storage = buffers(column)
        before = tuple(bytes(buf) for buf in storage)
        if source is target or (source, target) in conversions:
            result = cast_column(column, target)
            expected = [None if v is None else conversions.get((source, target), v) for v in values]
            assert_result(result, target, expected)
            assert result is not column
            assert all(buf is not original for buf in buffers(result) for original in storage)
        else:
            with pytest.raises(ComputeError):
                cast_column(column, target)
        assert tuple(bytes(buf) for buf in buffers(column)) == before
        assert_result(column, source, values)


def test_float_identity_and_numeric_limits() -> None:
    values = [nan, inf, -inf, -0.0, 0.0, None]
    source = Column(values, Float64)
    copy = cast_column(source, Float64)
    assert_result(copy, Float64, values)
    assert bytes(copy._values) == bytes(source._values)
    exact = [INT64_MIN, 2**53 - 1, 2**53, 2**53 + 2, INT64_MAX - 1023]
    floats = cast_column(Column(exact, Int64), Float64)
    assert_result(floats, Float64, list(map(float, exact)))
    assert_result(cast_column(floats, Int64), Int64, exact)
    assert_result(cast_column(Column([-0.0, None], Float64), Int64), Int64, [0, None])


@pytest.mark.parametrize("value", [INT64_MAX, INT64_MIN + 1, 2**53 + 1, -(2**53 + 1)])
def test_integer_precision_loss(value: int) -> None:
    with pytest.raises(
        ComputeError, match="^Int64 to Float64 at position 2: integer loses precision in Float64$"
    ):
        cast_column(Column([0, None, value], Int64), Float64)


@pytest.mark.parametrize(
    "value", [0.5, -1.5, nan, inf, -inf, float(2**63), nextafter(float(INT64_MIN), -inf)]
)
def test_invalid_float_to_int(value: float) -> None:
    with pytest.raises(ComputeError):
        cast_column(Column([None, value], Float64), Int64)


def test_decimal_grammar_and_limits() -> None:
    integers = [str(INT64_MIN), str(INT64_MAX), "+00012", "-0", "0" * 5000 + "7", None]
    assert_result(
        cast_column(Column(integers, String), Int64), Int64, [INT64_MIN, INT64_MAX, 12, 0, 7, None]
    )
    floats = ["-0", ".5", "1.", "-.5E+2", "0.1", "0e-9999", "5e-324", None]
    assert_result(
        cast_column(Column(floats, String), Float64),
        Float64,
        [-0.0, 0.5, 1.0, -50.0, 0.1, 0.0, 5e-324, None],
    )


@pytest.mark.parametrize("target", [Int64, Float64])
def test_invalid_numeric_text(target: DType) -> None:
    invalid = ["", " ", " 1", "1\n", "1_0", "٥", "0xff", "+", "1e", "true", "inf", "NaN", "🍋"]
    if target is Int64:
        invalid += [str(INT64_MAX + 1), str(INT64_MIN - 1), "1.0", "2e3", "9" * 5000]
    else:
        invalid += ["1e309", "-1e-9999"]
    for value in invalid:
        column = Column(["1", None, value], String)
        with pytest.raises(ComputeError):
            cast_column(column, target)
        assert_result(column, String, ["1", None, value])


def test_text_dictionary_order_and_utf8() -> None:
    values = [None, "🍋", "", "Київ", "🍋", "", None]
    encoded = cast_column(Column(values, String), Categorical)
    assert_result(encoded, Categorical, values)
    assert encoded._values.dictionary == ("🍋", "", "Київ")
    assert list(encoded._values.codes) == [0, 0, 1, 2, 0, 1, 0]
    assert_result(cast_column(encoded, String), String, values)
    with pytest.raises(ComputeError):
        cast_column(Column(["\ud800"], Categorical), String)


def test_bad_cast_arguments() -> None:
    with pytest.raises(SchemaError):
        cast_column(Column([], Int64), "Float64")
    with pytest.raises(TypeError):
        cast_column(None, Int64)


@given(st.lists(st.integers(INT64_MIN, INT64_MAX) | st.none(), max_size=40))
def test_int_text_and_exact_float_roundtrip(values: list[int | None]) -> None:
    text = Column((None if v is None else str(v) for v in values), String)
    integers = cast_column(text, Int64)
    assert_result(integers, Int64, values)
    if any(v is not None and int(float(v)) != v for v in values):
        with pytest.raises(ComputeError):
            cast_column(integers, Float64)
    else:
        floats = cast_column(integers, Float64)
        expected = pa.array(values, type=pa.int64()).cast(pa.float64(), safe=False).to_pylist()
        assert_result(floats, Float64, expected)
        assert_result(cast_column(floats, Int64), Int64, values)


@given(st.lists(st.text() | st.none(), max_size=40))
def test_dictionary_roundtrip(values: list[str | None]) -> None:
    encoded = cast_column(Column(values, String), Categorical)
    assert_result(encoded, Categorical, values)
    assert_result(cast_column(encoded, String), String, values)
    expected = pa.array(values, type=pa.string()).dictionary_encode()
    assert list(encoded._values.dictionary) == expected.dictionary.to_pylist()


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False) | st.none(), max_size=40))
def test_float_text_roundtrip_and_integer_cast(values: list[float | None]) -> None:
    strings = Column((None if v is None else repr(v) for v in values), String)
    floats = cast_column(strings, Float64)
    assert_result(floats, Float64, values)
    expected = [None if v is None else int(v) for v in values]
    if any(
        v is not None and (v != i or not INT64_MIN <= v <= INT64_MAX)
        for v, i in zip(values, expected, strict=True)
    ):
        with pytest.raises(ComputeError):
            cast_column(floats, Int64)
    else:
        assert_result(cast_column(floats, Int64), Int64, expected)
