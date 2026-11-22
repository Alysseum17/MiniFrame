import operator
from collections.abc import Callable, Iterator
from math import inf, nan

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from miniframe import (
    Boolean,
    Categorical,
    ComputeError,
    DType,
    Float64,
    Int64,
    SchemaError,
    Series,
    ShapeError,
    String,
)
from miniframe.column import Column
from miniframe.dtype import INT64_MAX, INT64_MIN
from miniframe.kernels import arithmetic, compare

type Scalar = int | float | bool | str | None
COMPARISONS = (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge)
SAMPLES = [(Int64, 2), (Float64, -0.0), (Boolean, False), (String, "Київ"), (Categorical, "🙂")]


def assert_series(series: Series, dtype: DType, expected: list[Scalar], name: str = "x") -> None:
    assert isinstance(series, Series)
    assert series.name == name
    assert series.dtype is dtype
    assert len(series) == len(expected)
    assert series.null_count == expected.count(None)
    for actual, wanted in zip(series.to_list(), expected, strict=True):
        assert type(actual) is type(wanted)
        if isinstance(wanted, float):
            assert actual.hex() == wanted.hex()
        else:
            assert actual == wanted
    series._column.check_invariants()


@pytest.mark.parametrize(("dtype", "value"), SAMPLES)
def test_construction_fill_and_copy(dtype: DType, value: Scalar) -> None:
    for values in ([], [None] * 3, [value, None], [value]):
        source = Series("x", iter(values), dtype)
        assert_series(source, dtype, values)
        for result in (source.slice(0, len(source)), source.cast(dtype), source.fill_null(None)):
            assert_series(result, dtype, values)
            assert result._column._values is not source._column._values
            assert result._column._validity._data is not source._column._validity._data
        assert_series(source.fill_null(value), dtype, [value] * len(values))
        assert_series(source, dtype, values)


@pytest.mark.parametrize(("dtype", "value"), SAMPLES[:-1])
def test_inference_from_one_pass_input(dtype: DType, value: Scalar) -> None:
    class Once:
        consumed = False

        def __iter__(self) -> Iterator[Scalar]:
            assert not self.consumed
            self.consumed = True
            yield from (None, None, value, None, value)

    for explicit in (None, dtype):
        assert_series(Series("x", Once(), explicit), dtype, [None, None, value, None, value])


@pytest.mark.parametrize(
    "values", [[], [None, None], [object()], [1, True], [False, 0], [1.0, 1], ["1", 1]]
)
def test_inference_requires_one_builtin_type(values: list[object]) -> None:
    with pytest.raises(SchemaError):
        Series("x", iter(values))


def test_constructor_validation_and_int64_bounds() -> None:
    with pytest.raises(SchemaError):
        Series(1, [])
    with pytest.raises(SchemaError):
        Series("x", [], "Int64")
    with pytest.raises(SchemaError):
        Series("x", [1], Boolean)
    assert_series(Series("x", [INT64_MIN, INT64_MAX]), Int64, [INT64_MIN, INT64_MAX])
    for overflow in (INT64_MIN - 1, INT64_MAX + 1):
        with pytest.raises(ComputeError, match="^Int64 overflow at position 1$"):
            Series("x", [None, overflow])


def test_public_access_and_immutability() -> None:
    values = [0, None, 12]
    source = Series("x", values)
    values.clear()
    source.to_list().clear()
    assert [source[i] for i in range(len(source))] == [0, None, 12]
    for index in (-1, 3):
        with pytest.raises(IndexError):
            source[index]
    for index in (True, 0.0, "0", slice(1, 2)):
        with pytest.raises(TypeError):
            source[index]
    for field in ("name", "dtype", "null_count"):
        with pytest.raises(AttributeError):
            setattr(source, field, None)
    with pytest.raises(TypeError):
        source[1] = 0
    with pytest.raises(IndexError):
        source.slice(1, 3)
    with pytest.raises(TypeError):
        source.slice(False, 1)


@pytest.mark.parametrize("op", (operator.add, operator.sub, operator.mul, operator.truediv))
def test_arithmetic_operands(op: Callable) -> None:
    source = Series("x", [6, None, 4])
    right = Series("y", [2.5, 8.0, None])
    assert_series(op(source, right), Float64, [op(6, 2.5), None, None])
    for a, b, expected in (
        (source, 2, [op(6, 2), None, op(4, 2)]),
        (2, source, [op(2, 6), None, op(2, 4)]),
        (None, source, [None] * 3),
        (source, None, [None] * 3),
    ):
        result = op(a, b)
        assert_series(result, Float64 if op is operator.truediv else Int64, expected)
        assert result._column._values is not source._column._values
        assert result._column._validity._data is not source._column._validity._data
    assert_series(source, Int64, [6, None, 4])
    assert_series(right, Float64, [2.5, 8.0, None], "y")


def test_numeric_edge_cases() -> None:
    assert_series(-Series("x", [3, None, -1]), Int64, [-3, None, 1])
    assert_series(Series("x", [-3, 0, 2, None]) / -0.0, Float64, [inf, nan, -inf, None])
    with pytest.raises(ComputeError):
        -Series("x", [INT64_MIN])
    with pytest.raises(ComputeError):
        Series("x", [INT64_MAX]) + 1
    with pytest.raises(TypeError):
        Series("x", [1]) * False
    with pytest.raises(SchemaError):
        Series("x", ["2"]) / 2


@pytest.mark.parametrize("op", COMPARISONS)
def test_comparisons_preserve_precision_nan_and_operand_order(op: Callable) -> None:
    integers = [2**53 + 1, 5, None, 3]
    floats = [float(2**53), nan, -0.0, None]
    source = Series("x", integers)
    expected = [op(integers[0], floats[0]), op(5, nan), None, None]
    for other in (Series("y", floats), Column(floats, Float64)):
        assert_series(op(source, other), Boolean, expected)
    assert_series(
        op(Column(floats, Float64), source),
        Boolean,
        [op(floats[0], integers[0]), op(nan, 5), None, None],
    )
    for scalar in (2.5, None):
        assert_series(
            op(source, scalar),
            Boolean,
            [None if x is None or scalar is None else op(x, scalar) for x in integers],
        )
        assert_series(
            op(scalar, source),
            Boolean,
            [None if x is None or scalar is None else op(scalar, x) for x in integers],
        )


def test_categorical_comparisons_use_logical_text() -> None:
    left = Series("x", ["я", "a", None], Categorical)
    for dtype in (String, Categorical):
        right = Series("y", ["a", "я", None], dtype)
        assert_series(left > right, Boolean, [True, False, None])
        assert_series(left == right, Boolean, [False, False, None])
    assert_series("я" == left, Boolean, [True, False, None])
    with pytest.raises(SchemaError):
        operator.eq(left, 0)
    with pytest.raises(SchemaError):
        operator.lt(Series("x", [False]), True)


@pytest.mark.parametrize(
    ("dtype", "op"), [(Int64, operator.sub), (String, operator.eq), (Boolean, operator.or_)]
)
def test_mismatched_lengths(dtype: DType, op: Callable) -> None:
    with pytest.raises(ShapeError):
        op(Series("x", [None], dtype), Series("y", [], dtype))


@pytest.mark.parametrize(
    ("scalar", "conjunction", "disjunction"),
    [
        (True, [True, False, None], [True, True, True]),
        (False, [False, False, False], [True, False, None]),
        (None, [None, False, None], [True, None, None]),
    ],
)
def test_kleene_truth_table(
    scalar: bool | None, conjunction: list[Scalar], disjunction: list[Scalar]
) -> None:
    source = Series("x", [True, False, None])
    right = Series("y", [scalar] * 3, Boolean)
    for result in (source & right, source & scalar, scalar & source):
        assert_series(result, Boolean, conjunction)
    for result in (source | right, source | scalar, scalar | source):
        assert_series(result, Boolean, disjunction)
    assert_series(~source, Boolean, [False, True, None])
    assert_series(source, Boolean, [True, False, None])


@pytest.mark.parametrize("values", [[], [None] * 9])
def test_boolean_empty_and_all_null(values: list[Scalar]) -> None:
    source = Series("x", values, Boolean)
    assert_series(~source, Boolean, values)
    assert_series(source | True, Boolean, [True] * len(values))
    assert_series(False & source, Boolean, [False] * len(values))
    for invalid in (0, 1):
        with pytest.raises(TypeError):
            operator.and_(source, invalid)
    with pytest.raises(SchemaError):
        operator.or_(source, Series("y", values, Int64))


def test_cast_and_packed_null_masks() -> None:
    source = Series("x", [None, "5", None, "", "🙂", None, None, "", None, "9"])
    assert_series(source.cast(Categorical), Categorical, source.to_list())
    assert_series(source.slice(0, 2).cast(Int64) + 2, Int64, [None, 7])
    with pytest.raises(ComputeError):
        source.cast(Int64)
    with pytest.raises(ComputeError):
        source.cast(Boolean)
    with pytest.raises(SchemaError):
        source.cast("String")
    for mask, packed in ((source.is_null(), b"\x65\x01"), (source.is_not_null(), b"\x9a\x02")):
        assert mask._column._values.to_bytes() == packed
        assert mask._column._validity.to_bytes() == b"\xff\x03"
        assert mask._column._values._data is not source._column._validity._data
        assert mask._column._validity._data is not source._column._validity._data
        mask._column.check_invariants()


@pytest.mark.parametrize(
    ("dtype", "bad"), [(Int64, True), (Float64, 1), (Boolean, 0), (String, b"a"), (Categorical, 4)]
)
def test_invalid_fill_even_without_nulls(dtype: DType, bad: object) -> None:
    for values in ([], [None]):
        with pytest.raises(SchemaError):
            Series("x", values, dtype).fill_null(bad)


def test_fill_validates_unused_replacement() -> None:
    with pytest.raises(ComputeError):
        Series("x", [1]).fill_null(2**63)
    with pytest.raises(SchemaError):
        Series("x", ["valid"]).fill_null("\ud800")


def test_equals_contract() -> None:
    left = Series("x", [nan, None, -0.0])
    right = Series("x", [float("nan"), None, 0.0])
    assert left.equals(right) is True
    assert left.equals(right, null_equal=False) is False
    assert left.slice(0, 1).equals(right.slice(0, 1), null_equal=False) is True
    assert_series(left == right, Boolean, [False, None, True])
    assert_series(left.is_null(), Boolean, [False, True, False])
    assert_series(left.fill_null(4.5), Float64, [nan, 4.5, -0.0])
    for other in (
        Series("y", right.to_list()),
        Series("x", [], Float64),
        Series("x", [0, None, 0]),
        Series("x", [nan, 1.0, 0.0]),
        Series("x", [nan, None, 1.0]),
        Series("x", [0.0, None, 0.0]),
    ):
        assert left.equals(other) is False
        assert other.equals(left) is False
    with pytest.raises(TypeError):
        left.equals([nan, None, 0.0])
    with pytest.raises(TypeError):
        left.equals(right, null_equal=1)
    with pytest.raises(TypeError, match="^the truth value of a Series is ambiguous$"):
        bool(left == right)
    with pytest.raises(TypeError):
        hash(left)


@pytest.mark.parametrize(
    ("dtype", "scalars"),
    [
        (Int64, st.integers(INT64_MIN, INT64_MAX)),
        (Float64, st.floats()),
        (Boolean, st.booleans()),
        (String, st.text()),
        (Categorical, st.text()),
    ],
)
@given(data=st.data())
@settings(derandomize=True)
def test_nullable_transformations(
    dtype: DType, scalars: st.SearchStrategy, data: st.DataObject
) -> None:
    values = data.draw(st.lists(st.none() | scalars))
    replacement = data.draw(scalars)
    offset = data.draw(st.integers(0, len(values)))
    length = data.draw(st.integers(0, len(values) - offset))
    source = Series("x", iter(values), dtype)
    assert_series(source.slice(offset, length), dtype, values[offset : offset + length])
    assert_series(source.is_null(), Boolean, [v is None for v in values])
    assert_series(source.is_not_null(), Boolean, [v is not None for v in values])
    assert_series(
        source.fill_null(replacement), dtype, [replacement if v is None else v for v in values]
    )
    assert_series(
        source == replacement, Boolean, [None if v is None else v == replacement for v in values]
    )
    if dtype is Boolean:
        expected = [False if v is False or replacement is False else v for v in values]
        assert_series(source & replacement, Boolean, expected)
    assert_series(source, dtype, values)


@given(st.lists(st.tuples(st.none() | st.integers(-1000, 1000), st.none() | st.floats())))
@settings(derandomize=True)
def test_mixed_numeric_series(pairs: list[tuple[int | None, float | None]]) -> None:
    left = Series("x", (x for x, _ in pairs), Int64)
    right = Series("y", (y for _, y in pairs), Float64)
    difference = left - right
    assert_series(
        difference, Float64, [None if x is None or y is None else x - y for x, y in pairs]
    )
    expected = compare.ge(arithmetic.subtract(left._column, right._column), 0)
    assert_series(difference >= 0, Boolean, expected.to_list())
    assert_series(
        left < right, Boolean, [None if x is None or y is None else x < y for x, y in pairs]
    )
    assert_series(left, Int64, [x for x, _ in pairs])
    assert_series(right, Float64, [y for _, y in pairs], "y")
