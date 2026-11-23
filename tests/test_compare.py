import operator
from collections.abc import Callable
from itertools import product
from math import inf, nan

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
from miniframe.kernels.compare import eq, ge, gt, le, lt, ne

type Scalar = int | float | bool | str | None
type Operand = Column | Scalar
type Kernel = Callable[[Operand, Operand], Column]
type Predicate = Callable[[Scalar, Scalar], bool]

comparisons = (
    (eq, operator.eq),
    (ne, operator.ne),
    (lt, operator.lt),
    (le, operator.le),
    (gt, operator.gt),
    (ge, operator.ge),
)
compatible = [
    *product((Int64, Float64), repeat=2),
    *product((String, Categorical), repeat=2),
    (Boolean, Boolean),
]


def assert_result(column: Column, expected: list[bool | None]) -> None:
    column.check_invariants()
    assert column.dtype is Boolean
    assert column.null_count == expected.count(None)
    assert column.to_list() == expected


def oracle(op: Predicate, a: Scalar, b: Scalar) -> bool | None:
    return None if a is None or b is None else op(a, b)


@pytest.mark.parametrize(("left_type", "right_type"), compatible)
def test_logical_values_and_scalar_broadcast(left_type: DType, right_type: DType) -> None:
    samples = {
        Int64: [2, 1],
        Float64: [2.0, 1.0],
        Boolean: [True, False],
        String: ["🍎", "é"],
        Categorical: ["🍎", "é"],
    }
    x, y = samples[left_type]
    u, v = samples[right_type]
    a, b = [x, y, y, None, x, None], [v, u, v, u, None, None]
    left, right = Column(a, left_type), Column(b, right_type)
    for kernel, op in comparisons[:2] if left_type is Boolean else comparisons:
        assert_result(kernel(left, right), [oracle(op, x, y) for x, y in zip(a, b, strict=True)])
        assert_result(kernel(left, v), [oracle(op, value, v) for value in a])
        assert_result(kernel(u, left), [oracle(op, u, value) for value in a])
    assert left.to_list() == a
    assert right.to_list() == b


@pytest.mark.parametrize(("kernel", "op"), comparisons)
def test_numeric_precision_and_ieee(kernel: Kernel, op: Predicate) -> None:
    ints = [INT64_MAX, INT64_MIN, INT64_MIN + 1, 2**53 + 1, 0, 0, 5, -5, None]
    floats = [
        float(2**63),
        float(INT64_MIN),
        float(INT64_MIN),
        float(2**53),
        -0.0,
        nan,
        inf,
        -inf,
        nan,
    ]
    left, right = Column(ints, Int64), Column(floats, Float64)
    pairs = list(zip(ints, floats, strict=True))
    assert_result(kernel(left, right), [oracle(op, x, y) for x, y in pairs])
    assert_result(kernel(right, left), [oracle(op, y, x) for x, y in pairs])
    assert_result(kernel(right, right), [op(y, y) for y in floats])
    assert_result(kernel(0.0, right), [op(0.0, y) for y in floats])


@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33])
def test_packed_storage_ownership(size: int) -> None:
    values = [[True, False, None][i % 3] for i in range(size)]
    column = Column(values, Boolean)
    sources = (column._validity, column._values)
    before = [bitmap.to_bytes() for bitmap in sources]
    for other in (column, True, None):
        result = eq(column, other)
        expected = [oracle(operator.eq, v, v if other is column else other) for v in values]
        assert_result(result, expected)
        assert result._values._data is not result._validity._data
        for bitmap in (result._values, result._validity):
            assert all(bitmap._data is not source._data for source in sources)
    assert [bitmap.to_bytes() for bitmap in sources] == before


@pytest.mark.parametrize(("kernel", "op"), comparisons)
def test_null_scalar_and_validation(kernel: Kernel, op: Predicate) -> None:
    for dtype in DType:
        column = Column([None] * 3, dtype)
        assert_result(kernel(None, column), [None] * 3)
        assert_result(kernel(column, None), [None] * 3)
    for a, b in product(DType, repeat=2):
        if (a, b) not in compatible or (a is b is Boolean and op not in (operator.eq, operator.ne)):
            reason = (
                "Boolean supports only equality and inequality"
                if a is b is Boolean
                else f"cannot compare {a.value} with {b.value}"
            )
            with pytest.raises(SchemaError) as error:
                kernel(Column([], a), Column([], b))
            assert str(error.value) == reason
    empty = Column([], Int64)
    for bad in (True, "1", object()):
        with pytest.raises(SchemaError):
            kernel(bad, empty)
        with pytest.raises(SchemaError):
            kernel(empty, bad)
    for bad_int in (INT64_MAX + 1, INT64_MIN - 1):
        with pytest.raises(ComputeError, match="^Int64 overflow$"):
            kernel(bad_int, empty)
    with pytest.raises(ShapeError):
        kernel(Column([1], Int64), Column([1, 2], Int64))
    with pytest.raises(SchemaError, match="^unsupported comparison operand: object$"):
        kernel(empty, object())
    with pytest.raises(TypeError, match="^at least one operand must be a Column$"):
        kernel(1, 2)


@given(data=st.data())
def test_comparisons_against_scalar_reference(data: st.DataObject) -> None:
    a, b = data.draw(st.sampled_from(compatible))
    strategies = {
        Int64: st.integers(INT64_MIN, INT64_MAX),
        Float64: st.floats(),
        String: st.text(),
        Categorical: st.text(),
        Boolean: st.booleans(),
    }
    xs, ys = strategies[a] | st.none(), strategies[b] | st.none()
    rows = data.draw(st.lists(st.tuples(xs, ys), max_size=35))
    scalar = data.draw(ys)
    left, right = Column((x for x, _ in rows), a), Column((y for _, y in rows), b)
    for kernel, op in comparisons[:2] if a is Boolean else comparisons:
        assert_result(kernel(left, right), [oracle(op, x, y) for x, y in rows])
        assert_result(kernel(left, scalar), [oracle(op, x, scalar) for x, _ in rows])
        assert_result(kernel(scalar, left), [oracle(op, scalar, x) for x, _ in rows])
    equality = [oracle(operator.eq, x, y) for x, y in rows]
    assert_result(eq(right, left), equality)
    assert_result(ne(left, right), [None if v is None else not v for v in equality])


def test_unicode_order_against_arrow() -> None:
    a, b = ["", "é", "e\u0301", "🙂", None], ["🙂", "e\u0301", "é", "", None]
    left, right = Column(a, Categorical), Column(b, String)
    arrows = (pc.equal, pc.not_equal, pc.less, pc.less_equal, pc.greater, pc.greater_equal)
    for (kernel, _), arrow in zip(comparisons, arrows, strict=True):
        assert_result(kernel(left, right), arrow(pa.array(a), pa.array(b)).to_pylist())
