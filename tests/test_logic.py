from collections.abc import Callable

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from hypothesis import given
from hypothesis import strategies as st

from miniframe.column import Column
from miniframe.dtype import Boolean, Categorical, DType, Float64, Int64, String
from miniframe.errors import SchemaError, ShapeError
from miniframe.kernels.logic import and_kleene, not_kleene, or_kleene

type NullableBool = bool | None
type BinaryKernel = Callable[[Column, Column], Column]

nullable_bool = st.none() | st.booleans()


def scalar_and(left: NullableBool, right: NullableBool) -> NullableBool:
    if left is False or right is False:
        return False
    return None if left is None or right is None else True


def scalar_or(left: NullableBool, right: NullableBool) -> NullableBool:
    if left is True or right is True:
        return True
    return None if left is None or right is None else False


def assert_result(result: Column, expected: list[NullableBool]) -> None:
    result.check_invariants()
    assert result.dtype is Boolean
    assert len(result) == len(expected)
    assert result.null_count == expected.count(None)
    assert result.to_list() == expected


@pytest.mark.parametrize(
    ("a", "b", "conjunction", "disjunction"),
    [
        (True, True, True, True),
        (True, False, False, True),
        (True, None, None, True),
        (False, True, False, True),
        (False, False, False, False),
        (False, None, False, None),
        (None, True, None, True),
        (None, False, False, None),
        (None, None, None, None),
    ],
)
def test_binary_truth_tables(
    a: NullableBool, b: NullableBool, conjunction: NullableBool, disjunction: NullableBool
) -> None:
    left, right = Column([a], Boolean), Column([b], Boolean)
    assert_result(and_kleene(left, right), [conjunction])
    assert_result(or_kleene(left, right), [disjunction])


@pytest.mark.parametrize(("value", "inverse"), [(True, False), (False, True), (None, None)])
def test_uniform_columns(value: NullableBool, inverse: NullableBool) -> None:
    column = Column([value] * 17, Boolean)
    assert_result(not_kleene(column), [inverse] * 17)
    assert_result(and_kleene(column, column), [value] * 17)
    assert_result(or_kleene(column, column), [value] * 17)


@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1025])
def test_packed_boundaries_and_buffer_ownership(length: int) -> None:
    a = ([True, None, False] * ((length + 2) // 3))[:length]
    b = ([None, False, True, True] * ((length + 3) // 4))[:length]
    left, right = Column(a, Boolean), Column(b, Boolean)
    sources = (left._values, left._validity, right._values, right._validity)
    snapshots = [buffer.to_bytes() for buffer in sources]
    results = (
        (and_kleene(left, right), [scalar_and(x, y) for x, y in zip(a, b, strict=True)]),
        (or_kleene(left, right), [scalar_or(x, y) for x, y in zip(a, b, strict=True)]),
        (not_kleene(left), [None if x is None else not x for x in a]),
    )
    for result, expected in results:
        assert_result(result, expected)
        assert result._values._data is not result._validity._data
        for buffer in (result._values, result._validity):
            assert all(buffer._data is not source._data for source in sources)
    assert [buffer.to_bytes() for buffer in sources] == snapshots
    assert_result(left, a)
    assert_result(right, b)


@pytest.mark.parametrize("dtype", [Int64, Float64, String, Categorical])
def test_non_boolean_column(dtype: DType) -> None:
    bad, good = Column([None], dtype), Column([None], Boolean)
    message = f"expected Boolean dtype, got {dtype.value}"
    with pytest.raises(SchemaError, match=f"^{message}$"):
        not_kleene(bad)
    for kernel in (and_kleene, or_kleene):
        for left, right in ((bad, good), (good, bad)):
            with pytest.raises(SchemaError, match=f"^{message}$"):
                kernel(left, right)


@pytest.mark.parametrize("operand", [True, None, [False]])
def test_non_column_operand(operand: object) -> None:
    good = Column([False], Boolean)
    message = f"^expected a Column, got {type(operand).__name__}$"
    with pytest.raises(TypeError, match=message):
        not_kleene(operand)
    for kernel in (and_kleene, or_kleene):
        for left, right in ((operand, good), (good, operand)):
            with pytest.raises(TypeError, match=message):
                kernel(left, right)


@pytest.mark.parametrize("kernel", [and_kleene, or_kleene])
def test_different_logical_lengths(kernel: BinaryKernel) -> None:
    with pytest.raises(ShapeError):
        kernel(Column([False] * 7, Boolean), Column([False] * 8, Boolean))


@given(st.lists(st.tuples(nullable_bool, nullable_bool)))
def test_agreement_with_scalar_and_arrow(rows: list[tuple[NullableBool, NullableBool]]) -> None:
    a, b = [x for x, _ in rows], [y for _, y in rows]
    left, right = Column(a, Boolean), Column(b, Boolean)
    arrow_left, arrow_right = pa.array(a, type=pa.bool_()), pa.array(b, type=pa.bool_())
    for kernel, scalar, arrow in (
        (and_kleene, scalar_and, pc.and_kleene),
        (or_kleene, scalar_or, pc.or_kleene),
    ):
        result = kernel(left, right)
        assert_result(result, [scalar(x, y) for x, y in rows])
        assert result.to_list() == arrow(arrow_left, arrow_right).to_pylist()
    result = not_kleene(left)
    assert_result(result, [None if x is None else not x for x in a])
    assert result.to_list() == pc.invert(arrow_left).to_pylist()


@given(st.lists(st.tuples(nullable_bool, nullable_bool, nullable_bool)))
def test_kleene_laws(rows: list[tuple[NullableBool, NullableBool, NullableBool]]) -> None:
    a, b, c = (Column((row[i] for row in rows), Boolean) for i in range(3))

    def checked(result: Column) -> Column:
        result.check_invariants()
        assert result.dtype is Boolean
        assert len(result) == len(rows)
        return result

    neg_a, neg_b = checked(not_kleene(a)), checked(not_kleene(b))
    assert checked(not_kleene(neg_a)).to_list() == a.to_list()
    for op, dual in ((and_kleene, or_kleene), (or_kleene, and_kleene)):
        ab, bc = checked(op(a, b)), checked(op(b, c))
        assert checked(op(a, a)).to_list() == a.to_list()
        assert ab.to_list() == checked(op(b, a)).to_list()
        assert checked(op(ab, c)).to_list() == checked(op(a, bc)).to_list()
        assert checked(not_kleene(ab)).to_list() == checked(dual(neg_a, neg_b)).to_list()
