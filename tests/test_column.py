from array import array
from math import copysign, inf, isnan, nan
from struct import pack

import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from miniframe import Boolean, Categorical, ComputeError, DType, Float64, Int64, SchemaError, String
from miniframe.bitmap import Bitmap
from miniframe.column import Column

type Scalar = int | float | bool | str | None

EXAMPLES = [
    (Int64, 0, -1),
    (Float64, -0.0, inf),
    (Boolean, False, True),
    (String, "", "Київ"),
    (Categorical, "", "Київ"),
]


def assert_scalars(actual: list[Scalar], expected: list[Scalar]) -> None:
    for a, b in zip(actual, expected, strict=True):
        assert type(a) is type(b)
        if type(b) is float and isnan(b):
            assert isnan(a)
        else:
            assert a == b
            if type(b) is float and b == 0.0:
                assert copysign(1.0, a) == copysign(1.0, b)


def assert_column(column: Column, dtype: DType, expected: list[Scalar]) -> None:
    assert column.dtype is dtype
    assert len(column) == len(expected)
    assert column.null_count == expected.count(None)
    assert_scalars(column.to_list(), expected)
    assert_scalars([column.get(i) for i in range(len(expected))], expected)
    column.check_invariants()


@pytest.mark.parametrize(("dtype", "zero", "other"), EXAMPLES)
def test_empty_singleton_and_mixed_columns(dtype: DType, zero: Scalar, other: Scalar) -> None:
    for values in ([], [None], [zero], [None] * 9, [None, zero, None, other, None]):
        column = Column.from_list(iter(values), dtype)
        assert_column(column, dtype, values)
        assert_column(column.slice(len(values), 0), dtype, [])
        assert_column(column.take(iter(())), dtype, [])


def test_int64_boundaries_and_numeric_storage() -> None:
    minimum, maximum = -(2**63), 2**63 - 1
    values = [minimum, None, maximum, 0]
    column = Column.from_list(values, Int64)
    assert_column(column, Int64, values)
    assert isinstance(column._values, array)
    assert column._values.typecode == "q"
    assert column._values.tobytes() == pack("<4q", minimum, 0, maximum, 0)
    assert_column(column.slice(1, 2), Int64, [None, maximum])
    assert_column(column.take(iter([2, 0, 2, 1])), Int64, [maximum, minimum, maximum, None])


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**63])
def test_int64_overflow(value: int) -> None:
    with pytest.raises(ComputeError, match=r"^Int64 overflow at position 1$"):
        Column.from_list([None, value], Int64)


def test_float64_special_values_and_numeric_storage() -> None:
    values = [nan, None, inf, -inf, -0.0, 0.0]
    column = Column.from_list(values, Float64)
    assert_column(column, Float64, values)
    assert isinstance(column._values, array)
    assert column._values.typecode == "d"
    assert column._values.tobytes() == pack("<6d", nan, 0.0, inf, -inf, -0.0, 0.0)
    assert_column(column.slice(1, 4), Float64, [None, inf, -inf, -0.0])
    assert_column(column.take([4, 0, 1, 4]), Float64, [-0.0, nan, None, -0.0])


def test_boolean_value_and_validity_layout() -> None:
    values = [True, None, False, None, False, False, None, True, True, None]
    column = Column.from_list(values, Boolean)
    assert_column(column, Boolean, values)
    assert isinstance(column._values, Bitmap)
    assert column._values.to_bytes() == b"\x81\x01"
    assert column._validity.to_bytes() == b"\xb5\x01"
    assert_column(column.slice(2, 7), Boolean, values[2:9])
    assert_column(column.take([8, 1, 0, 2, 8]), Boolean, [True, None, True, False, True])


def test_string_byte_ranges_and_ownership() -> None:
    values = [None, "Київ", "", None, "🙂\0", "Київ"]
    column = Column.from_list(values, String)
    assert column._values.offsets == array("i", [0, 0, 8, 8, 8, 13, 21])
    assert column._values.data == "Київ🙂\0Київ".encode()
    assert column._validity.to_bytes() == b"\x36"

    sliced = column.slice(2, 3)
    gathered = column.take(iter([4, 1, 4, 3, 2]))
    assert sliced._values.offsets == array("i", [0, 0, 0, 5])
    assert sliced._values.data == "🙂\0".encode()
    assert gathered._values.offsets == array("i", [0, 5, 13, 18, 18, 18])
    assert gathered._values.data == "🙂\0Київ🙂\0".encode()
    column._values.offsets[4] = 0
    column._values.data[:] = b""
    assert_column(sliced, String, ["", None, "🙂\0"])
    assert_column(gathered, String, ["🙂\0", "Київ", "🙂\0", None, ""])


def test_categorical_dictionary_codes_and_ownership() -> None:
    values = ["🙂", "Київ", "🙂", None, "", "Київ"]
    column = Column.from_list(iter(values), Categorical)
    arrow = pa.array(values, type=pa.string()).dictionary_encode()
    assert column._values.dictionary == tuple(arrow.dictionary.to_pylist()) == ("🙂", "Київ", "")
    assert (
        column._values.codes.tolist()
        == arrow.indices.fill_null(0).to_pylist()
        == [0, 1, 0, 0, 2, 1]
    )

    sliced = column.slice(3, 3)
    gathered = column.take(iter([4, 3, 4, 1]))
    assert sliced._values.codes == array("i", [0, 2, 1])
    assert gathered._values.codes == array("i", [2, 0, 2, 1])
    column._values.codes[4] = 0
    assert_column(sliced, Categorical, [None, "", "Київ"])
    assert_column(gathered, Categorical, ["", None, "", "Київ"])


@pytest.mark.parametrize("length", [0, 5])
def test_all_null_categorical_storage(length: int) -> None:
    column = Column.from_list([None] * length, Categorical)
    assert column._values.dictionary == ()
    assert column._values.codes == array("i", [0] * length)
    assert_column(column.slice(0, length), Categorical, [None] * length)


def test_string_rejects_unencodable_text() -> None:
    with pytest.raises(SchemaError, match="^String cannot store invalid UTF-8 text at position 1$"):
        Column.from_list(["", "\ud800"], String)


def test_string_offset_limit_on_construction_and_take(monkeypatch: pytest.MonkeyPatch) -> None:
    import miniframe.column as storage

    assert storage._INT32_MAX == 2**31 - 1
    monkeypatch.setattr(storage, "_INT32_MAX", 4)
    column = Column.from_list(["ї", "ї"], String)
    assert_column(column.take([1, 1]), String, ["ї", "ї"])
    message = "^String byte offset exceeds signed 32-bit range$"
    with pytest.raises(ComputeError, match=message):
        Column.from_list(["🙂", "!"], String)
    with pytest.raises(ComputeError, match=message):
        column.take([0, 1, 0])
    assert_column(column, String, ["ї", "ї"])


def test_categorical_code_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("miniframe.column._INT32_MAX", 1)
    assert_column(Column.from_list(["a", "b", "b"], Categorical), Categorical, ["a", "b", "b"])
    with pytest.raises(ComputeError, match="^Categorical code exceeds signed 32-bit range$"):
        Column.from_list(["a", "b", "c"], Categorical)


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class StrSubclass(str):
    pass


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (Int64, True),
        (Int64, 1.0),
        (Int64, IntSubclass(1)),
        (Float64, 1),
        (Float64, False),
        (Float64, FloatSubclass(1.0)),
        (Boolean, 1),
        (Boolean, 0.0),
        (Boolean, "false"),
        (String, b"text"),
        (String, StrSubclass("text")),
        (Categorical, 0),
        (Categorical, StrSubclass("text")),
    ],
)
def test_scalar_types_are_strict(dtype: DType, value: object) -> None:
    message = f"^{dtype.value} cannot store {type(value).__name__} at position 1$"
    with pytest.raises(SchemaError, match=message):
        Column.from_list(iter([None, value]), dtype)


@pytest.mark.parametrize("dtype", ["Int64", None])
def test_unsupported_dtype(dtype: DType) -> None:
    with pytest.raises(SchemaError) as error:
        Column.from_list([], dtype)
    assert str(error.value) == f"Unsupported column dtype: {dtype!r}"


@pytest.mark.parametrize(
    ("index", "error"),
    [
        (-1, IndexError),
        (3, IndexError),
        (10**100, IndexError),
        (True, TypeError),
        (1.0, TypeError),
        (IntSubclass(1), TypeError),
        (None, TypeError),
    ],
)
@pytest.mark.parametrize(("dtype", "zero", "other"), EXAMPLES)
def test_invalid_get_and_take_indices(
    dtype: DType, zero: Scalar, other: Scalar, index: int, error: type[Exception]
) -> None:
    values = [zero, None, other]
    column = Column.from_list(values, dtype)
    with pytest.raises(error):
        column.get(index)
    with pytest.raises(error):
        column.take(iter([0, index]))
    assert_column(column, dtype, values)


def test_nonempty_selection_from_empty_column() -> None:
    column = Column.from_list([], Boolean)
    with pytest.raises(IndexError):
        column.get(0)
    with pytest.raises(IndexError):
        column.take([0])


@pytest.mark.parametrize(
    ("offset", "length", "error"),
    [
        (-1, 1, IndexError),
        (4, 0, IndexError),
        (3, 1, IndexError),
        (2, 2, IndexError),
        (0, -1, ValueError),
        (0.0, 1, TypeError),
        (0, 1.0, TypeError),
        (True, 1, TypeError),
        (0, False, TypeError),
    ],
)
def test_invalid_slice(offset: int, length: int, error: type[Exception]) -> None:
    column = Column.from_list([False, None, True], Boolean)
    with pytest.raises(error):
        column.slice(offset, length)
    assert_column(column, Boolean, [False, None, True])


@pytest.mark.parametrize(("dtype", "zero", "other"), EXAMPLES)
def test_column_owns_its_buffers(dtype: DType, zero: Scalar, other: Scalar) -> None:
    source = [other, None, zero]
    column = Column.from_list(source, dtype)
    sliced = column.slice(0, len(source))
    gathered = column.take(range(len(source)))
    sources_and_results = (column, sliced, gathered)

    source[:] = [None]
    assert len({id(result) for result in sources_and_results}) == 3
    assert len({id(result._values) for result in sources_and_results}) == 3
    assert len({id(result._validity) for result in sources_and_results}) == 3
    for result in sources_and_results:
        snapshot = result.to_list()
        snapshot[:] = [None]
        with pytest.raises(AttributeError):
            result.dtype = String
        with pytest.raises(AttributeError):
            result.null_count = 0
        assert_column(result, dtype, [other, None, zero])


@pytest.mark.parametrize(
    ("dtype", "field", "storage", "message"),
    [
        (Int64, "_dtype", "Int64", "unsupported column dtype"),
        (Int64, "_dtype", String, "expected string storage"),
        (Int64, "_dtype", Categorical, "expected categorical storage"),
        (Int64, "_values", Bitmap(1), "expected an array"),
        (Int64, "_values", array("d", [0.0]), "incorrect array typecode"),
        (Int64, "_values", array("q"), "value length differs from column length"),
        (Int64, "_values", array("q", [1]), "nonzero value at a null position"),
        (Float64, "_values", array("d", [-0.0]), "negative zero at a null position"),
        (Boolean, "_values", array("q", [0]), "expected a Bitmap"),
        (Boolean, "_values", Bitmap(0), "bitmap length differs from column length"),
        (Boolean, "_values", Bitmap(1, True), "nonzero value at a null position"),
        (Int64, "_validity", Bitmap(2), "bitmap length differs from column length"),
        (Int64, "_null_count", 0, "incorrect null count"),
    ],
)
def test_invariants_detect_corrupt_metadata_and_storage(
    dtype: DType, field: str, storage: object, message: str
) -> None:
    column = Column.from_list([None], dtype)
    setattr(column, field, storage)
    with pytest.raises(AssertionError, match=f"^{message}$"):
        column.check_invariants()


@pytest.mark.parametrize("field", ["_values", "_validity"])
@pytest.mark.parametrize(
    ("packed", "message"),
    [(b"", "incorrect bitmap byte count"), (b"\x80", "nonzero bitmap padding")],
)
def test_invariants_detect_corrupt_bitmap_bytes(field: str, packed: bytes, message: str) -> None:
    column = Column.from_list([None], Boolean)
    getattr(column, field)._data[:] = packed
    with pytest.raises(AssertionError, match=f"^{message}$"):
        column.check_invariants()


@pytest.mark.parametrize(
    ("dtype", "field", "replacement", "message"),
    [
        (String, "data", b"ab", "expected a UTF-8 bytearray"),
        (String, "offsets", [0, 1, 1, 2], "expected an array"),
        (String, "offsets", array("q", [0, 1, 1, 2]), "incorrect array typecode"),
        (String, "offsets", array("i", [0, 1, 2]), "incorrect string offset count"),
        (String, "offsets", array("i", [1, 1, 1, 2]), "first string offset must be zero"),
        (
            String,
            "offsets",
            array("i", [0, 1, 1, 3]),
            "final string offset differs from byte count",
        ),
        (
            String,
            "offsets",
            array("i", [0, -1, 1, 2]),
            "string offset out of order or out of bounds",
        ),
        (
            String,
            "offsets",
            array("i", [0, 2, 1, 2]),
            "string offset out of order or out of bounds",
        ),
        (String, "offsets", array("i", [0, 0, 1, 2]), "nonempty string at a null position"),
        (String, "data", bytearray("ї", "utf-8"), "invalid UTF-8 at position 0"),
        (String, "data", bytearray(b"a\xff"), "invalid UTF-8 at position 2"),
        (Categorical, "dictionary", ["a", "b"], "expected a dictionary tuple"),
        (Categorical, "dictionary", ("a", 1), "expected string dictionary entries"),
        (Categorical, "dictionary", ("a", "a"), "duplicate dictionary entry"),
        (Categorical, "codes", [0, 0, 1], "expected an array"),
        (Categorical, "codes", array("I", [0, 0, 1]), "incorrect array typecode"),
        (Categorical, "codes", array("i", [0, 1]), "incorrect categorical code count"),
        (Categorical, "codes", array("i", [-1, 0, 1]), "categorical code out of range"),
        (Categorical, "codes", array("i", [0, 0, 2]), "categorical code out of range"),
        (Categorical, "codes", array("i", [0, 1, 1]), "nonzero code at a null position"),
    ],
)
def test_text_storage_invariants(
    dtype: DType, field: str, replacement: object, message: str
) -> None:
    column = Column.from_list(["a", None, "b"], dtype)
    setattr(column._values, field, replacement)
    with pytest.raises(AssertionError, match=f"^{message}$"):
        column.check_invariants()


SCALARS = {
    Int64: st.integers(-(2**63), 2**63 - 1),
    Float64: st.floats(width=64),
    Boolean: st.booleans(),
    String: st.text(),
    Categorical: st.sampled_from(["", "Львів", "🙂"]) | st.text(),
}

ARROW_TYPES = {
    Int64: pa.int64(),
    Float64: pa.float64(),
    Boolean: pa.bool_(),
    String: pa.string(),
    Categorical: pa.string(),
}


def assert_arrow_layout(column: Column, values: list[Scalar]) -> None:
    arrow = pa.array(values, type=ARROW_TYPES[column.dtype], from_pandas=False)
    assert_scalars(arrow.to_pylist(), column.to_list())
    packed = arrow.buffers()[0]
    if packed is None:
        packed = arrow.is_valid().buffers()[1]
    size = (len(values) + 7) // 8
    expected = bytearray(packed.slice(0, size).to_pybytes())
    if len(values) % 8:
        expected[-1] &= (1 << (len(values) % 8)) - 1
    assert column._validity.to_bytes() == expected
    if column.dtype is String:
        _, offsets, data = arrow.buffers()
        assert column._values.offsets.tobytes() == offsets.to_pybytes()
        assert column._values.data == data.to_pybytes()


@pytest.mark.parametrize("dtype", tuple(DType))
@settings(derandomize=True)
@given(data=st.data())
def test_roundtrip_and_transformations_against_list_and_arrow(
    dtype: DType, data: st.DataObject
) -> None:
    values = data.draw(st.lists(st.none() | SCALARS[dtype], max_size=64))
    column = Column.from_list((value for value in values), dtype)
    assert_column(column, dtype, values)
    assert_arrow_layout(column, values)

    offset = data.draw(st.integers(0, len(values)))
    length = data.draw(st.integers(0, len(values) - offset))
    sliced = column.slice(offset, length)
    expected_slice = values[offset : offset + length]
    assert_column(sliced, dtype, expected_slice)
    assert_arrow_layout(sliced, expected_slice)

    indices = (
        data.draw(st.lists(st.sampled_from(range(len(values))), max_size=64)) if values else []
    )
    gathered = column.take(iter(indices))
    expected_take = [values[i] for i in indices]
    assert_column(gathered, dtype, expected_take)
    assert_arrow_layout(gathered, expected_take)

    permutation = data.draw(st.permutations(tuple(range(len(values)))))
    inverse = sorted(range(len(values)), key=permutation.__getitem__)
    permuted = column.take(permutation)
    assert_column(permuted, dtype, [values[i] for i in permutation])
    assert_column(permuted.take(inverse), dtype, values)
    assert_column(column, dtype, values)
    assert_arrow_layout(column, values)
