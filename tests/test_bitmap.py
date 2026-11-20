import operator
from collections.abc import Callable

import pyarrow as pa
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from miniframe import ShapeError
from miniframe.bitmap import Bitmap


def assert_bitmap(bitmap: Bitmap, expected: list[bool]) -> None:
    assert len(bitmap) == len(expected)
    assert all(bitmap.get(i) is value for i, value in enumerate(expected))
    assert bitmap.popcount() == sum(expected)
    packed = bitmap.to_bytes()
    assert len(packed) == (len(expected) + 7) // 8
    if len(expected) % 8:
        assert packed[-1] >> (len(expected) % 8) == 0


@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 15, 16, 17])
@pytest.mark.parametrize("fill", [False, True])
def test_uniform_bitmaps(length: int, fill: bool) -> None:
    bitmap = Bitmap(length, fill)
    assert_bitmap(bitmap, [fill] * length)
    assert_bitmap(~bitmap, [not fill] * length)
    assert_bitmap(bitmap & ~bitmap, [False] * length)
    assert_bitmap(bitmap | ~bitmap, [True] * length)
    assert_bitmap(bitmap.slice(length, 0), [])
    if length:
        for i in (0, length - 1):
            bitmap.set(i, not fill)
            assert bitmap.get(i) is not fill
            bitmap.set(i, fill)
        assert_bitmap(bitmap, [fill] * length)


def test_lsb_first_layout_across_byte_boundaries() -> None:
    bitmap = Bitmap(17)
    for i in (0, 2, 7, 8, 15, 16):
        bitmap.set(i, True)
    assert bitmap.to_bytes() == b"\x85\x81\x01"
    bitmap.set(8, False)
    assert bitmap.to_bytes() == b"\x85\x80\x01"
    assert bitmap.slice(2, 9).to_bytes() == b"\x21\x00"


def test_snapshot_is_read_only_and_independent() -> None:
    bitmap = Bitmap(1, True)
    snapshot = bitmap.to_bytes()
    assert isinstance(snapshot, bytes)
    with pytest.raises(TypeError):
        memoryview(snapshot)[0] = 0
    bitmap.set(0, False)
    assert snapshot == b"\x01"


@pytest.mark.parametrize("bad", [0, 1, None, "", object()])
def test_only_booleans_accepted(bad: object) -> None:
    with pytest.raises(TypeError):
        Bitmap(0, bad)
    with pytest.raises(TypeError):
        Bitmap.from_bools([True, bad])
    bitmap = Bitmap(1, True)
    with pytest.raises(TypeError):
        bitmap.set(0, bad)
    assert_bitmap(bitmap, [True])


@pytest.mark.parametrize(("length", "error"), [(-1, ValueError), (1.5, TypeError)])
def test_invalid_length(length: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        Bitmap(length)
    with pytest.raises(error):
        Bitmap.from_bytes(b"", length)


@pytest.mark.parametrize(
    ("size", "i", "error"),
    [(0, 0, IndexError), (9, -1, IndexError), (9, 9, IndexError), (9, 1.0, TypeError)],
)
def test_invalid_index(size: int, i: object, error: type[Exception]) -> None:
    bitmap = Bitmap(size)
    with pytest.raises(error):
        bitmap.get(i)
    with pytest.raises(error):
        bitmap.set(i, True)
    assert_bitmap(bitmap, [False] * size)


@pytest.mark.parametrize(
    ("offset", "length", "error"),
    [
        (-1, 1, IndexError),
        (10, 0, IndexError),
        (9, 1, IndexError),
        (1, 9, IndexError),
        (0, -1, ValueError),
        (0.0, 1, TypeError),
        (0, 1.0, TypeError),
    ],
)
def test_invalid_slice(offset: object, length: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        Bitmap(9).slice(offset, length)


@pytest.mark.parametrize("operation", [operator.and_, operator.or_])
def test_invalid_operands(operation: Callable) -> None:
    with pytest.raises(ShapeError):
        operation(Bitmap(7), Bitmap(8))
    with pytest.raises(TypeError):
        operation(Bitmap(1), 1)


@pytest.mark.parametrize(("packed", "length"), [(b"", 1), (b"\x00", 0), (b"\x12\x34", 8)])
def test_invalid_packed_size(packed: bytes, length: int) -> None:
    with pytest.raises(ShapeError):
        Bitmap.from_bytes(packed, length)


def test_invalid_packed_type() -> None:
    with pytest.raises(TypeError):
        Bitmap.from_bytes(1, 1)


@given(st.lists(st.booleans(), min_size=1, max_size=128), st.booleans(), st.data())
def test_set_and_get(values: list[bool], value: bool, data: st.DataObject) -> None:
    bitmap = Bitmap.from_bools(iter(values))
    assert_bitmap(bitmap, values)
    i = data.draw(st.integers(0, len(values) - 1))
    bitmap.set(i, value)
    values[i] = value
    assert_bitmap(bitmap, values)


@given(st.binary(min_size=1, max_size=128), st.integers(0, 7))
def test_packed_input_ownership(raw: bytes, padding: int) -> None:
    source = bytearray(raw)
    length = 8 * len(raw) - padding
    expected = [c == "1" for byte in raw for c in f"{byte:08b}"[::-1]][:length]
    bitmap = Bitmap.from_bytes(source, length)
    assert source == raw
    source[0] ^= 1
    assert_bitmap(bitmap, expected)
    source_snapshot = bytes(source)
    bitmap.set(0, not expected[0])
    assert source == source_snapshot


@given(st.lists(st.booleans(), max_size=128), st.data())
def test_slice(values: list[bool], data: st.DataObject) -> None:
    bitmap = Bitmap.from_bools(values)
    offset = data.draw(st.integers(0, len(values)))
    length = data.draw(st.integers(0, len(values) - offset))
    result = bitmap.slice(offset, length)
    assert_bitmap(result, values[offset : offset + length])
    if length:
        result.set(0, not values[offset])
    assert_bitmap(bitmap, values)


@given(st.lists(st.tuples(st.booleans(), st.booleans()), max_size=128))
@example([(i % 3 == 0, i % 5 == 0) for i in range(4099)])
def test_bulk_operations(pairs: list[tuple[bool, bool]]) -> None:
    left = [a for a, _ in pairs]
    right = [b for _, b in pairs]
    a = Bitmap.from_bools(left)
    b = Bitmap.from_bools(right)
    intersection = a & b
    union = a | b
    complement = ~a
    assert_bitmap(intersection, [x and y for x, y in pairs])
    assert_bitmap(union, [x or y for x, y in pairs])
    assert_bitmap(complement, [not x for x in left])
    assert_bitmap(~complement, left)
    assert (~intersection).to_bytes() == (~a | ~b).to_bytes()
    assert (~union).to_bytes() == (~a & ~b).to_bytes()
    for result in (intersection, union, complement):
        if pairs:
            result.set(0, not result.get(0))
        assert_bitmap(a, left)
        assert_bitmap(b, right)


@pytest.fixture(scope="module")
def arrow_int8_type() -> pa.DataType:
    return pa.array([0], type=pa.int8()).type


@given(st.lists(st.booleans(), max_size=128))
def test_arrow_validity_layout(arrow_int8_type: pa.DataType, values: list[bool]) -> None:
    values = [False, *values]
    array = pa.array([1 if present else None for present in values], type=arrow_int8_type)
    size = (len(values) + 7) // 8
    expected = bytearray(array.buffers()[0].slice(0, size).to_pybytes())
    if len(values) % 8:
        expected[-1] &= (1 << (len(values) % 8)) - 1
    assert Bitmap.from_bools(values).to_bytes() == expected
