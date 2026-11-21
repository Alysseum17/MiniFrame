from array import array
from collections.abc import Iterable
from math import copysign
from sys import byteorder
from typing import Self

from .bitmap import Bitmap
from .dtype import Boolean, DType, Float64, Int64
from .errors import ComputeError, SchemaError

assert byteorder == "little", "MiniFrame requires little-endian byte order"
assert array("i").itemsize == 4, "MiniFrame requires 32-bit array('i') elements"
assert array("q").itemsize == 8, "MiniFrame requires 64-bit array('q') elements"
assert array("d").itemsize == 8, "MiniFrame requires 64-bit array('d') elements"

type _Scalar = int | float | bool | None
type _Storage = array[int] | array[float] | Bitmap


def _check_bitmap(bitmap: object, length: int) -> None:
    assert isinstance(bitmap, Bitmap), "expected a Bitmap"
    assert len(bitmap) == length, "bitmap length differs from column length"
    packed = bitmap.to_bytes()
    assert len(packed) == (length + 7) // 8, "incorrect bitmap byte count"
    used = length % 8
    assert used == 0 or packed[-1] >> used == 0, "nonzero bitmap padding"


class Column:
    """Immutable internal storage for nullable numeric and Boolean values."""

    __slots__ = ("_dtype", "_values", "_validity", "_length", "_null_count")

    def __init__(self, values: Iterable[object], dtype: DType) -> None:
        if dtype is not Int64 and dtype is not Float64 and dtype is not Boolean:
            raise SchemaError(f"Unsupported column dtype: {dtype!r}")

        scalars = []
        for position, value in enumerate(values):
            if not dtype.accepts(value):
                if dtype is Int64 and type(value) is int:
                    raise ComputeError(f"Int64 overflow at position {position}")
                raise SchemaError(
                    f"{dtype.value} cannot store {type(value).__name__} at position {position}"
                )
            scalars.append(value)

        validity = Bitmap.from_bools(value is not None for value in scalars)
        storage: _Storage
        if dtype is Boolean:
            storage = Bitmap.from_bools(value is True for value in scalars)
        else:
            typecode = dtype.array_typecode
            assert typecode is not None
            zero = 0 if dtype is Int64 else 0.0
            storage = array(typecode, (zero if value is None else value for value in scalars))
        self._init_storage(dtype, storage, validity)

    def _init_storage(self, dtype: DType, values: _Storage, validity: Bitmap) -> None:
        self._dtype = dtype
        self._values = values
        self._validity = validity
        self._length = len(values)
        self._null_count = self._length - validity.popcount()

    @classmethod
    def from_list(cls, values: Iterable[object], dtype: DType) -> Self:
        """Consume an iterable of nullable scalars using the explicit dtype."""
        return cls(values, dtype)

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def null_count(self) -> int:
        return self._null_count

    def __len__(self) -> int:
        return self._length

    def _value_at(self, index: int) -> int | float | bool:
        if isinstance(self._values, Bitmap):
            return self._values.get(index)
        return self._values[index]

    def get(self, index: int) -> _Scalar:
        """Read a zero-based position. Indices must be built-in integers."""
        if type(index) is not int:
            raise TypeError("column index must be an int")
        if self._validity.get(index):
            return self._value_at(index)
        return None

    def to_list(self) -> list[_Scalar]:
        return [self.get(index) for index in range(self._length)]

    def slice(self, offset: int, length: int) -> Self:
        """Copy a range wholly inside the column. An empty range may start at the end."""
        if type(offset) is not int or type(length) is not int:
            raise TypeError("slice offset and length must be ints")
        validity = self._validity.slice(offset, length)
        if isinstance(self._values, Bitmap):
            values = self._values.slice(offset, length)
        else:
            values = self._values[offset : offset + length]
        result = self.__class__.__new__(self.__class__)
        result._init_storage(self._dtype, values, validity)
        return result

    def take(self, indices: Iterable[int]) -> Self:
        """Copy the selected positions in order, including repeated positions."""
        return self.from_list((self.get(index) for index in indices), self._dtype)

    def check_invariants(self) -> None:
        """Assert that storage matches the dtype and uses canonical null placeholders."""
        values = self._values
        if self._dtype is Boolean:
            _check_bitmap(values, self._length)
        else:
            assert self._dtype is Int64 or self._dtype is Float64, "unsupported column dtype"
            assert isinstance(values, array), "expected an array"
            assert values.typecode == self._dtype.array_typecode, "incorrect array typecode"
            assert len(values) == self._length, "value length differs from column length"

        _check_bitmap(self._validity, self._length)
        assert self._null_count == self._length - self._validity.popcount(), "incorrect null count"

        for index in range(self._length):
            if not self._validity.get(index):
                value = self._value_at(index)
                assert value == 0, "nonzero value at a null position"
                if self._dtype is Float64:
                    assert copysign(1.0, value) == 1.0, "negative zero at a null position"
