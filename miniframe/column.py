from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from math import copysign
from sys import byteorder
from typing import Self, cast

from .bitmap import Bitmap
from .dtype import Boolean, Categorical, DType, Float64, Int64, String
from .errors import ComputeError, SchemaError

assert byteorder == "little", "MiniFrame requires little-endian byte order"
assert array("i").itemsize == 4, "MiniFrame requires 32-bit array('i') elements"
assert array("q").itemsize == 8, "MiniFrame requires 64-bit array('q') elements"
assert array("d").itemsize == 8, "MiniFrame requires 64-bit array('d') elements"

_INT32_MAX = 2**31 - 1

type _Scalar = int | float | bool | str | None


@dataclass(slots=True)
class _StringStorage:
    offsets: array[int]
    data: bytearray

    @classmethod
    def from_chunks(cls, chunks: Iterable[bytes | memoryview]) -> Self:
        offsets = array("i", [0])
        data = bytearray()
        for chunk in chunks:
            end = len(data) + len(chunk)
            if end > _INT32_MAX:
                raise ComputeError("String byte offset exceeds signed 32-bit range")
            offsets.append(end)
            data.extend(chunk)
        return cls(offsets, data)

    def slice(self, offset: int, length: int) -> Self:
        start = self.offsets[offset]
        stop = offset + length
        offsets = array("i", (self.offsets[i] - start for i in range(offset, stop + 1)))
        return type(self)(offsets, self.data[start : self.offsets[stop]])

    def check_invariants(self, validity: Bitmap) -> None:
        assert isinstance(self.data, bytearray), "expected a UTF-8 bytearray"
        _check_array(self.offsets, "i")
        assert len(self.offsets) == len(validity) + 1, "incorrect string offset count"
        assert self.offsets[0] == 0, "first string offset must be zero"
        assert self.offsets[-1] == len(self.data), "final string offset differs from byte count"
        for i, (start, end) in enumerate(pairwise(self.offsets)):
            assert start <= end <= len(self.data), "string offset out of order or out of bounds"
            assert validity.get(i) or start == end, "nonempty string at a null position"
            try:
                self.data[start:end].decode("utf-8")
            except UnicodeDecodeError as error:
                raise AssertionError(f"invalid UTF-8 at position {i}") from error


def _utf8_chunks(values: Iterable[str | None]) -> Iterator[bytes]:
    for i, value in enumerate(values):
        try:
            yield b"" if value is None else value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise SchemaError(f"String cannot store invalid UTF-8 text at position {i}") from error


@dataclass(slots=True)
class _CategoricalStorage:
    codes: array[int]
    dictionary: tuple[str, ...]

    @classmethod
    def from_strings(cls, values: Iterable[str | None]) -> Self:
        lookup: dict[str, int] = {}
        codes = array("i")
        for value in values:
            code = 0 if value is None else lookup.setdefault(value, len(lookup))
            if code > _INT32_MAX:
                raise ComputeError("Categorical code exceeds signed 32-bit range")
            codes.append(code)
        return cls(codes, tuple(lookup))

    def check_invariants(self, validity: Bitmap) -> None:
        _check_array(self.codes, "i")
        assert len(self.codes) == len(validity), "incorrect categorical code count"
        assert type(self.dictionary) is tuple, "expected a dictionary tuple"
        assert all(type(s) is str for s in self.dictionary), "expected string dictionary entries"
        assert len(set(self.dictionary)) == len(self.dictionary), "duplicate dictionary entry"
        for i, code in enumerate(self.codes):
            if validity.get(i):
                assert 0 <= code < len(self.dictionary), "categorical code out of range"
            else:
                assert code == 0, "nonzero code at a null position"


type _Storage = array[int] | array[float] | Bitmap | _StringStorage | _CategoricalStorage


def _check_array(values: object, typecode: str) -> None:
    assert isinstance(values, array), "expected an array"
    assert values.typecode == typecode, "incorrect array typecode"


def _check_bitmap(bitmap: object, length: int) -> None:
    assert isinstance(bitmap, Bitmap), "expected a Bitmap"
    assert len(bitmap) == length, "bitmap length differs from column length"
    packed = bitmap.to_bytes()
    assert len(packed) == (length + 7) // 8, "incorrect bitmap byte count"
    used = length % 8
    assert used == 0 or packed[-1] >> used == 0, "nonzero bitmap padding"


class Column:
    """Immutable internal storage for nullable column values."""

    __slots__ = ("_dtype", "_values", "_validity", "_length", "_null_count")

    def __init__(self, values: Iterable[object], dtype: DType) -> None:
        if not isinstance(dtype, DType):
            raise SchemaError(f"Unsupported column dtype: {dtype!r}")

        valid = bytearray()

        def checked_values() -> Iterator[_Scalar]:
            for position, value in enumerate(values):
                if not dtype.accepts(value):
                    if dtype is Int64 and type(value) is int:
                        raise ComputeError(f"Int64 overflow at position {position}")
                    raise SchemaError(
                        f"{dtype.value} cannot store {type(value).__name__} at position {position}"
                    )
                valid.append(value is not None)
                yield cast(_Scalar, value)

        scalars = checked_values()
        storage: _Storage
        if dtype is String:
            storage = _StringStorage.from_chunks(_utf8_chunks(cast(Iterable[str | None], scalars)))
        elif dtype is Categorical:
            storage = _CategoricalStorage.from_strings(cast(Iterable[str | None], scalars))
        elif dtype is Boolean:
            storage = Bitmap.from_bools(value is True for value in scalars)
        else:
            typecode = dtype.array_typecode
            assert typecode is not None
            storage = array(typecode, (0 if value is None else value for value in scalars))
        self._init_storage(dtype, storage, Bitmap.from_bools(map(bool, valid)))

    def _init_storage(self, dtype: DType, values: _Storage, validity: Bitmap) -> None:
        self._dtype = dtype
        self._values = values
        self._validity = validity
        self._length = len(validity)
        self._null_count = self._length - validity.popcount()

    def _with_storage(
        self, values: _Storage, validity: Bitmap, *, dtype: DType | None = None
    ) -> Self:
        result = self.__class__.__new__(self.__class__)
        result._init_storage(self._dtype if dtype is None else dtype, values, validity)
        return result

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

    def _is_valid(self, index: int) -> bool:
        if type(index) is not int:
            raise TypeError("column index must be an int")
        return self._validity.get(index)

    def get(self, index: int) -> _Scalar:
        """Read a zero-based position. Indices must be built-in integers."""
        if not self._is_valid(index):
            return None
        values = self._values
        if isinstance(values, Bitmap):
            return values.get(index)
        if isinstance(values, _StringStorage):
            return values.data[values.offsets[index] : values.offsets[index + 1]].decode("utf-8")
        if isinstance(values, _CategoricalStorage):
            return values.dictionary[values.codes[index]]
        return values[index]

    def to_list(self) -> list[_Scalar]:
        return [self.get(index) for index in range(self._length)]

    def slice(self, offset: int, length: int) -> Self:
        """Copy a range wholly inside the column. An empty range may start at the end."""
        if type(offset) is not int or type(length) is not int:
            raise TypeError("slice offset and length must be ints")
        validity = self._validity.slice(offset, length)
        values = self._values
        selected: _Storage
        if isinstance(values, (Bitmap, _StringStorage)):
            selected = values.slice(offset, length)
        elif isinstance(values, _CategoricalStorage):
            selected = _CategoricalStorage(
                values.codes[offset : offset + length], values.dictionary
            )
        else:
            selected = values[offset : offset + length]
        return self._with_storage(selected, validity)

    def take(self, indices: Iterable[int]) -> Self:
        """Copy the selected positions in order, including repeated positions."""
        values = self._values
        if not isinstance(values, (_StringStorage, _CategoricalStorage)):
            return self.from_list((self.get(index) for index in indices), self._dtype)

        valid = bytearray()

        def checked_indices() -> Iterator[int]:
            for index in indices:
                valid.append(self._is_valid(index))
                yield index

        if isinstance(values, _StringStorage):
            data = memoryview(values.data)
            selected = _StringStorage.from_chunks(
                data[values.offsets[i] : values.offsets[i + 1]] for i in checked_indices()
            )
        else:
            codes = array("i", (values.codes[i] for i in checked_indices()))
            selected = _CategoricalStorage(codes, values.dictionary)
        return self._with_storage(selected, Bitmap.from_bools(map(bool, valid)))

    def check_invariants(self) -> None:
        """Assert that storage matches the dtype and uses canonical null placeholders."""
        _check_bitmap(self._validity, self._length)
        assert self._null_count == self._length - self._validity.popcount(), "incorrect null count"
        values = self._values
        if self._dtype is String:
            assert isinstance(values, _StringStorage), "expected string storage"
            values.check_invariants(self._validity)
            return
        if self._dtype is Categorical:
            assert isinstance(values, _CategoricalStorage), "expected categorical storage"
            values.check_invariants(self._validity)
            return
        if self._dtype is Boolean:
            _check_bitmap(values, self._length)
        else:
            assert self._dtype is Int64 or self._dtype is Float64, "unsupported column dtype"
            _check_array(values, self._dtype.array_typecode)
            assert len(values) == self._length, "value length differs from column length"

        for index in range(self._length):
            if not self._validity.get(index):
                value = values.get(index) if isinstance(values, Bitmap) else values[index]
                assert value == 0, "nonzero value at a null position"
                if self._dtype is Float64:
                    assert copysign(1.0, value) == 1.0, "negative zero at a null position"
