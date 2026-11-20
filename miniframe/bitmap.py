from collections.abc import Iterable
from operator import index
from types import NotImplementedType
from typing import Self

from .errors import ShapeError


def _length(value: int) -> int:
    value = index(value)
    if value < 0:
        raise ValueError("length must be nonnegative")
    return value


def _require_bool(value: bool) -> None:
    if type(value) is not bool:
        raise TypeError("bit value must be a bool")


class Bitmap:
    """Packed, LSB-first bits with mutable positions and a fixed logical length."""

    __slots__ = ("_data", "_length")

    def __init__(self, length: int, fill: bool = False) -> None:
        self._length = _length(length)
        _require_bool(fill)
        self._data = bytearray(b"\xff" if fill else b"\x00") * ((self._length + 7) // 8)
        if fill:
            self._clear_padding()

    def _clear_padding(self) -> None:
        used = self._length % 8
        if used:
            self._data[-1] &= (1 << used) - 1

    @classmethod
    def _from_owned(cls, length: int, data: bytearray) -> Self:
        result = cls.__new__(cls)
        result._length = length
        result._data = data
        return result

    @classmethod
    def from_bools(cls, values: Iterable[bool]) -> Self:
        data = bytearray()
        length = 0
        for value in values:
            _require_bool(value)
            position = length % 8
            if position == 0:
                data.append(0)
            if value:
                data[-1] |= 1 << position
            length += 1
        return cls._from_owned(length, data)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, length: int) -> Self:
        """Copy exactly ceil(length / 8) packed bytes and clear unused trailing bits."""
        length = _length(length)
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("packed data must be bytes or bytearray")
        if len(data) != (length + 7) // 8:
            raise ShapeError("packed byte count does not match logical length")
        result = cls._from_owned(length, bytearray(data))
        result._clear_padding()
        return result

    def __len__(self) -> int:
        return self._length

    def to_bytes(self) -> bytes:
        return bytes(self._data)

    def _position(self, bit_index: int) -> tuple[int, int]:
        bit_index = index(bit_index)
        if not 0 <= bit_index < self._length:
            raise IndexError("bit index out of range")
        byte, bit = divmod(bit_index, 8)
        return byte, 1 << bit

    def get(self, index: int) -> bool:
        byte, mask = self._position(index)
        return bool(self._data[byte] & mask)

    def set(self, index: int, value: bool) -> None:
        byte, mask = self._position(index)
        _require_bool(value)
        if value:
            self._data[byte] |= mask
        else:
            self._data[byte] &= ~mask

    def _as_int(self) -> int:
        return int.from_bytes(self._data, "little")

    @classmethod
    def _from_int(cls, value: int, length: int) -> Self:
        return cls._from_owned(length, bytearray(value.to_bytes((length + 7) // 8, "little")))

    def popcount(self) -> int:
        return self._as_int().bit_count()

    def slice(self, offset: int, length: int) -> Self:
        """Copy a range wholly inside the bitmap. An empty range may start at the end."""
        offset = index(offset)
        length = _length(length)
        if offset < 0 or offset + length > self._length:
            raise IndexError("slice outside bitmap")
        start, shift = divmod(offset, 8)
        stop = (offset + length + 7) // 8
        window = int.from_bytes(memoryview(self._data)[start:stop], "little")
        return self._from_int((window >> shift) & ((1 << length) - 1), length)

    def _check_shape(self, other: Bitmap) -> None:
        if self._length != other._length:
            raise ShapeError(f"bitmap lengths differ: {self._length} and {other._length}")

    def __and__(self, other: object) -> Self | NotImplementedType:
        if not isinstance(other, Bitmap):
            return NotImplemented
        self._check_shape(other)
        return self._from_int(self._as_int() & other._as_int(), self._length)

    def __or__(self, other: object) -> Self | NotImplementedType:
        if not isinstance(other, Bitmap):
            return NotImplemented
        self._check_shape(other)
        return self._from_int(self._as_int() | other._as_int(), self._length)

    def __invert__(self) -> Self:
        mask = (1 << self._length) - 1
        return self._from_int(self._as_int() ^ mask, self._length)
