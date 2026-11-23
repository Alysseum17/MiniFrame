"""MiniFrame's MF1 version-1 reader and writer. See docs/mf1-format.md."""

import os
from array import array
from dataclasses import dataclass
from pathlib import Path
from struct import Struct
from tempfile import NamedTemporaryFile
from typing import BinaryIO
from zlib import crc32

from ..bitmap import Bitmap
from ..column import Column, _CategoricalStorage, _Storage, _StringStorage
from ..dtype import Boolean, Categorical, DType, Float64, Int64, String
from ..errors import CorruptedFileError, SchemaError
from ..frame import DataFrame

_MAGIC = b"MF1\0"
_VERSION = 1
_PREFIX = Struct("<4sHII")
_U16 = Struct("<H")
_U32 = Struct("<I")
_DESCRIPTOR = Struct("<QQI")
_TAGS = {Int64: 1, Float64: 2, Boolean: 3, String: 4, Categorical: 5}
_DTYPES = {tag: dtype for dtype, tag in _TAGS.items()}
_MAX_ROWS = 8_388_608
_MAX_COLUMNS = 4096
_MAX_CELLS = 33_554_432
_MAX_ENTRIES = 1_000_000
_MAX_BUFFER = 64 << 20
_MAX_FILE = 128 << 20

type _Buffer = bytes | bytearray | memoryview


def _supported_shape(rows: int, columns: int) -> bool:
    return 0 <= rows <= _MAX_ROWS and columns <= _MAX_COLUMNS and rows * columns <= _MAX_CELLS


def _encode_text(value: str, area: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise SchemaError(f"{area}: invalid UTF-8 text") from error


def _encode_dictionary(entries: tuple[str, ...], name: str) -> bytearray:
    result = bytearray(_U32.pack(len(entries)))
    area = f"Column {name!r} dictionary"
    for entry in entries:
        encoded = _encode_text(entry, area)
        if len(result) + _U32.size + len(encoded) > _MAX_BUFFER:
            raise SchemaError(f"Column {name!r}: dictionary exceeds MF1 buffer limit")
        result.extend(_U32.pack(len(encoded)))
        result.extend(encoded)
    return result


def _physical_buffers(column: Column, name: str) -> list[_Buffer]:
    buffers: list[_Buffer] = []
    if column.null_count:
        buffers.append(column._validity.to_bytes())
    values = column._values
    if isinstance(values, _StringStorage):
        buffers.extend((memoryview(values.offsets).cast("B"), memoryview(values.data)))
    elif isinstance(values, _CategoricalStorage):
        buffers.extend(
            (memoryview(values.codes).cast("B"), _encode_dictionary(values.dictionary, name))
        )
    elif isinstance(values, Bitmap):
        buffers.append(values.to_bytes())
    else:
        buffers.append(memoryview(values).cast("B"))
    return buffers


def _prepare(df: DataFrame) -> tuple[bytearray, list[_Buffer]]:
    if not _supported_shape(*df.shape):
        raise SchemaError("DataFrame exceeds MF1 dimension limits")
    metadata: list[tuple[bytes, list[_Buffer]]] = []
    header_size = _PREFIX.size
    payload_size = 0
    entries = 0
    for name, column in df._columns.items():
        encoded = _encode_text(name, "Column name")
        if not 1 <= len(encoded) <= 65535:
            raise SchemaError("MF1 column name must occupy 1 to 65535 UTF-8 bytes")
        if len(column) != df.height:
            raise SchemaError(f"Column {name!r}: length differs from frame height")
        if isinstance(column._values, _CategoricalStorage):
            entries += len(column._values.dictionary)
            if entries > _MAX_ENTRIES:
                raise SchemaError("MF1 dictionary entry limit exceeded")
        try:
            column.check_invariants()
        except AssertionError as error:
            raise SchemaError(f"Column {name!r}: {error}") from error
        prefix = (
            _U16.pack(len(encoded))
            + encoded
            + bytes((_TAGS[column.dtype], bool(column.null_count)))
        )
        buffers = _physical_buffers(column, name)
        if any(len(buffer) > _MAX_BUFFER for buffer in buffers):
            raise SchemaError(f"Column {name!r}: MF1 buffer limit exceeded")
        header_size += len(prefix) + len(buffers) * _DESCRIPTOR.size
        payload_size += sum(map(len, buffers))
        if header_size + _U32.size + payload_size + len(_MAGIC) > _MAX_FILE:
            raise SchemaError("MF1 file size limit exceeded")
        metadata.append((prefix, buffers))

    offset = header_size + _U32.size
    header = bytearray(_PREFIX.pack(_MAGIC, _VERSION, *df.shape))
    payload: list[_Buffer] = []
    for prefix, buffers in metadata:
        header.extend(prefix)
        for buffer in buffers:
            header.extend(_DESCRIPTOR.pack(offset, len(buffer), crc32(buffer)))
            offset += len(buffer)
            payload.append(buffer)
    header.extend(_U32.pack(crc32(header)))
    return header, payload


def write_mf(df: DataFrame, path: str | os.PathLike[str]) -> None:
    """Atomically replace path with the canonical MF1 encoding of df."""
    if not isinstance(df, DataFrame):
        raise TypeError("df must be a DataFrame")
    header, payload = _prepare(df)
    temporary = NamedTemporaryFile(dir=Path(path).parent, delete=False)
    try:
        with temporary:
            temporary.write(header)
            temporary.writelines(payload)
            temporary.write(_MAGIC)
        os.replace(temporary.name, path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


@dataclass(slots=True)
class _ColumnMetadata:
    dtype: DType
    nullable: bool
    descriptors: list[tuple[int, int, int]]


def _lengths(dtype: DType, nullable: bool, rows: int) -> tuple[int | None, ...]:
    packed = (rows + 7) // 8
    validity = (packed,) if nullable else ()
    if dtype is Boolean:
        return validity + (packed,)
    if dtype is String:
        return validity + (4 * (rows + 1), None)
    if dtype is Categorical:
        return validity + (4 * rows, None)
    return validity + (8 * rows,)


def _read_exact(stream: BinaryIO, size: int, area: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise CorruptedFileError(f"Truncated MF1 {area}")
    return data


def _text(data: bytes | memoryview, area: str) -> str:
    try:
        return str(data, "utf-8")
    except UnicodeDecodeError as error:
        raise CorruptedFileError(f"{area}: invalid UTF-8") from error


def _read_header(stream: BinaryIO, end: int) -> tuple[int, dict[str, _ColumnMetadata]]:
    checksum = 0

    def take(size: int) -> bytes:
        nonlocal checksum
        if size > end - stream.tell():
            raise CorruptedFileError("Truncated MF1 header")
        data = _read_exact(stream, size, "header")
        checksum = crc32(data, checksum)
        return data

    magic, version, rows, width = _PREFIX.unpack(take(_PREFIX.size))
    if magic != _MAGIC:
        raise CorruptedFileError("Invalid MF1 magic")
    if version != _VERSION:
        raise CorruptedFileError(f"Unsupported MF1 version {version}")
    if not _supported_shape(rows, width):
        raise CorruptedFileError("MF1 dimension limits exceeded")
    columns = {}
    for _ in range(width):
        (length,) = _U16.unpack(take(_U16.size))
        name = _text(take(length), "Column name")
        if not name or name in columns:
            raise CorruptedFileError(f"Empty or duplicate MF1 column name: {name!r}")
        tag, flag = take(2)
        if tag not in _DTYPES:
            raise CorruptedFileError(f"Column {name!r}: unknown MF1 dtype tag {tag}")
        if flag > 1:
            raise CorruptedFileError(f"Column {name!r}: invalid null flag")
        dtype = _DTYPES[tag]
        descriptors = [
            _DESCRIPTOR.unpack(take(_DESCRIPTOR.size)) for _ in _lengths(dtype, bool(flag), rows)
        ]
        columns[name] = _ColumnMetadata(dtype, bool(flag), descriptors)
    (stored,) = _U32.unpack(_read_exact(stream, _U32.size, "header CRC"))
    if checksum != stored:
        raise CorruptedFileError("Header CRC mismatch")
    return rows, columns


def _validate_layout(columns: dict[str, _ColumnMetadata], rows: int, start: int, end: int) -> None:
    for name, meta in columns.items():
        sizes = _lengths(meta.dtype, meta.nullable, rows)
        for (offset, length, _), expected in zip(meta.descriptors, sizes, strict=True):
            if offset != start or length > min(_MAX_BUFFER, end - start):
                raise CorruptedFileError(f"Column {name!r}: invalid buffer bounds")
            if expected is not None and length != expected:
                raise CorruptedFileError(f"Column {name!r}: incorrect buffer length")
            start += length
    if start != end:
        raise CorruptedFileError("Unexpected MF1 payload size")


def _read_payload(stream: BinaryIO, columns: dict[str, _ColumnMetadata]) -> dict[str, list[bytes]]:
    payload = {}
    for name, meta in columns.items():
        buffers = []
        for _, length, checksum in meta.descriptors:
            data = _read_exact(stream, length, f"buffer in column {name!r}")
            if crc32(data) != checksum:
                raise CorruptedFileError(f"Column {name!r}: buffer CRC mismatch")
            buffers.append(data)
        payload[name] = buffers
    return payload


def _bitmap(data: bytes, rows: int, name: str) -> Bitmap:
    if rows % 8 and data[-1] >> (rows % 8):
        raise CorruptedFileError(f"Column {name!r}: nonzero bitmap padding")
    return Bitmap.from_bytes(data, rows)


def _dictionary(data: bytes, remaining_entries: int, name: str) -> tuple[str, ...]:
    area = f"Column {name!r} dictionary"
    if len(data) < _U32.size:
        raise CorruptedFileError(f"{area}: missing entry count")
    count = _U32.unpack_from(data)[0]
    pos = _U32.size
    if count > min(remaining_entries, (len(data) - pos) // _U32.size):
        raise CorruptedFileError(f"{area}: invalid entry count")
    view = memoryview(data)
    entries = []
    for _ in range(count):
        if len(data) - pos < _U32.size:
            raise CorruptedFileError(f"{area}: missing entry length")
        length = _U32.unpack_from(data, pos)[0]
        pos += _U32.size
        if length > len(data) - pos:
            raise CorruptedFileError(f"{area}: invalid entry length")
        entries.append(_text(view[pos : pos + length], area))
        pos += length
    if pos != len(data):
        raise CorruptedFileError(f"{area}: trailing bytes")
    return tuple(entries)


def _decode(
    name: str, meta: _ColumnMetadata, buffers: list[bytes], rows: int, remaining_entries: int
) -> Column:
    validity = _bitmap(buffers[0], rows, name) if meta.nullable else Bitmap(rows, True)
    if meta.nullable and validity.popcount() == rows:
        raise CorruptedFileError(f"Column {name!r}: redundant validity bitmap")
    primary = buffers[int(meta.nullable)]
    values: _Storage
    if meta.dtype is Boolean:
        values = _bitmap(primary, rows, name)
    else:
        typecode = meta.dtype.array_typecode
        assert typecode is not None
        physical = array(typecode)
        physical.frombytes(primary)
        if meta.dtype is String:
            values = _StringStorage(physical, bytearray(buffers[-1]))
        elif meta.dtype is Categorical:
            values = _CategoricalStorage(
                physical, _dictionary(buffers[-1], remaining_entries, name)
            )
        else:
            values = physical
    column = Column._from_storage(meta.dtype, values, validity)
    try:
        column.check_invariants()
    except AssertionError as error:
        raise CorruptedFileError(f"Column {name!r}: {error}") from error
    return column


def read_mf(path: str | os.PathLike[str]) -> DataFrame:
    """Read and validate a complete MF1 file, raising CorruptedFileError on failure."""
    try:
        with open(path, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if not _PREFIX.size + _U32.size + len(_MAGIC) <= size <= _MAX_FILE:
                raise CorruptedFileError("Invalid MF1 file size")
            end = size - len(_MAGIC)
            rows, metadata = _read_header(stream, end - _U32.size)
            start = stream.tell()
            _validate_layout(metadata, rows, start, end)
            stream.seek(end)
            if stream.read(len(_MAGIC) + 1) != _MAGIC:
                raise CorruptedFileError("Invalid MF1 trailing magic")
            stream.seek(start)
            payload = _read_payload(stream, metadata)
    except OSError as error:
        raise CorruptedFileError(f"Cannot read MF1 file: {error}") from error

    columns = {}
    remaining_entries = _MAX_ENTRIES
    for name, meta in metadata.items():
        column = _decode(name, meta, payload.pop(name), rows, remaining_entries)
        if isinstance(column._values, _CategoricalStorage):
            remaining_entries -= len(column._values.dictionary)
        columns[name] = column
    return DataFrame._from_columns(columns, rows)
