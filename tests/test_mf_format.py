import os
import tracemalloc
from io import BytesIO
from pathlib import Path
from struct import pack, pack_into, unpack, unpack_from
from zlib import crc32

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import frames

import miniframe as mf
from miniframe.io import mf_format

GOLDEN = bytes.fromhex(
    "4d463100 0100 03000000 01000000 0100 62 03 01"
    "3f00000000000000 0100000000000000 021b68a2"
    "4000000000000000 0100000000000000 1bdf05a5"
    "127e2d63 05 01 4d463100"
)

type WireColumn = tuple[bytes, int, int, list[bytes]]


def wire(rows: int, columns: list[WireColumn]) -> bytes:
    header = bytearray(pack("<4sHII", b"MF1\0", 1, rows, len(columns)))
    slots = []
    for name, tag, flag, buffers in columns:
        header.extend(pack("<H", len(name)) + name + bytes([tag, flag]))
        for buffer in buffers:
            slots.append((len(header), buffer))
            header.extend(bytes(20))
    payload = bytearray()
    for slot, buffer in slots:
        pack_into("<QQI", header, slot, len(header) + 4 + len(payload), len(buffer), crc32(buffer))
        payload.extend(buffer)
    return header + pack("<I", crc32(header)) + payload + b"MF1\0"


def rejected(path: Path, data: bytes | bytearray, message: str | None = None) -> None:
    path.write_bytes(data)
    with pytest.raises(mf.CorruptedFileError) as error:
        mf.read_mf(path)
    if message is not None:
        assert str(error.value) == message


def assert_same(restored: mf.DataFrame, original: mf.DataFrame) -> None:
    assert restored.equals(original)
    assert restored.shape == original.shape
    assert list(restored.schema.items()) == list(original.schema.items())
    assert restored.columns == original.columns
    for name in original.columns:
        actual, expected = restored[name], original[name]
        actual._column.check_invariants()
        expected._column.check_invariants()
        assert actual._column is not expected._column
        assert actual.null_count == expected.null_count
        if actual.dtype is mf.Float64:
            for left, right in zip(actual.to_list(), expected.to_list(), strict=True):
                if right is None:
                    assert left is None
                else:
                    assert pack("<d", left) == pack("<d", right)
        else:
            assert actual.to_list() == expected.to_list()
        if actual.dtype is mf.Categorical:
            assert actual._column._values.codes == expected._column._values.codes
            assert actual._column._values.dictionary == expected._column._values.dictionary


def test_golden_nullable_boolean(tmp_path: Path) -> None:
    df = mf.DataFrame({"b": [True, None, False]})
    path = tmp_path / "golden.mf"
    path.write_bytes(GOLDEN)
    assert_same(mf.read_mf(str(path)), df)
    mf.write_mf(df, str(path))
    assert path.read_bytes() == GOLDEN


def test_schema_descriptors_and_physical_bytes(tmp_path: Path) -> None:
    nan_bits = 0xFFF8000000000042
    df = mf.DataFrame(
        {
            "識": [None, "", "123456789"],
            "i": [-(2**63), None, 2**63 - 1],
            "f": [-0.0, unpack("<d", pack("<Q", nan_bits))[0], float("inf")],
            "b": [False, None, True],
            "c": ["猫", "", "猫"],
        },
        schema={"c": mf.Categorical},
    )
    expected = [
        ("識", 4, 1, [b"\x06", pack("<4i", 0, 0, 0, 9), b"123456789"]),
        ("i", 1, 1, [b"\x05", pack("<3q", -(2**63), 0, 2**63 - 1)]),
        ("f", 2, 0, [pack("<3Q", 1 << 63, nan_bits, 0x7FF0000000000000)]),
        ("b", 3, 1, [b"\x05", b"\x04"]),
        ("c", 5, 0, [pack("<3i", 0, 1, 0), pack("<II", 2, 3) + "猫".encode() + pack("<I", 0)]),
    ]
    path = tmp_path / "mixed"
    mf.write_mf(df, path)
    raw = path.read_bytes()
    cursor = BytesIO(raw)
    assert unpack("<4sHII", cursor.read(14)) == (b"MF1\0", 1, 3, 5)
    descriptors = []
    for name, tag, null_flag, buffers in expected:
        size = unpack("<H", cursor.read(2))[0]
        assert cursor.read(size) == name.encode()
        assert unpack("<BB", cursor.read(2)) == (tag, null_flag)
        for buffer in buffers:
            descriptors.append((*unpack("<QQI", cursor.read(20)), buffer))
    header_crc = crc32(raw[: cursor.tell()])
    assert unpack("<I", cursor.read(4)) == (header_crc,)
    for offset, length, checksum, buffer in descriptors:
        assert (offset, length, checksum) == (cursor.tell(), len(buffer), crc32(buffer))
        assert cursor.read(length) == buffer
        if buffer == b"123456789":
            assert checksum == 0xCBF43926
    assert cursor.read() == b"MF1\0"
    assert_same(mf.read_mf(path), df)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (0, 5), (5, 0), (9, 5)])
def test_empty_shapes_and_all_null_columns(tmp_path: Path, shape: tuple[int, int]) -> None:
    height, width = shape
    schema = {dtype.value: dtype for dtype in list(mf.DType)[:width]}
    df = (
        mf.DataFrame({name: [None] * height for name in schema}, schema=schema)
        if width
        else mf.DataFrame({"x": range(height)}, schema={"x": mf.Int64}).drop("x")
    )
    path = tmp_path / "empty"
    mf.write_mf(df, path)
    assert df.shape == shape
    assert_same(mf.read_mf(path), df)


def test_long_strings_bitmaps_and_float_bits(tmp_path: Path) -> None:
    bits = [
        0,
        1 << 63,
        0xFFF0000000000000,
        0x7FF0000000000000,
        0x7FF0000000000001,
        0x7FF80000000000FF,
    ]
    values = [unpack("<d", pack("<Q", bit))[0] for bit in bits] + [None]
    long_text = '🙂,\n"Київ\x00' * 10_000
    df = mf.DataFrame(
        {
            "float": values * 37,
            "bool": [None if i % 17 == 0 else i % 3 == 1 for i in range(259)],
            "text": [long_text, "", None] + ["á"] * 256,
        }
    )
    path = tmp_path / "long"
    mf.write_mf(df, path)
    assert_same(mf.read_mf(path), df)


def test_sliced_category_dictionaries_stay_independent(tmp_path: Path) -> None:
    df = mf.DataFrame(
        {"left": ["unused", "x", "y", None, "x"], "right": ["y", "x", "y", None, "x"]},
        schema=dict.fromkeys(["left", "right"], mf.Categorical),
    ).tail(4)
    path = tmp_path / "cats"
    for original in [df, df.head(0)]:
        mf.write_mf(original, path)
        restored = mf.read_mf(path)
        assert_same(restored, original)
        assert restored["left"]._column._values.dictionary == ("unused", "x", "y")
        assert restored["right"]._column._values.dictionary == ("y", "x")


def test_every_prefix_and_byte_replacement(tmp_path: Path) -> None:
    path = tmp_path / "mutated"
    for original in (GOLDEN, wire(5, []), wire(0, [(b"s", 4, 0, [bytes(4), b""])])):
        for stop in range(len(original)):
            rejected(path, original[:stop])
    for offset, initial in enumerate(GOLDEN):
        sample = bytearray(GOLDEN)
        for replacement in range(256):
            if replacement != initial:
                sample[offset] = replacement
                rejected(path, sample)


def changed_header(offset: int, replacement: bytes) -> bytearray:
    sample = bytearray(GOLDEN)
    sample[offset : offset + len(replacement)] = replacement
    pack_into("<I", sample, 59, crc32(sample[:59]))
    return sample


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (0, b"X", "Invalid MF1 magic"),
        (4, pack("<H", 2), "Unsupported MF1 version 2"),
        (6, pack("<I", 2**32 - 1), "MF1 dimension limits exceeded"),
        (10, pack("<I", 2**32 - 1), "MF1 dimension limits exceeded"),
        (14, b"\xff\xff", "Truncated MF1 header"),
        (16, b"\xff", "Column name: invalid UTF-8"),
        (17, b"\xff", "Column 'b': unknown MF1 dtype tag 255"),
        (18, b"\x02", "Column 'b': invalid null flag"),
        (19, pack("<Q", 18), "Column 'b': invalid buffer bounds"),
        (19, pack("<Q", 2**64 - 1), "Column 'b': invalid buffer bounds"),
        (27, pack("<Q", 2**64 - 1), "Column 'b': invalid buffer bounds"),
        (27, pack("<Q", 0), "Column 'b': incorrect buffer length"),
        (39, pack("<Q", 63), "Column 'b': invalid buffer bounds"),
        (39, pack("<Q", 65), "Column 'b': invalid buffer bounds"),
        (47, pack("<Q", 2), "Column 'b': invalid buffer bounds"),
    ],
)
def test_header_structure(tmp_path: Path, offset: int, replacement: bytes, message: str) -> None:
    rejected(tmp_path / "bad", changed_header(offset, replacement), message)


def test_checksums_footer_and_extra_bytes(tmp_path: Path) -> None:
    path = tmp_path / "bad"
    for offset, message in [
        (59, "Header CRC mismatch"),
        (64, "Column 'b': buffer CRC mismatch"),
        (63, "Column 'b': buffer CRC mismatch"),
        (66, "Invalid MF1 trailing magic"),
    ]:
        sample = bytearray(GOLDEN)
        sample[offset] ^= 1
        rejected(path, sample, message)
    for sample in [GOLDEN + b"!", GOLDEN[:-4] + b"!" + GOLDEN[-4:]]:
        rejected(path, sample, "Unexpected MF1 payload size")
    empty = bytearray(wire(0, [(b"i", 1, 0, [b""])]))
    pack_into("<I", empty, 35, 1)
    pack_into("<I", empty, 39, crc32(empty[:39]))
    rejected(path, empty, "Column 'i': buffer CRC mismatch")


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ([(b"", 1, 0, [b""])], "Empty or duplicate MF1 column name: ''"),
        ([(b"n", 1, 0, [b""])] * 2, "Empty or duplicate MF1 column name: 'n'"),
        ([(b"b", 3, 1, [b"", b""])], "Column 'b': redundant validity bitmap"),
        ([(b"b", 3, 1, [b""])], "Truncated MF1 header"),
    ],
)
def test_schema_corruption(tmp_path: Path, columns: list[WireColumn], message: str) -> None:
    rejected(tmp_path / "bad", wire(0, columns), message)


@pytest.mark.parametrize(
    ("tag", "flag", "buffers", "reason"),
    [
        (1, 0, [bytes(15)], "incorrect buffer length"),
        (2, 0, [bytes(17)], "incorrect buffer length"),
        (3, 0, [b""], "incorrect buffer length"),
        (3, 0, [b"\x80"], "nonzero bitmap padding"),
        (3, 1, [b"\x83", b"\x01"], "nonzero bitmap padding"),
        (3, 1, [b"\x03", b"\x01"], "redundant validity bitmap"),
        (3, 1, [b"\x01", b"\x03"], "nonzero value at a null position"),
        (1, 1, [b"\x01", pack("<2q", 5, -1)], "nonzero value at a null position"),
        (2, 1, [b"\x01", pack("<2d", 2.0, -0.0)], "negative zero at a null position"),
        (2, 1, [b"\x01", pack("<2d", 2.0, float("nan"))], "nonzero value at a null position"),
        (4, 0, [bytes(8), b""], "incorrect buffer length"),
        (4, 0, [pack("<3i", 1, 1, 2), b"ab"], "first string offset must be zero"),
        (4, 0, [pack("<3i", 0, -1, 2), b"ab"], "string offset out of order or out of bounds"),
        (4, 0, [pack("<3i", 0, 3, 2), b"ab"], "string offset out of order or out of bounds"),
        (4, 0, [pack("<3i", 0, 1, 1), b"ab"], "final string offset differs from byte count"),
        (4, 0, [pack("<3i", 0, 1, 2), "é".encode()], "invalid UTF-8 at position 0"),
        (4, 1, [b"\x01", pack("<3i", 0, 1, 2), b"ab"], "nonempty string at a null position"),
        (5, 0, [bytes(4), bytes(4)], "incorrect buffer length"),
        (5, 0, [pack("<2i", -1, 0), pack("<II", 1, 0)], "categorical code out of range"),
        (5, 0, [pack("<2i", 0, 1), pack("<II", 1, 0)], "categorical code out of range"),
        (5, 1, [b"\x01", pack("<2i", 0, 1), pack("<II", 1, 0)], "nonzero code at a null position"),
        (5, 0, [bytes(8), pack("<III", 2, 0, 0)], "duplicate dictionary entry"),
    ],
)
def test_checksum_valid_storage_corruption(
    tmp_path: Path, tag: int, flag: int, buffers: list[bytes], reason: str
) -> None:
    rejected(tmp_path / "bad", wire(2, [(b"x", tag, flag, buffers)]), f"Column 'x': {reason}")


@pytest.mark.parametrize(
    ("dictionary", "reason"),
    [
        (b"\x01", "missing entry count"),
        (pack("<I", 2**32 - 1), "invalid entry count"),
        (pack("<II", 1, 2**32 - 1), "invalid entry length"),
        (pack("<II", 2, 4) + b"text", "missing entry length"),
        (pack("<II", 1, 1) + b"\xff", "invalid UTF-8"),
        (pack("<I", 0) + b"!", "trailing bytes"),
    ],
)
def test_malformed_bounded_dictionary(tmp_path: Path, dictionary: bytes, reason: str) -> None:
    rejected(
        tmp_path / "dict",
        wire(0, [(b"x", 5, 0, [b"", dictionary])]),
        f"Column 'x' dictionary: {reason}",
    )


def test_tiny_files_with_extreme_claims_have_bounded_allocation(tmp_path: Path) -> None:
    path = tmp_path / "claims"
    samples = [
        (wire(8_388_608, [(b"i", 1, 0, [b""])]), "Column 'i': incorrect buffer length"),
        (
            wire(8_388_608, [(bytes([65 + n]), 1, 0, [b""]) for n in range(5)]),
            "MF1 dimension limits exceeded",
        ),
        (changed_header(19, pack("<Q", (1 << 64) - 4)), "Column 'b': invalid buffer bounds"),
        (changed_header(27, pack("<Q", (1 << 64) - 1)), "Column 'b': invalid buffer bounds"),
        (
            wire(0, [(b"c", 5, 0, [b"", pack("<I", 4_000_000_000)])]),
            "Column 'c' dictionary: invalid entry count",
        ),
    ]
    for sample, reason in samples:
        path.write_bytes(sample)
        tracemalloc.start()
        try:
            with pytest.raises(mf.CorruptedFileError) as error:
                mf.read_mf(path)
            assert str(error.value) == reason
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 1024 * 1024
    with path.open("wb") as stream:
        stream.truncate((128 << 20) + 1)
    with pytest.raises(mf.CorruptedFileError) as exc:
        mf.read_mf(path)
    assert str(exc.value) == "Invalid MF1 file size"


def test_name_and_dimension_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "limits"
    limit_name = "猫" * 21845
    for original in [
        mf.DataFrame(schema={limit_name: mf.Boolean}),
        mf.DataFrame._from_columns({}, 8_388_608),
    ]:
        mf.write_mf(original, path)
        assert_same(mf.read_mf(path), original)
    for invalid in [
        mf.DataFrame(schema={limit_name + "x": mf.Boolean}),
        mf.DataFrame(schema={"": mf.Int64}),
        mf.DataFrame(schema={"\ud800": mf.Int64}),
        mf.DataFrame({"cat": ["\ud800"]}, schema={"cat": mf.Categorical}),
        mf.DataFrame._from_columns({}, 8_388_609),
        mf.DataFrame(schema={str(i): mf.Boolean for i in range(4097)}),
    ]:
        path.write_bytes(GOLDEN)
        with pytest.raises(mf.SchemaError):
            mf.write_mf(invalid, path)
        assert path.read_bytes() == GOLDEN


@pytest.mark.parametrize(
    ("limit", "maximum", "column", "message"),
    [
        ("_MAX_BUFFER", 7, mf.Series("i", [1]), "Column 'i': MF1 buffer limit exceeded"),
        (
            "_MAX_BUFFER",
            8,
            mf.Series("c", ["a"], mf.Categorical),
            "Column 'c': dictionary exceeds MF1 buffer limit",
        ),
        ("_MAX_FILE", 49, mf.Series("i", [1]), "MF1 file size limit exceeded"),
        (
            "_MAX_ENTRIES",
            1,
            mf.Series("c", ["a", "b"], mf.Categorical),
            "MF1 dictionary entry limit exceeded",
        ),
    ],
)
def test_symmetric_size_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    maximum: int,
    column: mf.Series,
    message: str,
) -> None:
    df = mf.DataFrame.from_series([column])
    path = tmp_path / "limit"
    mf.write_mf(df, path)
    original = path.read_bytes()
    monkeypatch.setattr(mf_format, limit, maximum)
    with pytest.raises(mf.SchemaError) as exc:
        mf.write_mf(df, path)
    assert str(exc.value) == message
    assert path.read_bytes() == original
    with pytest.raises(mf.CorruptedFileError):
        mf.read_mf(path)


def test_dictionary_budget_spans_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cats"
    df = mf.DataFrame(
        {"a": ["same"], "b": ["same"]}, schema={"a": mf.Categorical, "b": mf.Categorical}
    )
    mf.write_mf(df, path)
    monkeypatch.setattr(mf_format, "_MAX_ENTRIES", 1)
    with pytest.raises(mf.CorruptedFileError) as exc:
        mf.read_mf(path)
    assert str(exc.value) == "Column 'b' dictionary: invalid entry count"
    with pytest.raises(mf.SchemaError):
        mf.write_mf(df, path)


def test_failed_writes_preserve_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dest"
    path.write_bytes(GOLDEN)
    invalid = mf.DataFrame({"x": [1, None]})
    invalid["x"]._column._values[1] = 2
    with pytest.raises(mf.SchemaError) as exc:
        mf.write_mf(invalid, path)
    assert str(exc.value) == "Column 'x': nonzero value at a null position"
    with pytest.raises(mf.SchemaError):
        mf.write_mf(mf.DataFrame._from_columns(invalid._columns, 3), path)
    with pytest.raises(TypeError):
        mf.write_mf(None, path)
    assert path.read_bytes() == GOLDEN

    def fail_replace(temporary: str, destination: Path) -> None:
        assert Path(temporary).parent == tmp_path
        assert mf.read_mf(temporary).shape == (0, 0)
        assert destination == path
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(mf_format.os, "replace", fail_replace)
    with pytest.raises(OSError, match="^simulated filesystem failure$"):
        mf.write_mf(mf.DataFrame(), path)
    assert path.read_bytes() == GOLDEN
    assert list(tmp_path.iterdir()) == [path]


def test_truncation_after_size_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "shrinking"
    path.write_bytes(GOLDEN)
    stat_fd = os.fstat

    def truncate_after_stat(fd: int) -> os.stat_result:
        result = stat_fd(fd)
        with path.open("r+b") as changing:
            changing.truncate(7)
        return result

    monkeypatch.setattr(os, "fstat", truncate_after_stat)
    with pytest.raises(mf.CorruptedFileError) as exc:
        mf.read_mf(path)
    assert str(exc.value) == "Truncated MF1 header"


def test_read_filesystem_error(tmp_path: Path) -> None:
    with pytest.raises(mf.CorruptedFileError) as exc:
        mf.read_mf(tmp_path / "absent")
    assert isinstance(exc.value.__cause__, FileNotFoundError)


def test_generated_roundtrips_and_corruption(tmp_path: Path) -> None:
    path, second = tmp_path / "mf", tmp_path / "second"

    @settings(derandomize=True, deadline=None)
    @given(frames(), st.data())
    def exercise(original: mf.DataFrame, choices: st.DataObject) -> None:
        mf.write_mf(original, path)
        raw = path.read_bytes()
        restored = mf.read_mf(path)
        assert_same(restored, original)
        for source in (original, restored):
            mf.write_mf(source, second)
            assert second.read_bytes() == raw
        rejected(second, raw[: choices.draw(st.integers(0, len(raw) - 1))])
        if original.width:
            desc = 14 + 2 + len(original.columns[0].encode()) + 2
            payload = unpack_from("<Q", raw, desc)[0]
            sample = bytearray(raw)
            sample[desc + choices.draw(st.integers(0, 7))] ^= choices.draw(st.integers(1, 255))
            pack_into("<I", sample, payload - 4, crc32(sample[: payload - 4]))
            rejected(second, sample)
            if payload < len(raw) - 4:
                sample = bytearray(raw)
                sample[choices.draw(st.integers(payload, len(raw) - 5))] ^= choices.draw(
                    st.integers(1, 255)
                )
                rejected(second, sample)
        mf.write_mf(original, second)
        assert second.read_bytes() == raw
        assert_same(restored, original)

    exercise()


def test_generated_mutators(tmp_path: Path) -> None:
    path = tmp_path / "fuzz"
    values = bytes(range(64))
    numeric = wire(8, [(b"n", 1, 0, [values])])

    @settings(derandomize=True, deadline=None)
    @given(st.integers(0, len(GOLDEN) - 1), st.integers(1, 255))
    def exercise(pos: int, delta: int) -> None:
        replacement = bytes([GOLDEN[pos] ^ delta])
        split = 1 + pos % (len(values) - 1)
        for sample in [
            GOLDEN[:pos] + replacement + GOLDEN[pos:],
            GOLDEN[:pos] + GOLDEN[pos + 1 :],
            GOLDEN[:pos] + replacement + GOLDEN[pos + 1 :],
            GOLDEN[:pos],
            numeric[: -len(values) - 4] + values[split:] + values[:split] + numeric[-4:],
            changed_header(27, pack("<Q", (1 << 64) - delta)),
        ]:
            rejected(path, sample)

    exercise()


def test_arbitrary_bytes_and_checksum_valid_payloads(tmp_path: Path) -> None:
    path = tmp_path / "bytes"

    @settings(derandomize=True, deadline=None)
    @given(st.binary(max_size=256), st.integers(0, 9), st.integers(1, 5), st.booleans(), st.data())
    def exercise(raw: bytes, rows: int, tag: int, nullable: bool, choices: st.DataObject) -> None:
        bitmap_size = (rows + 7) // 8
        lengths = {1: 8 * rows, 2: 8 * rows, 3: bitmap_size, 4: 4 * (rows + 1), 5: 4 * rows}
        length = lengths[tag]
        buffers = [choices.draw(st.binary(min_size=length, max_size=length))]
        if nullable:
            buffers.insert(0, choices.draw(st.binary(min_size=bitmap_size, max_size=bitmap_size)))
        if tag >= 4:
            buffers.append(raw)
        for sample in [raw, wire(rows, [(b"x", tag, int(nullable), buffers)])]:
            path.write_bytes(sample)
            try:
                df = mf.read_mf(path)
            except mf.CorruptedFileError:
                continue
            for name in df.columns:
                df[name]._column.check_invariants()
            mf.write_mf(df, path)
            assert path.read_bytes() == sample

    exercise()
