# MF1 binary format, version 1

All integers use little-endian byte order and the stated width. Fields and buffers follow each other without alignment padding. Offsets count bytes from the beginning of the file.

| Sequence | Encoding |
| --- | --- |
| Preamble | Four bytes `4d 46 31 00` (ASCII MF1 followed by NUL), then u16 version 1 |
| Dimensions | u32 row count, u32 column count, independently meaningful even when either is zero |
| Column metadata, in schema order | u16 byte length, that many bytes of strict UTF-8 name, u8 type tag, u8 null flag (0 or 1), then one descriptor per buffer in the table below |
| Header checksum | u32 CRC32 of **all bytes from position 0 up to this field**, including the magic, version, dimensions, names, flags and descriptors |
| Payload | Every described buffer in column and descriptor order |
| Trailer | The same four magic bytes, ending at EOF |

Names must be nonempty and unique as decoded. Text is stored without Unicode normalization. Each descriptor contains a u64 absolute offset, u64 byte length and u32 CRC32 over exactly those buffer bytes. The first offset is immediately after the header checksum. Each subsequent offset is the previous offset plus its length, including for empty buffers. The final buffer ends at the trailer. Both checksum types use standard unsigned CRC-32 as computed by `zlib.crc32(data)`, including CRC 0 for empty input.

| Tag | DType | Buffers after optional validity |
| --- | --- | --- |
| 1 | Int64 | One signed 64-bit integer per row |
| 2 | Float64 | One IEEE 754 binary64 bit pattern per row, including signed zeros and NaN payloads |
| 3 | Boolean | Packed Boolean values |
| 4 | String | Signed 32-bit byte offsets (row count + 1), then concatenated UTF-8 data |
| 5 | Categorical | One signed 32-bit dictionary index per row, then the dictionary |

A null flag of 1 requires a **first** descriptor for validity and at least one null row. Flag 0 implies all rows are valid and contributes zero descriptors. Both validity and Boolean values use row 0 in the least significant bit of byte 0. Each bitmap occupies exactly ceil(rows / 8) bytes with zero tail padding. A validity bit of 1 marks a present value. Null numeric values, Boolean bits and category codes have all-zero storage, including positive zero for Float64.

String offsets start at zero, never decrease, stay within the data and end at its byte length. Each adjacent pair delimits an independently valid UTF-8 string. Null rows occupy zero bytes. The validity bit distinguishes an empty string from null.

A dictionary starts with a u32 entry count. Each entry is a u32 byte length followed by exactly that many UTF-8 bytes. The buffer ends with the last entry, or after the count for an empty dictionary. Entries must be distinct. Valid codes index this ordered dictionary starting at zero. Each column retains its own code assignments and unused entries, even for a zero-row slice.

## Implementation limits and errors

Both APIs support at most 8,388,608 rows, 4,096 columns, 33,554,432 cells, 65,535 UTF-8 bytes per name, and 1,000,000 dictionary entries summed across columns. A buffer can occupy at most 64 MiB and the entire file at most 128 MiB. These limits also apply to zero-width or zero-height shapes as applicable.

The reader checks the actual file size, the entire descriptor layout and the checksums before constructing columns. Malformed data, exceeded limits and filesystem read failures raise `CorruptedFileError`. The original filesystem exception is available as the cause. The writer raises `SchemaError` for unrepresentable frames (including empty names and unencodable text) and propagates filesystem errors. It closes a temporary file in the destination directory before atomically replacing the destination.

Version 1 stores complete, uncompressed frames on the engine's supported little-endian platforms. CRC32 detects accidental corruption. An author able to change data and recompute its checksums can produce a different valid frame.

For a longer deterministic run of the MF1 properties and mutation cases, run `HYPOTHESIS_PROFILE=ci uv run pytest tests/test_mf_format.py`.
