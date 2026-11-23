"""Equality joins on prepared columns, with original left-major output order."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from math import isnan
from typing import Literal, cast

from ..column import Column
from ..errors import ColumnNotFoundError, SchemaError
from ..kernels.aggregate import _distinct_key
from .sort import argsort

type JoinMode = Literal["inner", "left", "right", "full"]
type KeyNames = str | Sequence[str]
type _Pair = tuple[int | None, int | None]


def _ordered_pairs(
    matches: Iterable[Sequence[int]], right_height: int, how: JoinMode
) -> list[_Pair]:
    keep_left = how in ("left", "full")
    used = bytearray(right_height) if how in ("right", "full") else None
    pairs: list[_Pair] = []
    for left, rights in enumerate(matches):
        for right in rights:
            pairs.append((left, right))
            if used is not None:
                used[right] = 1
        if keep_left and not rights:
            pairs.append((left, None))
    if used is not None:
        pairs.extend((None, right) for right, matched in enumerate(used) if not matched)
    return pairs


def _hash_key(keys: Sequence[Column], row: int) -> tuple[object, ...] | None:
    components = []
    for column in keys:
        value = column.get(row)
        if value is None:
            return None
        components.append(_distinct_key(value))
    return tuple(components)


def hash_join_pairs(
    left_keys: Sequence[Column], right_keys: Sequence[Column], *, how: JoinMode
) -> list[_Pair]:
    """Match validated key columns using ordered right-side position lists."""
    index: dict[tuple[object, ...], list[int]] = {}
    for right in range(len(right_keys[0])):
        key = _hash_key(right_keys, right)
        if key is not None:
            index.setdefault(key, []).append(right)

    def probe() -> Iterator[Sequence[int]]:
        for left in range(len(left_keys[0])):
            key = _hash_key(left_keys, left)
            yield () if key is None else index.get(key, ())

    return _ordered_pairs(probe(), len(right_keys[0]), how)


def _sorted_valid_rows(keys: Sequence[Column]) -> list[int]:
    nullable = [column for column in keys if column.null_count]
    if not nullable:
        return argsort(keys)
    valid = [i for i in range(len(keys[0])) if all(column._is_valid(i) for column in nullable)]
    permutation = argsort([column.take(valid) for column in keys])
    return [valid[i] for i in permutation]


def _compare(
    left_keys: Sequence[Column], left: int, right_keys: Sequence[Column], right: int
) -> int:
    for a, b in zip(left_keys, right_keys, strict=True):
        x, y = a.get(left), b.get(right)
        if x == y:
            continue
        if isinstance(x, float) and isnan(x):
            if isinstance(y, float) and isnan(y):
                continue
            return 1
        if isinstance(y, float) and isnan(y):
            return -1
        assert x is not None and y is not None
        return -1 if x < y else 1
    return 0


def _run_end(keys: Sequence[Column], rows: list[int], start: int) -> int:
    end = start + 1
    while end < len(rows) and _compare(keys, rows[start], keys, rows[end]) == 0:
        end += 1
    return end


def sort_merge_join_pairs(
    left_keys: Sequence[Column], right_keys: Sequence[Column], *, how: JoinMode
) -> list[_Pair]:
    """Merge stable sorted runs, assigning each matching left row its full right run."""
    left = _sorted_valid_rows(left_keys)
    right = _sorted_valid_rows(right_keys)
    matches: list[Sequence[int]] = [()] * len(left_keys[0])
    i = j = 0
    while i < len(left) and j < len(right):
        order = _compare(left_keys, left[i], right_keys, right[j])
        if order < 0:
            i += 1
        elif order > 0:
            j += 1
        else:
            left_end = _run_end(left_keys, left, i)
            right_end = _run_end(right_keys, right, j)
            run = right[j:right_end]
            for position in range(i, left_end):
                matches[left[position]] = run
            i, j = left_end, right_end
    return _ordered_pairs(matches, len(right_keys[0]), how)


def _names(spec: KeyNames) -> tuple[str, ...]:
    if isinstance(spec, str):
        spec = (spec,)
    if not isinstance(spec, Sequence) or isinstance(spec, (bytes, bytearray)):
        raise TypeError("join keys must be a name or sequence of names")
    names = tuple(spec)
    if any(type(name) is not str for name in names):
        raise TypeError("join key names must be strings")
    if not names or len(set(names)) != len(names):
        raise SchemaError("join keys must be nonempty and unique on each side")
    return names


def _key_columns(columns: Mapping[str, Column], names: tuple[str, ...]) -> list[Column]:
    try:
        return [columns[name] for name in names]
    except KeyError as error:
        raise ColumnNotFoundError(f"column not found: {error.args[0]!r}") from None


def _right_payload(
    left: Mapping[str, Column],
    right: Mapping[str, Column],
    right_names: tuple[str, ...],
    suffix: str,
) -> dict[str, Column]:
    payload = {}
    for name, column in right.items():
        if name in right_names:
            continue
        output = name + suffix if name in left else name
        if output in left or output in payload:
            raise SchemaError(f"duplicate join output name: {output!r}")
        payload[output] = column
    return payload


def _gather(column: Column, indices: list[int | None]) -> Column:
    if None not in indices:
        return column.take(cast(list[int], indices))
    return Column((None if i is None else column.get(i) for i in indices), column.dtype)


def _materialize(
    left: Mapping[str, Column],
    right_payload: Mapping[str, Column],
    right_key_for_left: Mapping[str, Column],
    pairs: list[_Pair],
) -> dict[str, Column]:
    left_rows = [i for i, _ in pairs]
    right_rows = [j for _, j in pairs]
    needs_coalesce = None in left_rows
    result = {}
    for name, column in left.items():
        if needs_coalesce and name in right_key_for_left:
            other = right_key_for_left[name]
            values = (column.get(i) if i is not None else other.get(cast(int, j)) for i, j in pairs)
            result[name] = Column(values, column.dtype)
        else:
            result[name] = _gather(column, left_rows)
    for name, column in right_payload.items():
        result[name] = _gather(column, right_rows)
    return result


def join_columns(
    left: Mapping[str, Column],
    right: Mapping[str, Column],
    *,
    on: KeyNames | None,
    left_on: KeyNames | None,
    right_on: KeyNames | None,
    how: JoinMode,
    suffix: str,
) -> tuple[dict[str, Column], int]:
    if how not in ("inner", "left", "right", "full"):
        raise ValueError(f"unsupported join mode: {how!r}")
    if type(suffix) is not str:
        raise TypeError("suffix must be a string")
    if on is not None and left_on is None and right_on is None:
        left_names = right_names = _names(on)
    elif on is None and left_on is not None and right_on is not None:
        left_names, right_names = _names(left_on), _names(right_on)
    else:
        raise SchemaError("specify either on or both left_on and right_on")
    if len(left_names) != len(right_names):
        raise SchemaError("join key counts must match")
    left_keys = _key_columns(left, left_names)
    right_keys = _key_columns(right, right_names)
    for a, b in zip(left_keys, right_keys, strict=True):
        if a.dtype is not b.dtype:
            raise SchemaError("corresponding join keys must have identical dtypes")
    payload = _right_payload(left, right, right_names, suffix)
    pairs = hash_join_pairs(left_keys, right_keys, how=how)
    coalesced = dict(zip(left_names, right_keys, strict=True))
    return _materialize(left, payload, coalesced, pairs), len(pairs)
