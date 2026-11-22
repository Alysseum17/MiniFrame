from math import inf, isnan, nan
from pickle import dumps
from random import Random
from struct import pack

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from miniframe import Boolean, Categorical, DType, Float64, Int64, ShapeError, String
from miniframe.column import Column
from miniframe.ops.sort import argsort

type Scalar = int | float | bool | str | None


@pytest.mark.parametrize(
    ("descending", "last", "expected"),
    [
        (False, False, [1, 5, 3, 0, 2, 4]),
        (True, False, [1, 5, 0, 2, 4, 3]),
        (False, True, [3, 0, 2, 4, 1, 5]),
        (True, True, [0, 2, 4, 3, 1, 5]),
    ],
)
def test_stable_indices_and_null_placement(
    descending: bool, last: bool, expected: list[int]
) -> None:
    keys = [Column([2, None, 2, 1, 2, None], Int64)]
    assert argsort(keys, descending=descending, nulls_last=last) == expected


@pytest.mark.parametrize(
    ("descending", "expected"),
    [
        (False, [1, 5, 6, 2, 4, 0, 7, 3]),
        (True, [0, 7, 4, 2, 5, 6, 1, 3]),
    ],
)
def test_ieee_values_and_ties(descending: bool, expected: list[int]) -> None:
    column = Column([nan, -inf, 5.0, None, inf, -0.0, 0.0, float("nan")], Float64)
    assert argsort([column], descending=descending, nulls_last=True) == expected
    ids = Column(range(len(column)), Int64)
    assert argsort([column, ids], descending=[descending, True], nulls_last=True) == [
        {0: 7, 7: 0, 5: 6, 6: 5}.get(i, i) for i in expected
    ]


@pytest.mark.parametrize("dtype", [String, Categorical])
def test_unicode_logical_order(dtype: DType) -> None:
    keys = [Column(["🙂", "", "Я", "é", "e\u0301", None, "", "🙂"], dtype)]
    assert argsort(keys) == [5, 1, 6, 4, 3, 2, 0, 7]
    assert argsort(keys, descending=True) == [5, 0, 7, 2, 3, 4, 1, 6]


def test_multikey_flags_and_payload_alignment() -> None:
    cities = ["Kyiv", "Lviv", "Kyiv", "Kyiv", None, None, None, "Kyiv", "Lviv"]
    scores = [20, 10, 30, 20, 1, None, 2, None, 10]
    keys = [Column(cities, Categorical), Column(scores, Int64)]
    perm = argsort(keys, descending=(False, True), nulls_last=(False, True))
    assert perm == [6, 4, 5, 2, 0, 3, 7, 1, 8]
    payload = Column(["row " + str(i) for i in range(9)], String)
    assert payload.take(perm).to_list() == ["row " + str(i) for i in [6, 4, 5, 2, 0, 3, 7, 1, 8]]
    assert argsort(keys, descending=True, nulls_last=True) == [1, 8, 2, 0, 3, 7, 6, 4, 5]


@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33])
def test_merge_run_boundaries(length: int) -> None:
    indices = list(range(length))
    for values in ([None] * length, [1] * length, indices):
        assert argsort([Column(values, Int64)]) == indices
    assert argsort([Column(reversed(indices), Int64)]) == indices[::-1]
    alternating = Column([None if i % 2 else i for i in indices], Int64)
    assert argsort([alternating], descending=True) == indices[1::2] + indices[::2][::-1]


def test_large_runs_and_random_input() -> None:
    indices = list(range(8193))
    rng = Random(304)
    for values in ([1] * 4096 + [0] + [1] * 4096, [rng.randrange(100) for _ in indices]):
        expected = sorted(indices, key=values.__getitem__, reverse=True)
        assert argsort([Column(values, Int64)], descending=True) == expected


def test_key_validation() -> None:
    with pytest.raises(ValueError, match="^at least one sort key is required$"):
        argsort([])
    empty = Column([], Int64)
    for keys in (empty, "x", [empty, True]):
        with pytest.raises(TypeError):
            argsort(keys)
    with pytest.raises(ShapeError):
        argsort([empty, Column([None], Boolean)])


@pytest.mark.parametrize("flag", ["descending", "nulls_last"])
def test_flag_validation(flag: str) -> None:
    keys = [Column([], Int64), Column([], Boolean)]
    for invalid in (0, 1, None, "ff", b"ff", [True, 0], [False, None]):
        with pytest.raises(TypeError):
            argsort(keys, **{flag: invalid})
    for invalid in ([], [False], [True] * 3):
        with pytest.raises(ShapeError):
            argsort(keys, **{flag: invalid})


def reference_order(
    values: list[list[Scalar]], descending: list[bool], nulls_last: list[bool]
) -> list[int]:
    def nonnull_key(value: Scalar) -> tuple:
        return (1,) if isinstance(value, float) and isnan(value) else (0, value)

    order = list(range(len(values[0])))
    for key, reverse, last in reversed(list(zip(values, descending, nulls_last, strict=True))):
        nulls = [i for i in order if key[i] is None]
        known = sorted(
            (i for i in order if key[i] is not None),
            key=lambda i: nonnull_key(key[i]),
            reverse=reverse,
        )
        order = known + nulls if last else nulls + known
    return order


def scalar_bits(value: Scalar) -> Scalar | bytes:
    return pack("!d", value) if isinstance(value, float) else value


@settings(derandomize=True)
@given(
    data=st.data(),
    length=st.integers(0, 35),
    dtypes=st.lists(st.sampled_from(tuple(DType)), min_size=1, max_size=3),
)
def test_stable_permutation_properties(
    data: st.DataObject, length: int, dtypes: list[DType]
) -> None:
    scalars = {
        Int64: st.integers(-(2**63), 2**63 - 1) | st.integers(-2, 2),
        Float64: st.floats() | st.sampled_from([nan, -0.0, 0.0, inf, -inf]),
        Boolean: st.booleans(),
        String: st.text(),
        Categorical: st.sampled_from(["🙂", "Я", "", "a"]),
    }
    values = [
        data.draw(st.lists(st.none() | scalars[t], min_size=length, max_size=length))
        for t in dtypes
    ]
    flags = st.lists(st.booleans(), min_size=len(dtypes), max_size=len(dtypes))
    descending, nulls_last = data.draw(flags), data.draw(flags)
    keys = [Column(v, t) for v, t in zip(values, dtypes, strict=True)]
    before = [dumps(key) for key in keys]
    perm = argsort(keys, descending=descending, nulls_last=nulls_last)
    assert sorted(perm) == list(range(length))
    assert perm == reference_order(values, descending, nulls_last)
    gathered = [key.take(perm) for key in keys]
    for column, original in zip(gathered, values, strict=True):
        column.check_invariants()
        assert [scalar_bits(v) for v in column.to_list()] == [
            scalar_bits(original[i]) for i in perm
        ]
    assert argsort(gathered, descending=descending, nulls_last=nulls_last) == list(range(length))
    assert [dumps(key) for key in keys] == before


@pytest.mark.parametrize(
    ("descending", "last"),
    [
        ([False, True, False], [False, True, True]),
        ([True, False, True], [True, False, False]),
    ],
)
def test_stable_multikey_sort_against_polars(descending: list[bool], last: list[bool]) -> None:
    city = ["z", None, "z", "a", "z", None, None, "z", "a", "z"]
    score = [nan, 9.0, nan, -0.0, None, 9.0, None, inf, 0.0, nan]
    flag = [False, True, True, False, True, True, True, None, False, False]
    keys = [Column(city, String), Column(score, Float64), Column(flag, Boolean)]
    frame = pl.DataFrame({"city": city, "score": score, "flag": flag, "row": range(len(city))})
    expected = frame.sort(
        ["city", "score", "flag"], descending=descending, nulls_last=last, maintain_order=True
    )
    assert argsort(keys, descending=descending, nulls_last=last) == expected["row"].to_list()
