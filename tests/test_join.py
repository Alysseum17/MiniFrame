from collections import Counter
from math import copysign, inf, isnan, nan
from typing import Any

import duckdb
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from miniframe import (
    Boolean,
    Categorical,
    ColumnNotFoundError,
    DataFrame,
    DType,
    Float64,
    Int64,
    SchemaError,
    String,
)
from miniframe.ops.join import JoinMode, hash_join_pairs, sort_merge_join_pairs

type Value = int | float | str | bool | None
type Row = tuple[Value, ...]
type Pair = tuple[int | None, int | None]
MODES: tuple[JoinMode, ...] = ("inner", "left", "right", "full")


def nested_pairs(left: list[Row], right: list[Row], how: JoinMode) -> list[Pair]:
    def equal(x: Value, y: Value) -> bool:
        if x is None or y is None:
            return False
        return x == y or (isinstance(x, float) and isinstance(y, float) and isnan(x) and isnan(y))

    pairs: list[Pair] = []
    for i, a in enumerate(left):
        matches = [
            (i, j)
            for j, b in enumerate(right)
            if all(equal(x, y) for x, y in zip(a, b, strict=True))
        ]
        if matches:
            pairs.extend(matches)
        elif how in ("left", "full"):
            pairs.append((i, None))
    if how in ("right", "full"):
        used = {j for _, j in pairs}
        pairs.extend((None, j) for j in range(len(right)) if j not in used)
    return pairs


def canonical(row: Row) -> tuple[object, ...]:
    return tuple(("nan",) if isinstance(x, float) and isnan(x) else x for x in row)


def check_invariants(result: DataFrame) -> None:
    for name in result.columns:
        column = result[name]._column
        assert len(column) == result.height
        column.check_invariants()


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        ("inner", [(0, 0), (0, 1), (1, 0), (1, 1)]),
        ("left", [(0, 0), (0, 1), (1, 0), (1, 1), (2, None), (3, None)]),
        ("right", [(0, 0), (0, 1), (1, 0), (1, 1), (None, 2), (None, 3)]),
        ("full", [(0, 0), (0, 1), (1, 0), (1, 1), (2, None), (3, None), (None, 2), (None, 3)]),
    ],
)
def test_exact_order_and_schema(how: JoinMode, expected: list[Pair]) -> None:
    left = DataFrame({"p": range(4), "key": [1, 1, None, 2]})
    right = DataFrame({"id": [1, 1, None, 3], "p": range(4)})
    for algorithm in (hash_join_pairs, sort_merge_join_pairs):
        assert algorithm([left["key"]._column], [right["id"]._column], how=how) == expected
    result = left.join(right, left_on="key", right_on="id", how=how, suffix="_R")
    assert list(result.schema.items()) == [("p", Int64), ("key", Int64), ("p_R", Int64)]
    assert result.rows() == [
        (i, left["key"][i] if i is not None else right["id"][j], j) for i, j in expected
    ]
    check_invariants(result)
    assert left.rows() == [(0, 1), (1, 1), (2, None), (3, 2)]
    assert right.rows() == [(1, 0), (1, 1), (None, 2), (3, 3)]


def test_categorical_dictionaries_and_differently_ordered_keys() -> None:
    left = DataFrame(
        {
            "a": ["猫", "A", None, "A"],
            "p": [True, None, False, True],
            "b": [True, False, True, None],
        },
        schema={"a": Categorical},
    )
    right = DataFrame(
        {"y": [False, True, None, None], "x": ["A", "猫", "A", "Z"], "p": ["", "Ї", None, "末"]},
        schema={"x": Categorical, "p": Categorical},
    )
    result = left.join(right, left_on=["b", "a"], right_on=["y", "x"], how="full")
    assert list(result.schema.items()) == [
        ("a", Categorical),
        ("p", Boolean),
        ("b", Boolean),
        ("p_right", Categorical),
    ]
    assert result.rows() == [
        ("猫", True, True, "Ї"),
        ("A", None, False, ""),
        (None, False, True, None),
        ("A", True, None, None),
        ("A", None, None, None),
        ("Z", None, None, "末"),
    ]
    check_invariants(result)


@pytest.mark.parametrize("how", MODES)
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([], []),
        ([], [1, 2]),
        ([1, 2], []),
        ([None, None], [None, None]),
        ([1], [1]),
        ([1], [2]),
        ([3] * 3, [3] * 4),
    ],
)
def test_key_only_and_empty(a: list[int | None], b: list[int | None], how: JoinMode) -> None:
    left = DataFrame({"k": a}, schema={"k": Int64})
    right = DataFrame({"k": b}, schema={"k": Int64})
    expected = nested_pairs(left.rows(), right.rows(), how)
    for algorithm in (hash_join_pairs, sort_merge_join_pairs):
        assert algorithm([left["k"]._column], [right["k"]._column], how=how) == expected
    result = left.join(right, on="k", how=how)
    assert result.schema == {"k": Int64}
    assert result.rows() == [(a[i] if i is not None else b[j],) for i, j in expected]
    check_invariants(result)


def test_nan_cartesian_product_and_signed_zero_coalescing() -> None:
    left = DataFrame({"f": [nan, -0.0, float("nan"), nan], "s": ["s"] * 4})
    right = DataFrame({"g": [0.0, nan, nan, nan, nan, -0.0], "t": ["s"] * 5 + ["u"]})
    result = left.join(right, left_on=["f", "s"], right_on=["g", "t"], how="full")
    assert result.height == 14
    assert [canonical(r) for r in result.rows()] == [canonical((nan, "s"))] * 4 + [(0.0, "s")] + [
        canonical((nan, "s"))
    ] * 8 + [(0.0, "u")]
    assert copysign(1, result["f"][4]) == copysign(1, result["f"][13]) == -1
    check_invariants(result)


@pytest.mark.parametrize(
    "dtype,values",
    [
        (Int64, [3, None, 7]),
        (Boolean, [True, None, False]),
        (Float64, [nan, None, -0.0]),
        (String, ["Ї", None, ""]),
        (Categorical, ["猫", None, "猫"]),
    ],
)
def test_nullable_gather_on_both_sides(dtype: DType, values: list[Value]) -> None:
    left = DataFrame({"k": [0, 0, 1], "v": values}, schema={"v": dtype})
    right = DataFrame({"k": [0, 2, 2], "v": values}, schema={"v": dtype})
    result = left.join(right, on="k", how="full")
    assert list(result.schema.items()) == [("k", Int64), ("v", dtype), ("v_right", dtype)]
    expected = [
        (0, values[0], values[0]),
        (0, None, values[0]),
        (1, values[2], None),
        (2, None, None),
        (2, None, values[2]),
    ]
    assert list(map(canonical, result.rows())) == list(map(canonical, expected))
    check_invariants(result)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({}, SchemaError),
        ({"on": []}, SchemaError),
        ({"left_on": "id"}, SchemaError),
        ({"right_on": "id"}, SchemaError),
        ({"on": "id", "left_on": "id"}, SchemaError),
        ({"on": "id", "right_on": "id"}, SchemaError),
        ({"on": "id", "how": "outer"}, ValueError),
        ({"on": "id", "suffix": None}, TypeError),
        ({"on": "missing"}, ColumnNotFoundError),
        ({"left_on": "id", "right_on": "missing"}, ColumnNotFoundError),
        ({"left_on": "id", "right_on": ["id", "v"]}, SchemaError),
        ({"on": ["id", "id"]}, SchemaError),
        ({"left_on": ["id", "v"], "right_on": ["id", "id"]}, SchemaError),
        ({"on": 2}, TypeError),
        ({"on": b"id"}, TypeError),
        ({"on": {"id"}}, TypeError),
        ({"on": ["id", 1]}, TypeError),
        ({"on": "id", "suffix": ""}, SchemaError),
        ({"left_on": "id", "right_on": "v"}, SchemaError),
    ],
)
def test_invalid_arguments(kwargs: dict[str, Any], error: type[Exception]) -> None:
    df = DataFrame(schema={"id": Int64, "v": Float64})
    with pytest.raises(error):
        df.join(df, **kwargs)


def test_name_collisions_and_other_type() -> None:
    left = DataFrame({"k": [1], "v": [2]})
    with pytest.raises(TypeError):
        left.join({"k": [1]}, on="k")
    for names in (["k", "v", "v_right"], ["k", "v_right", "v"]):
        right = DataFrame({name: [1] for name in names})
        with pytest.raises(SchemaError):
            left.join(right, on="k")
        with pytest.raises(SchemaError):
            right.join(left, on="k")
    assert left.join(left, on="k").rows() == [(1, 2, 2)]


VALUES = {
    Int64: st.integers(-2, 2),
    Float64: st.sampled_from([nan, -inf, -1.0, -0.0, 0.0, 2.0, inf]),
    Boolean: st.booleans(),
    String: st.sampled_from(["", "a", "猫"]),
    Categorical: st.sampled_from(["", "a", "猫"]),
}


@given(st.lists(st.sampled_from(list(DType)), min_size=1, max_size=3), st.data())
@settings(
    derandomize=True,
    deadline=None,
    max_examples=120
    if settings.get_current_profile_name() == "dev"
    else settings.default.max_examples,
)
def test_three_way_equivalence(types: list[DType], data: st.DataObject) -> None:
    row = st.tuples(*(st.none() | VALUES[dtype] for dtype in types))
    rows = st.lists(row, max_size=15)
    a, b = data.draw(rows), data.draw(rows)
    names = [f"k{i}" for i in range(len(types))]
    schema = dict(zip(names, types, strict=True))

    def frame(records: list[Row], payload: str) -> DataFrame:
        columns = {name: [r[i] for r in records] for i, name in enumerate(names)}
        return DataFrame(columns | {payload: range(len(records))}, schema=schema | {payload: Int64})

    left, right = frame(a, "l"), frame(b, "r")
    left_keys, right_keys = ([df[name]._column for name in names] for df in (left, right))
    ca, cb = (Counter(canonical(key) for key in side if None not in key) for side in (a, b))
    for how in MODES:
        expected = nested_pairs(a, b, how)
        assert hash_join_pairs(left_keys, right_keys, how=how) == expected
        assert sort_merge_join_pairs(left_keys, right_keys, how=how) == expected
        result = left.join(right, on=names, how=how)
        assert list(result.schema.items()) == list((schema | {"l": Int64, "r": Int64}).items())
        assert list(map(canonical, result.rows())) == [
            canonical((*(a[i] if i is not None else b[j]), i, j)) for i, j in expected
        ]
        check_invariants(result)
        if how == "inner":
            assert result.height == sum(n * cb[k] for k, n in ca.items())
        if how in ("left", "full"):
            assert {i for i, _ in expected if i is not None} == set(range(len(a)))
        if how in ("right", "full"):
            assert {j for _, j in expected if j is not None} == set(range(len(b)))
    assert left.equals(frame(a, "l")) and right.equals(frame(b, "r"))


@pytest.mark.parametrize("how", MODES)
def test_against_polars_and_duckdb(how: JoinMode) -> None:
    a = {
        "f": [nan, 0.0, None, nan, 1.0, 2.0],
        "s": ["x", "y", "x", "x", "x", None],
        "l": list(range(6)),
    }
    b = {
        "g": [0.0, nan, nan, None, 3.0, 2.0],
        "t": ["y", "x", "x", "x", "x", None],
        "r": list(range(6)),
    }
    result = DataFrame(a).join(DataFrame(b), left_on=["f", "s"], right_on=["g", "t"], how=how)
    actual = list(map(canonical, result.rows()))
    check_invariants(result)
    polars_rows = (
        pl.DataFrame(a)
        .join(pl.DataFrame(b), left_on=["f", "s"], right_on=["g", "t"], how=how, coalesce=False)
        .select(pl.coalesce("f", "g"), pl.coalesce("s", "t"), "l", "r")
        .sort(["l", "r"], nulls_last=True)
        .rows()
    )
    assert actual == list(map(canonical, polars_rows))
    with duckdb.connect() as db:
        db.execute("create table a(f double,s varchar,l integer)")
        db.execute("create table b(g double,t varchar,r integer)")
        db.executemany("insert into a values (?, ?, ?)", list(zip(*a.values(), strict=True)))
        db.executemany("insert into b values (?, ?, ?)", list(zip(*b.values(), strict=True)))
        duck_rows = db.execute(f"""
            select coalesce(f,g), coalesce(s,t), l, r
            from a {how} join b on f=g and s=t
            order by l nulls last, r nulls last
        """).fetchall()
    assert actual == list(map(canonical, duck_rows))


def test_large_interleaved_runs() -> None:
    groups = 1021
    left = DataFrame({"k": list(reversed(range(groups))) * 4})
    right = DataFrame({"k": list(range(groups)) * 3})
    expected = [
        (i, j * groups + (groups - 1 - i % groups)) for i in range(left.height) for j in range(3)
    ]
    for algorithm in (hash_join_pairs, sort_merge_join_pairs):
        assert algorithm([left["k"]._column], [right["k"]._column], how="inner") == expected


@pytest.mark.parametrize("how", MODES)
def test_all_nan_keys(how: JoinMode) -> None:
    left = DataFrame({"k": [float("nan")] * 3, "l": range(3)})
    right = DataFrame({"k": [float("nan")] * 4, "r": range(4)})
    expected = [(i, j) for i in range(3) for j in range(4)]
    for algorithm in (hash_join_pairs, sort_merge_join_pairs):
        assert algorithm([left["k"]._column], [right["k"]._column], how=how) == expected
    result = left.join(right, on="k", how=how)
    assert result.select("l", "r").rows() == expected
    assert all(isnan(value) for value in result["k"].to_list())
    check_invariants(result)
