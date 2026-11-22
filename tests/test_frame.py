from collections.abc import Callable, Iterator
from math import copysign, isnan, nan

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from miniframe import (
    Boolean,
    Categorical,
    ColumnNotFoundError,
    DataFrame,
    Float64,
    Int64,
    SchemaError,
    Series,
    ShapeError,
    String,
    col,
    lit,
)
from miniframe.column import Column
from miniframe.eval import evaluate_expr


class SinglePass:
    def __init__(self) -> None:
        self.calls = 0

    def __iter__(self) -> Iterator[int | None]:
        self.calls += 1
        assert self.calls == 1
        yield from (None, 2)


def test_construction_and_metadata() -> None:
    source = SinglePass()
    labels = ["Ї", ""]
    df = DataFrame({"id": source, "label": labels}, schema={"label": Categorical})
    labels.clear()
    df.columns.reverse()
    df.schema.clear()
    df["id"].to_list().clear()
    assert df.shape == (2, 2)
    assert df.columns == ["id", "label"]
    assert df.schema == {"id": Int64, "label": Categorical}
    assert df.rows() == [(None, "Ї"), (2, "")]
    assert df.column("label").equals(Series("label", ["Ї", ""], Categorical))
    schema = {"text": String, "flag": Boolean, "cat": Categorical}
    for empty in (DataFrame(schema=schema), DataFrame({}, schema=schema)):
        assert empty.columns == list(schema)
        assert empty.schema == schema
        assert empty.shape == (0, 3)
    assert DataFrame({"": [None]}, schema={"": Float64}).rows() == [(None,)]


def test_from_series() -> None:
    items = [Series("猫", ["b", None, "a"], Categorical), Series("f", [nan, -0.0, None])]
    frame = DataFrame.from_series(items)
    assert frame.columns == ["猫", "f"]
    assert all(frame[item.name].equals(item) for item in items)
    items.clear()
    assert frame.shape == (3, 2)
    assert DataFrame.from_series([]).equals(DataFrame())


@pytest.mark.parametrize(
    ("construct", "error"),
    [
        (lambda: DataFrame({"a": [1], "b": []}, schema={"b": String}), ShapeError),
        (lambda: DataFrame({"a": [None]}), SchemaError),
        (lambda: DataFrame({"a": []}), SchemaError),
        (lambda: DataFrame({"a": [1, True]}), SchemaError),
        (lambda: DataFrame({"a": [1]}, schema={"a": Float64}), SchemaError),
        (lambda: DataFrame({1: [1]}), SchemaError),
        (lambda: DataFrame({"a": [1]}, schema={"z": Int64}), SchemaError),
        (lambda: DataFrame(schema={"a": None}), SchemaError),
        (lambda: DataFrame(schema={1: Int64}), SchemaError),
        (lambda: DataFrame(schema=[]), SchemaError),
        (lambda: DataFrame([1]), TypeError),
        (lambda: DataFrame.from_series([1]), TypeError),
        (lambda: DataFrame.from_series([Series("x", [1])] * 2), SchemaError),
        (lambda: DataFrame.from_series([Series("x", [1]), Series("y", [1, 2])]), ShapeError),
    ],
)
def test_invalid_construction(construct: Callable[[], DataFrame], error: type[Exception]) -> None:
    with pytest.raises(error):
        construct()


def test_project_and_replace() -> None:
    df = DataFrame({"a": [1, None, 3], "b": [2, 3, 4]})
    selected = df.select(
        "b", col("a") * col("b"), col("a").mean().alias("avg"), lit(None).cast(String).alias("null")
    )
    assert selected.columns == ["b", "a", "avg", "null"]
    assert selected.schema == {"b": Int64, "a": Int64, "avg": Float64, "null": String}
    assert selected.rows() == [(2, 2, 2.0, None), (3, None, 2.0, None), (4, 12, 2.0, None)]
    result = df.with_columns(
        lit(False).alias("new"), (col("a") + 1).alias("a"), (col("a") * 2).alias("twice")
    )
    assert result.columns == ["a", "b", "new", "twice"]
    assert result.rows() == [(2, 2, False, 2), (None, 3, False, None), (4, 4, False, 6)]
    assert df.rows() == [(1, 2), (None, 3), (3, 4)]
    assert df.with_columns() is not df
    assert df.with_columns().equals(df)
    assert df.head(0).select(col("b").mean()).schema == {"b": Float64}
    assert df.head(0).select(col("b").mean()).shape == (0, 1)


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        (lambda d: d["z"], ColumnNotFoundError),
        (lambda d: d.select("z"), ColumnNotFoundError),
        (lambda d: d.select("a", col("a").mean()), SchemaError),
        (lambda d: d.with_columns("a", lit(3).alias("a")), SchemaError),
        (lambda d: d.with_columns(lit(1).alias("x"), col("x").alias("y")), ColumnNotFoundError),
        (lambda d: d.select(lit(None)), SchemaError),
        (lambda d: d.select(1), TypeError),
        (lambda d: d.filter("a"), TypeError),
        (lambda d: d.filter(col("a")), SchemaError),
        (lambda d: d.filter(lit("text")), SchemaError),
        (lambda d: d.drop("z"), ColumnNotFoundError),
        (lambda d: d.drop(2), SchemaError),
        (lambda d: d.drop("a", "a"), SchemaError),
        (lambda d: d.rename({"z": "k"}), ColumnNotFoundError),
        (lambda d: d.rename({"a": 1}), SchemaError),
        (lambda d: d.rename({"a": "b"}), SchemaError),
        (lambda d: d.rename([]), TypeError),
        (lambda d: d.unique("ab"), TypeError),
        (lambda d: d.drop_nulls(["z"]), ColumnNotFoundError),
        (lambda d: d.sort([]), ValueError),
        (lambda d: d.sort(["a", "b"], descending=[False]), ShapeError),
        (lambda d: d.sort("a", nulls_last=[1]), TypeError),
        (lambda d: d.vstack(None), TypeError),
        (lambda d: d.equals(None), TypeError),
        (lambda d: d.equals(d, null_equal=1), TypeError),
    ],
)
def test_invalid_operations(
    operation: Callable[[DataFrame], object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        operation(DataFrame({"a": [1], "b": [2]}))


def test_result_height_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("miniframe.frame.evaluate_expr", lambda *args, **kwargs: Column([], Int64))
    with pytest.raises(ShapeError):
        DataFrame({"a": [1]}).select("a")


def test_rename_drop_and_slice() -> None:
    df = DataFrame({"a": range(7), "b": range(10, 17)})
    swapped = df.rename({"a": "b", "b": "a"})
    assert swapped.columns == ["b", "a"]
    assert swapped.rows() == df.rows()
    assert swapped.drop("b").equals(df.select(col("b").alias("a")))
    assert df.head().rows() == df.rows()[:5]
    assert df.tail().rows() == df.rows()[-5:]
    for method in (df.head, df.tail):
        with pytest.raises(TypeError):
            method(True)
        with pytest.raises(ValueError):
            method(-1)
        assert method(0).schema == df.schema
        assert method(9).equals(df)


def test_zero_columns() -> None:
    df = DataFrame({"id": range(5)})
    for bare in (df.select(), df.drop("id")):
        assert bare.shape == (5, 0)
        assert bare.rows() == [()] * 5
        records = bare.to_dicts()
        records[0]["a"] = 1
        assert records == [{"a": 1}, {}, {}, {}, {}]
        assert bare.to_dicts() == [{} for _ in range(5)]
        assert bare.filter(lit(True)).equals(bare)
        assert bare.filter(lit(False)).equals(DataFrame())
        assert bare.filter(lit(None).cast(Boolean)).equals(DataFrame())
        assert bare.unique().shape == (1, 0)
        assert bare.drop_nulls().equals(bare)
        assert bare.head(2).vstack(bare.tail(3)).equals(bare)
        assert not bare.equals(DataFrame())
        assert bare.select(lit(2)).rows() == [(2,)] * 5
    assert DataFrame().unique().shape == (0, 0)
    assert df.filter(lit(False)).schema == df.schema


def test_float_and_categorical_keys() -> None:
    df = DataFrame(
        {"f": [nan, None, -0.0, nan, 0.0, None, nan], "c": list("bbabbba"), "id": range(7)},
        schema={"c": Categorical},
    )
    assert df.unique(["c", "f"])["id"].to_list() == [0, 1, 2, 4, 6]
    assert df.unique([])["id"].to_list() == [0]
    assert df.drop_nulls(["f"])["id"].to_list() == [0, 2, 3, 4, 6]
    assert df.drop_nulls([]).equals(df)
    assert df.sort(["f", "c"], descending=[True, False], nulls_last=[True, False])[
        "id"
    ].to_list() == [6, 0, 3, 2, 4, 1, 5]
    assert df.sort(-col("id"))["id"].to_list() == list(reversed(range(7)))
    unique = df["f"].unique()
    assert unique.equals(Series("f", [nan, None, 0.0]))
    assert copysign(1, unique[2]) == -1
    assert not df.equals(df, null_equal=False)


def test_vstack_and_equality() -> None:
    schema = {"c": Categorical, "s": String, "f": Float64, "b": Boolean, "i": Int64}
    left = DataFrame(
        {"c": ["b", "猫"], "s": [None, ""], "f": [nan, -0.0], "b": [True, None], "i": [0, None]},
        schema=schema,
    )
    right = DataFrame(
        {
            "c": ["猫", "b", None],
            "s": ["猫", "", None],
            "f": [0.0, None, 3.0],
            "b": [None, False, True],
            "i": [1, 2, -3],
        },
        schema=schema,
    )
    both = left.vstack(right)
    assert both.head(2).equals(left)
    assert both.tail(3).equals(right)
    assert both.schema == schema
    assert copysign(1, both["f"][1]) == -1
    assert both.vstack(right).tail(6).equals(right.vstack(right))
    assert both.head(0).vstack(left).vstack(right.head(0)).equals(left)
    assert list(both.to_dicts()[0]) == list(schema)
    assert both.to_dicts()[2] == {"c": "猫", "s": "猫", "f": 0.0, "b": None, "i": 1}
    for incompatible in (
        left.select(*reversed(left.columns)),
        left.drop("f"),
        left.with_columns(col("c").cast(String)),
        left.rename({"b": "flag"}),
    ):
        with pytest.raises(SchemaError):
            left.vstack(incompatible)
        assert not left.equals(incompatible)
    assert DataFrame({"f": [-0.0, nan]}).equals(DataFrame({"f": [0.0, float("nan")]}))


def canonical(value: object) -> object:
    return "NaN" if isinstance(value, float) and isnan(value) else value


@settings(derandomize=True, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.one_of(st.none(), st.sampled_from([nan, -0.0, 0.0, 1.0, -2.0])),
            st.one_of(st.none(), st.sampled_from(["", "猫", "a"])),
            st.one_of(st.none(), st.booleans()),
        ),
        max_size=45,
    ),
    st.integers(0, 50),
)
def test_row_operations_against_python(
    records: list[tuple[float | None, str | None, bool | None]], n: int
) -> None:
    rows = [(*row, i) for i, row in enumerate(records)]
    names = ["f", "c", "keep", "id"]
    df = DataFrame(
        {name: [row[j] for row in rows] for j, name in enumerate(names)},
        schema=dict(zip(names, (Float64, Categorical, Boolean, Int64), strict=True)),
    )
    seen = []
    unique = []
    for row in rows:
        key = (canonical(row[0]), row[1])
        if key not in seen:
            seen.append(key)
            unique.append(row)
    ordered = sorted(rows, key=lambda r: (r[1] is not None, r[1]))
    ordered = sorted(
        ordered,
        key=lambda r: (
            r[0] is not None,
            r[0] is not None and isnan(r[0]),
            0 if r[0] is None or isnan(r[0]) else r[0],
        ),
        reverse=True,
    )
    for frame, expected in (
        (df.filter(col("keep")), [r for r in rows if r[2] is True]),
        (df.sort(["f", "c"], descending=[True, False], nulls_last=[True, False]), ordered),
        (df.unique(["f", "c"]), unique),
        (df.drop_nulls(), [r for r in rows if None not in r]),
        (df.drop_nulls(["c"]), [r for r in rows if r[1] is not None]),
        (df.head(n), rows[:n]),
        (df.tail(n), rows[len(rows) - min(n, len(rows)) :]),
        (df.vstack(df), rows + rows),
        (df, rows),
    ):
        assert frame.columns == names
        assert frame.schema == df.schema
        assert frame.shape == (len(expected), len(names))
        assert [tuple(map(canonical, row)) for row in frame.rows()] == [
            tuple(map(canonical, row)) for row in expected
        ]
        for name in names:
            column = frame[name]._column
            assert len(column) == frame.height
            column.check_invariants()
    expr = col("id") * 2 + 1
    reference = evaluate_expr(expr, {"id": Column(range(len(rows)), Int64)}, height=len(rows))
    assert df.select(expr)["id"].to_list() == reference.to_list()
    copy = df.head(n).vstack(df.tail(max(0, len(rows) - n)))
    assert df.equals(copy) and copy.equals(df)


def test_polars_eager_pipeline() -> None:
    data = {"k": [2, None, 1, 2, 2], "v": [5, 9, 4, 6, 7], "flag": [True, True, None, False, True]}
    df = DataFrame(data)
    ref = pl.DataFrame(data)
    result = df.with_columns((col("v") * 2).alias("v")).filter(col("flag")).sort("k").unique(["k"])
    expected = (
        ref.with_columns((pl.col("v") * 2).alias("v"))
        .filter(pl.col("flag"))
        .sort("k", maintain_order=True)
        .unique(subset=["k"], keep="first", maintain_order=True)
    )
    assert result.rows() == expected.rows()


def test_unique_all_columns_and_boolean_filter() -> None:
    df = DataFrame(
        {"age": [18, 18, 22, None, 16, 22], "active": [True, True, None, True, True, False]}
    )
    assert df.unique().rows() == [(18, True), (22, None), (None, True), (16, True), (22, False)]
    assert df.filter((col("age") >= 18) & col("active")).rows() == [(18, True), (18, True)]
    assert df.filter(lit(False)).columns == ["age", "active"]
