from collections.abc import Callable
from decimal import localcontext
from itertools import accumulate
from math import copysign, inf, isclose, isnan, nan, sqrt, ulp
from statistics import variance
from sys import float_info

import duckdb
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from miniframe import (
    Boolean,
    Categorical,
    ColumnNotFoundError,
    ComputeError,
    DataFrame,
    DType,
    Expr,
    Float64,
    Int64,
    SchemaError,
    ShapeError,
    String,
    col,
    lit,
)
from miniframe.column import Column
from miniframe.dtype import INT64_MAX, INT64_MIN


def test_all_reductions_and_schema() -> None:
    df = DataFrame({"k": ["c", "b", "c", None, "c", "c"], "v": [None, None, 2, 4, 6, None]})
    v = col("v")
    result = df.group_by("k").agg(
        v.sum(),
        v.mean().alias("avg"),
        v.min().alias("min"),
        v.max().alias("max"),
        v.len().alias("len"),
        v.count().alias("count"),
        v.n_unique().alias("distinct"),
        v.var().alias("var"),
        v.std().alias("std"),
        v.first().alias("first"),
        v.last().alias("last"),
    )
    assert list(result.schema.items()) == [
        ("k", String),
        ("v", Int64),
        ("avg", Float64),
        ("min", Int64),
        ("max", Int64),
        ("len", Int64),
        ("count", Int64),
        ("distinct", Int64),
        ("var", Float64),
        ("std", Float64),
        ("first", Int64),
        ("last", Int64),
    ]
    assert result.rows() == [
        ("c", 8, 4.0, 2, 6, 4, 2, 3, 8.0, sqrt(8), None, None),
        ("b", 0, None, None, None, 1, 0, 1, None, None, None, None),
        (None, 4, 4.0, 4, 4, 1, 1, 1, None, None, 4, 4),
    ]
    for name in result.columns:
        result[name]._column.check_invariants()


def test_key_and_operand_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    df = DataFrame({"p": [4, None, 2, 8], "q": [2, 5, 1, 3]})
    evaluate = DataFrame._evaluate
    calls: list[int] = []

    def trace(frame: DataFrame, expr: Expr) -> Column:
        assert frame is df
        calls.append(id(expr))
        return evaluate(frame, expr)

    monkeypatch.setattr(DataFrame, "_evaluate", trace)
    key = (col("p") >= 4).alias("big")
    revenue = col("p") * col("q")
    margin = col("p") - col("q")
    grouped = df.group_by(key)
    assert calls == []
    result = grouped.agg(revenue.sum().alias("old").alias("total"), margin.mean().alias("margin"))
    assert calls == list(map(id, (key, revenue, margin)))
    assert result.columns == ["big", "total", "margin"]
    assert result.rows() == [(True, 32, 3.5), (None, 0, None), (False, 2, 1.0)]
    assert grouped.agg(col("q").sum().sum()).rows() == [(True, 22), (None, 11), (False, 11)]
    assert df.rows() == [(4, 2), (None, 5), (2, 1), (8, 3)]


def test_composite_keys_and_representatives() -> None:
    df = DataFrame(
        {
            "c": ["Я", None, "Я", "a", "Я", None, "a", "Я"],
            "f": [nan, None, float("nan"), -0.0, -0.0, nan, 0.0, 0.0],
        },
        schema={"c": Categorical},
    )
    result = df.group_by("f", "c").agg(col("f").len().alias("n"))
    expected = DataFrame(
        {"f": [nan, None, -0.0, -0.0, nan], "c": ["Я", None, "a", "Я", None], "n": [2, 1, 2, 2, 1]},
        schema={"c": Categorical},
    )
    assert result.equals(expected)
    assert copysign(1, result["f"][2]) == -1
    assert df.group_by("f", "c").agg().equals(expected.drop("n"))
    assert df.group_by(lit(True)).agg(col("f").n_unique()).rows() == [(True, 3)]


@pytest.mark.parametrize("dtype", [String, Categorical])
def test_ordered_text_and_boolean(dtype: DType) -> None:
    df = DataFrame(
        {
            "k": [2, 1, 2, 1, 2],
            "s": ["🙂", None, "я", "", "я"],
            "b": [False, None, True, False, None],
        },
        schema={"s": dtype},
    )
    s, b = col("s"), col("b")
    result = df.group_by("k").agg(
        s.first(),
        s.last().alias("end"),
        s.min().alias("low"),
        s.max().alias("high"),
        s.n_unique().alias("n"),
        b.first(),
        b.last().alias("last"),
        b.count().alias("valid"),
        b.len().alias("rows"),
    )
    assert result.rows() == [
        (2, "🙂", "я", "я", "🙂", 2, False, None, 2, 3),
        (1, None, "", "", "", 2, None, False, 1, 2),
    ]
    assert [result.schema[n] for n in ("s", "end", "low", "high")] == [dtype] * 4


def test_empty_and_scalar_inputs() -> None:
    v = col("v")
    df = DataFrame(schema={"k": String, "v": Float64})
    result = df.group_by("k").agg(v.sum(), v.count().alias("n"), v.var().alias("var"))
    assert result.shape == (0, 4)
    assert list(result.schema.items()) == [
        ("k", String),
        ("v", Float64),
        ("n", Int64),
        ("var", Float64),
    ]
    bare = DataFrame({"id": range(4)}).select()
    grouped = bare.group_by(lit(None).cast(Categorical).alias("key"))
    f = lit(None).cast(Float64)
    result = grouped.agg(f.sum().alias("sum"), f.mean().alias("mean"), f.len().alias("len"))
    assert result.rows() == [(None, 0.0, None, 4)]
    assert result.schema == {"key": Categorical, "sum": Float64, "mean": Float64, "len": Int64}
    assert df.group_by("k").agg().equals(df.select("k"))


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda d: d.group_by(), ValueError),
        (lambda d: d.group_by(1), TypeError),
        (lambda d: d.group_by("missing").agg(), ColumnNotFoundError),
        (lambda d: d.group_by("k", "k").agg(), SchemaError),
        (lambda d: d.group_by("k").agg(col("k").first()), SchemaError),
        (lambda d: d.group_by("k").agg(col("b").count(), col("b").len()), SchemaError),
        (lambda d: d.group_by("k").agg("b"), TypeError),
        (lambda d: d.group_by("k").agg((col("k") * 2).alias("x")), ShapeError),
        (lambda d: d.group_by("k").agg(col("k").sum() + 1), ShapeError),
        (lambda d: d.group_by("k").agg(col("missing").mean()), ColumnNotFoundError),
        (lambda d: d.group_by("k").agg(col("b").sum()), SchemaError),
        (lambda d: d.group_by("k").agg(col("b").max()), SchemaError),
        (lambda d: d.group_by(lit(None)).agg(), SchemaError),
        (lambda d: d.group_by("k").agg(lit(None).first()), SchemaError),
    ],
)
def test_errors_even_without_rows(
    call: Callable[[DataFrame], object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        call(DataFrame(schema={"k": Int64, "b": Boolean}))


@pytest.mark.parametrize(("limit", "step"), [(INT64_MAX, 1), (INT64_MIN, -1)])
def test_checked_prefix_and_unbounded_mean(limit: int, step: int) -> None:
    df = DataFrame({"k": [0, 1, 0, 0], "v": [limit, -step, step, -step]})
    grouped = df.group_by("k")
    with pytest.raises(ComputeError, match="^Int64 overflow$"):
        grouped.agg(col("v").sum())
    assert grouped.agg(col("v").mean()).rows() == [(0, limit / 3), (1, -step)]


def test_precision_and_nonfinite_values() -> None:
    largest = float_info.max
    samples = [
        [largest, largest],
        [largest, 1.0, -largest],
        [nan],
        [inf, -inf],
        [nan, 2.0],
        [-1e200, 1e200],
        [1e15, 1e15 + 0.25, 1e15 + 0.5],
        [inf, inf],
    ]
    data = [(k, xs[i]) for i in range(3) for k, xs in enumerate(samples) if i < len(xs)]
    df = DataFrame({"k": [k for k, _ in data], "v": [v for _, v in data]})
    v = col("v")
    with localcontext(prec=3):
        result = df.group_by("k").agg(
            v.sum(),
            v.mean().alias("avg"),
            v.var().alias("var"),
            v.std().alias("std"),
            v.min().alias("lo"),
            v.max().alias("hi"),
        )
    assert result["v"].to_list()[:2] == [inf, 1.0]
    assert result["avg"].to_list()[:2] == [largest, 1 / 3]
    assert result["var"].to_list()[:3] == [0.0, inf, None]
    assert result["std"][2] is None
    for i in (3, 4, 7):
        assert isnan(result["var"][i]) and isnan(result["std"][i])
    for i in (2, 3, 4):
        assert isnan(result["v"][i]) and isnan(result["avg"][i])
    assert result["lo"][4] == 2.0 and isnan(result["hi"][4])
    assert result["var"][5] == inf
    assert isclose(result["std"][5], sqrt(2) * 1e200, rel_tol=4 * ulp(1.0))
    assert result["var"][6] == 0.0625 and result["std"][6] == 0.25
    assert result["v"][7] == result["avg"][7] == inf


def test_many_interleaved_groups() -> None:
    n = 1009
    df = DataFrame({"k": list(range(n)) * 13, "v": range(13 * n)})
    result = df.group_by("k").agg(col("v").sum(), col("v").len().alias("n"))
    assert result.rows() == [(k, 13 * k + 78 * n, 13) for k in range(n)]
    assert df.head(n).group_by("k").agg(col("v").last()).equals(df.head(n))


def test_polars_and_duckdb() -> None:
    data = {"k": ["b", None, "a", "b", None, "a"], "v": [2, 4, None, 6, 4, None]}
    v = col("v")
    actual = (
        DataFrame(data)
        .group_by("k")
        .agg(v.sum(), v.n_unique().alias("n"), v.var().alias("var"))
        .rows()
    )
    p = pl.col("v")
    assert (
        actual
        == pl.DataFrame(data)
        .group_by("k", maintain_order=True)
        .agg(p.sum(), p.n_unique().alias("n"), p.var(ddof=1).alias("var"))
        .rows()
    )
    with duckdb.connect() as db:
        db.execute("create table samples (i integer, k varchar, v bigint)")
        db.executemany(
            "insert into samples values (?, ?, ?)",
            [(i, k, v) for i, (k, v) in enumerate(zip(data["k"], data["v"], strict=True))],
        )
        assert (
            actual
            == db.execute("""
            select k, coalesce(sum(v), 0),
                   count(distinct v) + max(case when v is null then 1 else 0 end),
                   var_samp(v)
            from samples group by k order by min(i)
        """).fetchall()
        )


type Row = tuple[float | None, bool | None, str | None, int | None]


def key_of(row: Row, width: int) -> tuple[object, ...]:
    return tuple("NaN" if isinstance(x, float) and isnan(x) else x for x in row[:width])


@settings(derandomize=True, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.none() | st.sampled_from([nan, -0.0, 0.0, -2.0, 4.0]),
            st.none() | st.booleans(),
            st.none() | st.sampled_from(["", "猫", "a"]),
            st.none() | st.integers(-100, 100),
        ),
        max_size=40,
    ),
    st.integers(1, 3),
)
def test_partition_properties(rows: list[Row], width: int) -> None:
    names = ["f", "b", "s", "v"]
    schema = dict(zip(names, (Float64, Boolean, String, Int64), strict=True))
    v = col("v")
    summaries = []
    for records in (rows, rows[::-1]):
        data = {name: [r[i] for r in records] for i, name in enumerate(names)}
        df = DataFrame(data, schema=schema)
        result = df.group_by(*names[:width]).agg(
            v.len().alias("size"),
            v.count().alias("count"),
            v.sum(),
            v.n_unique().alias("distinct"),
            v.first().alias("first"),
            v.last().alias("last"),
            v.var().alias("var"),
        )
        keys = list(dict.fromkeys(key_of(r, width) for r in records))
        assert result.height == len(keys)
        assert result["size"].sum() == df.height
        summary = {}
        for index, (key, output) in enumerate(zip(keys, result.rows(), strict=True)):
            assert key_of(output, width) == key
            values = [r[-1] for r in records if key_of(r, width) == key]
            valid = [x for x in values if x is not None]
            assert output[width:-1] == (
                len(values),
                len(valid),
                sum(valid),
                len(set(values)),
                values[0],
                values[-1],
            )
            if len(valid) < 2:
                assert result["var"][index] is None
            else:
                assert isclose(result["var"][index], variance(valid), rel_tol=4 * ulp(1.0))
            summary[key] = output[width : width + 4]
        summaries.append(summary)
        assert df.equals(DataFrame(data, schema=schema))
        for frame in (df, result):
            for name in frame.columns:
                frame[name]._column.check_invariants()
    assert summaries[0] == summaries[1]


@settings(derandomize=True)
@given(
    st.lists(
        st.tuples(st.booleans(), st.none() | st.sampled_from([INT64_MIN, -1, 0, 1, INT64_MAX])),
        max_size=25,
    )
)
def test_checked_sums(rows: list[tuple[bool, int | None]]) -> None:
    df = DataFrame(
        {"k": [k for k, _ in rows], "v": [v for _, v in rows]}, schema={"k": Boolean, "v": Int64}
    )
    expected = []
    for key in dict.fromkeys(k for k, _ in rows):
        values = [v for k, v in rows if k == key and v is not None]
        if any(x < INT64_MIN or x > INT64_MAX for x in accumulate(values)):
            with pytest.raises(ComputeError):
                df.group_by("k").agg(col("v").sum())
            return
        expected.append((key, sum(values)))
    assert df.group_by("k").agg(col("v").sum()).rows() == expected
