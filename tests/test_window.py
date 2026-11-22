from collections.abc import Callable
from dataclasses import FrozenInstanceError
from fractions import Fraction
from math import copysign, inf, isnan, nan
from statistics import stdev, variance
from sys import float_info

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
from miniframe.column import Column, _Scalar
from miniframe.dtype import INT64_MAX, INT64_MIN
from miniframe.expr import is_root_aggregation, output_name
from miniframe.ops import window

METHODS = ("average", "min", "max", "dense", "ordinal")


def order(value: _Scalar) -> tuple[bool, _Scalar]:
    if isinstance(value, float) and isnan(value):
        return True, 0
    return False, value


def same(values: list[_Scalar], expected: list[_Scalar]) -> None:
    assert list(map(order, values)) == list(map(order, expected))


def check(df: DataFrame, expr: Expr, dtype: DType, expected: list[_Scalar]) -> None:
    result = df.select(expr)
    assert result.shape == (df.height, 1)
    series = result[result.columns[0]]
    assert series.dtype is dtype
    series._column.check_invariants()
    same(series.to_list(), expected)


def reference(values: list[_Scalar], op: str, minimum: int) -> _Scalar:
    valid = [v for v in values if v is not None]
    if len(valid) < minimum:
        return None
    if op in ("min", "max"):
        return (min if op == "min" else max)(valid, key=order, default=None)
    if not valid:
        return None if op == "mean" else 0
    if any(isinstance(v, float) and isnan(v) for v in valid) or (inf in valid and -inf in valid):
        return nan
    if inf in valid:
        return inf
    if -inf in valid:
        return -inf
    total = sum(map(Fraction, valid))
    if op == "mean":
        total /= len(valid)
    try:
        return float(total) if op == "mean" or isinstance(valid[0], float) else int(total)
    except OverflowError:
        return -inf if total < 0 else inf


def ranks(values: list[_Scalar], method: str, descending: bool) -> list[_Scalar]:
    valid = [order(v) for v in values if v is not None]
    result = []
    for i, value in enumerate(values):
        if value is None:
            result.append(None)
            continue
        key = order(value)
        earlier = [k for k in valid if (k > key if descending else k < key)]
        low, high = len(earlier) + 1, len(earlier) + valid.count(key)
        by_method = {
            "average": (low + high) / 2,
            "min": low,
            "max": high,
            "dense": len(set(earlier)) + 1,
            "ordinal": low + sum(order(v) == key for v in values[:i] if v is not None),
        }
        result.append(by_method[method])
    return result


@pytest.mark.parametrize(
    "values", [[], [None] * 5, [1, None, 3, 4], [None, 4, 4, 2, None, -1, None]]
)
@pytest.mark.parametrize(
    ("size", "minimum"), [(1, None), (2, None), (2, 1), (3, 0), (3, 3), (10, None)]
)
def test_trailing_frames(values: list[int | None], size: int, minimum: int | None) -> None:
    df = DataFrame({"x": values}, schema={"x": Int64})
    for op in ("sum", "mean", "min", "max"):
        expected = [
            reference(
                values[max(0, i - size + 1) : i + 1], op, size if minimum is None else minimum
            )
            for i in range(len(values))
        ]
        check(
            df,
            getattr(col("x"), "rolling_" + op)(size, minimum),
            Float64 if op == "mean" else Int64,
            expected,
        )
    for op in ("sum", "min", "max"):
        check(
            df,
            getattr(col("x"), "cum_" + op)(),
            Int64,
            [
                None if v is None else reference(values[: i + 1], op, 1)
                for i, v in enumerate(values)
            ],
        )


@pytest.mark.parametrize(
    "values",
    [
        [float_info.max, float_info.max, -float_info.max, 1.0, float_info.max, -float_info.max],
        [nan, 1.0, None, 2.0, inf, -inf, 3.0, 0.0, -0.0, -inf, 2.0, 2.0, 4.0],
        [1e300, 1.0, -1e300, 5e-324, -5e-324, 0.0],
    ],
)
def test_float_extremes(values: list[float | None]) -> None:
    df = DataFrame({"x": values})
    for op in ("sum", "mean", "min", "max"):
        check(
            df,
            getattr(col("x"), "rolling_" + op)(3, 0),
            Float64,
            [reference(values[max(0, i - 2) : i + 1], op, 0) for i in range(len(values))],
        )
    for op in ("sum", "min", "max"):
        check(
            df,
            getattr(col("x"), "cum_" + op)(),
            Float64,
            [
                None if v is None else reference(values[: i + 1], op, 1)
                for i, v in enumerate(values)
            ],
        )


def test_signed_zero_extrema() -> None:
    df = DataFrame({"x": [-0.0, 0.0, 0.0]})
    for expr in (col("x").rolling_min(2, 1), col("x").rolling_max(2, 1)):
        assert [copysign(1, x) for x in df.select(expr)["x"].to_list()] == [-1, -1, 1]


@pytest.mark.parametrize(("limit", "step"), [(INT64_MAX, 1), (INT64_MIN, -1)])
def test_int_overflow_only_on_emission(limit: int, step: int) -> None:
    df = DataFrame({"x": [limit, step, -step]})
    for expr in (col("x").rolling_sum(2), col("x").cum_sum()):
        with pytest.raises(ComputeError, match="^Int64 overflow$"):
            df.select(expr)
    check(df, col("x").rolling_sum(3), Int64, [None, None, limit])
    check(df, col("x").rolling_mean(2), Float64, [None, (limit + step) / 2, 0.0])
    check(df, col("x").rolling_sum(1), Int64, [limit, step, -step])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (Float64, [nan, 0.0, float("nan"), -0.0, inf, -inf, None]),
        (Boolean, [True, False, True, None, False]),
        (Categorical, ["Я", "", None, "a", "a"]),
        (String, ["z", None, "y", "a"]),
        (Int64, [10, 20, 20, 40]),
        (Int64, [None] * 4),
        (Float64, []),
    ],
)
def test_tie_blocks(
    dtype: DType, values: list[_Scalar], method: window.RankMethod, descending: bool
) -> None:
    check(
        DataFrame({"x": values}, schema={"x": dtype}),
        col("x").rank(method, descending),
        Float64 if method == "average" else Int64,
        ranks(values, method, descending),
    )


def test_expr_composition_and_scalar_broadcast() -> None:
    x, y = col("x"), col("y")
    df = DataFrame({"x": [1, 3, None, 4], "y": [2, 1, 2, 2]})
    expr = (x + y).rolling_mean(3, 1)
    assert output_name(expr) == "x"
    check(df, expr, Float64, [3.0, 3.5, 3.5, 5.0])
    check(df, (x > 2).cast(Int64).rolling_sum(2, 1), Int64, [0, 1, 1, 1])
    result = df.with_columns((x.cum_sum() * 2).alias("run"))
    assert result.rows() == [(1, 2, 2), (3, 1, 8), (None, 2, None), (4, 2, 16)]
    check(df.select(), lit(3).shift(-1), Int64, [3, 3, 3, None])
    check(df, lit(None).cast(Float64).rolling_sum(2, 0), Float64, [0.0] * 4)
    check(df, lit(4).rank(), Float64, [2.5] * 4)
    for node in (expr, x.cum_sum(), x.rank(), x.shift(1), x.sum().over("y")):
        assert not is_root_aggregation(node)
        with pytest.raises(TypeError):
            bool(node)
        with pytest.raises(FrozenInstanceError):
            node.operand = x
        assert not hasattr(node, "__dict__")
    assert df.rows() == [(1, 2), (3, 1), (None, 2), (4, 2)]


@pytest.mark.parametrize(
    ("build", "error"),
    [
        (lambda: col("x").rolling_sum(True), TypeError),
        (lambda: col("x").rolling_min(3, False), TypeError),
        (lambda: col("x").rolling_max(0), ValueError),
        (lambda: col("x").rolling_mean(-1), ValueError),
        (lambda: col("x").rolling_sum(2, -1), ValueError),
        (lambda: col("x").rolling_sum(2, 3), ValueError),
        (lambda: col("x").rank("random"), ValueError),
        (lambda: col("x").rank(descending=1), TypeError),
        (lambda: col("x").shift(False), TypeError),
        (lambda: col("x").over("k"), ShapeError),
        (lambda: col("x").sum().over(), ValueError),
        (lambda: col("x").sum().over(2), TypeError),
    ],
)
def test_invalid_parameters(build: Callable[[], Expr], error: type[Exception]) -> None:
    with pytest.raises(error):
        build()


def test_context_errors_and_types() -> None:
    x = col("x")
    df = DataFrame(schema={"x": Boolean})
    for expr in (
        x.rolling_max(1),
        x.rolling_min(1),
        x.rolling_sum(1),
        x.rolling_mean(1),
        x.cum_sum(),
        x.cum_min(),
        x.cum_max(),
        lit(None).shift(0),
    ):
        with pytest.raises(SchemaError):
            df.select(expr)
    for expr in (
        (x.len() + 1).sum().over("x"),
        x.first().over("x").count().over("x"),
        x.len().over(x.first()),
    ):
        with pytest.raises(ShapeError):
            df.select(expr)
    with pytest.raises(ColumnNotFoundError):
        df.select(x.len().over("missing"))
    with pytest.raises(ShapeError):
        df.group_by("x").agg(x.rank().alias("rank"))
    with pytest.raises(ShapeError):
        window.over(Column([], Int64), "sum", [Column([1], Int64)])


@pytest.mark.parametrize(
    "expr",
    [
        col("x").sum().alias("total").over("city"),
        col("x").sum().over("city").alias("total"),
    ],
    ids=["alias-before-over", "alias-after-over"],
)
def test_over_preserves_alias(expr: Expr) -> None:
    df = DataFrame({"city": ["Kyiv", "Lviv", "Kyiv"], "x": [1, 7, 2]})
    assert df.select(expr).equals(DataFrame({"total": [3, 7, 3]}))
    assert df.with_columns(expr).equals(
        DataFrame({"city": ["Kyiv", "Lviv", "Kyiv"], "x": [1, 7, 2], "total": [3, 7, 3]})
    )


def test_over_all_reductions() -> None:
    keys = [nan, None, nan, -0.0, None, 0.0, nan, nan]
    labels = ["b", "a", "b", "x", "a", "x", "a", "b"]
    values = [2, None, 4, 1, None, 9, 6, None]
    df = DataFrame({"k": keys, "c": labels, "x": values}, schema={"c": Categorical})
    operand = col("x") * 2
    partitions = [(order(k), c) for k, c in zip(keys, labels, strict=True)]
    for op in (
        "sum",
        "min",
        "max",
        "mean",
        "std",
        "var",
        "first",
        "last",
        "n_unique",
        "count",
        "len",
    ):
        expected = []
        for key in partitions:
            observations = [
                2 * v if v is not None else None
                for k, v in zip(partitions, values, strict=True)
                if k == key
            ]
            valid = [v for v in observations if v is not None]
            reductions = {
                "sum": sum(valid),
                "min": min(valid, default=None),
                "max": max(valid, default=None),
                "mean": sum(valid) / len(valid) if valid else None,
                "var": variance(valid) if len(valid) > 1 else None,
                "std": stdev(valid) if len(valid) > 1 else None,
                "first": observations[0],
                "last": observations[-1],
                "count": len(valid),
                "len": len(observations),
                "n_unique": len(set(observations)),
            }
            expected.append(reductions[op])
        expr = getattr(operand, op)().alias("inside").over(col("k"), "c").alias("out")
        dtype = Float64 if op in ("mean", "std", "var") else Int64
        check(df, expr, dtype, expected)
        check(df.head(0), expr, dtype, [])
    check(df, lit(1).sum().over(col("x").is_null()), Int64, [5, 3, 5, 5, 3, 5, 5, 3])
    rebuilt = DataFrame({"k": keys, "c": labels, "x": values}, schema={"c": Categorical})
    assert df.equals(rebuilt)


VALUES = {
    Int64: st.integers(-20, 20),
    Float64: st.one_of(
        st.floats(allow_nan=False, allow_infinity=False),
        st.sampled_from([nan, inf, -inf, 0.0, -0.0]),
    ),
    Boolean: st.booleans(),
    String: st.sampled_from(["", "Lviv", "猫"]),
    Categorical: st.sampled_from(["b", "a", "猫"]),
}


@pytest.mark.parametrize("dtype", list(DType))
@settings(max_examples=45, deadline=None, derandomize=True)
@given(data=st.data(), size=st.integers(1, 12), offset=st.integers(-20, 20))
def test_properties(dtype: DType, data: st.DataObject, size: int, offset: int) -> None:
    values = data.draw(st.lists(st.none() | VALUES[dtype], max_size=18))
    minimum = data.draw(st.integers(0, size))
    df = DataFrame({"x": values, "id": range(len(values))}, schema={"x": dtype, "id": Int64})
    x = col("x")
    if dtype is not Boolean:
        ops = ("min", "max", "sum", "mean") if dtype.is_numeric else ("min", "max")
        for op in ops:
            check(
                df,
                getattr(x, "rolling_" + op)(size, minimum),
                Float64 if op == "mean" else dtype,
                [
                    reference(values[max(0, i - size + 1) : i + 1], op, minimum)
                    for i in range(len(values))
                ],
            )
            if op != "mean":
                check(
                    df,
                    getattr(x, "cum_" + op)(),
                    dtype,
                    [
                        None if v is None else reference(values[: i + 1], op, 1)
                        for i, v in enumerate(values)
                    ],
                )
    descending = data.draw(st.booleans())
    for method in METHODS:
        check(
            df,
            x.rank(method, descending),
            Float64 if method == "average" else Int64,
            ranks(values, method, descending),
        )
    indices = range(len(values))
    check(
        df,
        x.shift(offset),
        dtype,
        [values[i - offset] if 0 <= i - offset < len(values) else None for i in indices],
    )
    check(
        df,
        x.shift(offset).shift(-offset),
        dtype,
        [values[i] if 0 <= i + offset < len(values) else None for i in indices],
    )
    check(
        df,
        col("id").sum().over(x),
        Int64,
        [sum(j for j in indices if order(values[i]) == order(values[j])) for i in indices],
    )
    same(df["x"].to_list(), values)


def test_polars_differential() -> None:
    data = {"x": [2, None, 4, 2, 1, 7, None, 4], "k": ["a", "b", None, "b", "a", "a", "b", None]}
    df = DataFrame(data)
    oracle = pl.DataFrame(data)
    x, p = col("x"), pl.col("x")
    pairs = [(x.shift(-2), p.shift(-2)), (x.sum().over("k"), p.sum().over("k"))]
    for op in ("sum", "mean", "min", "max"):
        pairs.append(
            (getattr(x, "rolling_" + op)(3, 1), getattr(p, "rolling_" + op)(3, min_samples=1))
        )
        if op != "mean":
            pairs.append((getattr(x, "cum_" + op)(), getattr(p, "cum_" + op)()))
    pairs.extend((x.rank(method, True), p.rank(method, descending=True)) for method in METHODS)
    for expr, ref in pairs:
        result = df.select(expr)
        assert result["x"].to_list() == oracle.select(ref).to_series().to_list()
        result["x"]._column.check_invariants()


def test_large_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    length, size = 6000, 1800
    values = [None if i % 5 == 0 else (i % 113) - 56 for i in range(length)]
    keys = [i % 137 for i in range(length)]
    df = DataFrame({"x": values, "k": keys})
    x = col("x")
    reads = 0
    original_get = Column.get

    def counted(column: Column, i: int) -> _Scalar:
        nonlocal reads
        reads += 1
        return original_get(column, i)

    monkeypatch.setattr(Column, "get", counted)
    for op in ("min", "max", "sum", "mean"):
        reads = 0
        result = df.select(getattr(x, "rolling_" + op)(size, 1))["x"]
        assert reads <= 5 * length
        for i in (1, size - 1, size, length - 1):
            assert result[i] == reference(values[max(0, i - size + 1) : i + 1], op, 1)
        result._column.check_invariants()
    grouped = df.select(x.sum().over("k"))["x"]
    ranked = df.select(x.rank("dense"))["x"]
    prefix = df.select(x.cum_sum())["x"]
    distinct = sorted({v for v in values if v is not None})
    for i in (0, 2, 57, 2048, length - 1):
        assert grouped[i] == sum(
            v for k, v in zip(keys, values, strict=True) if k == keys[i] and v is not None
        )
        if values[i] is None:
            assert ranked[i] is prefix[i] is None
        else:
            assert ranked[i] == distinct.index(values[i]) + 1
            assert prefix[i] == sum(v for v in values[: i + 1] if v is not None)
    for series in (grouped, ranked, prefix):
        series._column.check_invariants()


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (Boolean, [True, None, False]),
        (String, ["Ї", None, ""]),
        (Categorical, ["z", None, "a"]),
        (Int64, [3, None, 1]),
        (Float64, [nan, None, -0.0]),
    ],
)
def test_shift_boundaries(dtype: DType, values: list[_Scalar]) -> None:
    df = DataFrame({"x": values}, schema={"x": dtype})
    for offset in (-(10**100), -3, -1, 0, 1, 3, 10**100):
        check(
            df,
            col("x").shift(offset),
            dtype,
            [
                values[i - offset] if 0 <= i - offset < len(values) else None
                for i in range(len(values))
            ],
        )
        check(df.head(0), col("x").shift(offset), dtype, [])


@pytest.mark.parametrize("dtype", [String, Categorical])
def test_text_extrema_expire_in_logical_order(dtype: DType) -> None:
    df = DataFrame({"x": ["я", "b", "b", None, "a", "猫", None]}, schema={"x": dtype})
    x = col("x")
    check(df, x.rolling_min(2, 1), dtype, ["я", "b", "b", "b", "a", "a", "猫"])
    check(df, x.rolling_max(2, 1), dtype, ["я", "я", "b", "b", "a", "猫", "猫"])
    check(df, x.cum_min(), dtype, ["я", "b", "b", None, "a", "a", None])
    check(df, x.cum_max(), dtype, ["я", "я", "я", None, "я", "猫", None])
