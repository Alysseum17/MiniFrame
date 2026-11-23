import operator
from collections import defaultdict
from collections.abc import Callable, Mapping
from math import inf, isnan, nan, sqrt

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import tree_cases

from miniframe import (
    Boolean,
    Categorical,
    ColumnNotFoundError,
    ComputeError,
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
from miniframe.eval import Scalar, Value, broadcast_scalar, evaluate_expr
from miniframe.kernels import arithmetic, compare, logic


def assert_column(result: Value, dtype: DType, values: list[_Scalar]) -> None:
    assert isinstance(result, Column)
    result.check_invariants()
    assert result.dtype is dtype
    assert len(result) == len(values)
    assert result.null_count == values.count(None)
    assert result.to_list() == values


@pytest.mark.parametrize(
    ("expr", "dtype", "expected"),
    [
        (col("a") + col("b"), Int64, [11, None, 33]),
        ((col("a") + 1) * 2, Int64, [4, None, 8]),
        (10 - col("a"), Int64, [9, None, 7]),
        (10 / col("a"), Float64, [10.0, None, 10 / 3]),
        (col("a") / col("b"), Float64, [0.1, None, 0.1]),
        (-(col("a") * 3), Int64, [-3, None, -9]),
        ((col("a") - 5) / (col("b") + 1), Float64, [-4 / 11, None, -2 / 31]),
        (col("a").fill_null(0), Int64, [1, 0, 3]),
        ((col("a") + col("b")).fill_null(100), Int64, [11, 100, 33]),
        (col("a").is_null(), Boolean, [False, True, False]),
        (col("a").is_not_null(), Boolean, [True, False, True]),
        (col("a") > 2, Boolean, [False, None, True]),
        ((col("a") > 0) & (col("b") < 25), Boolean, [True, None, False]),
        (col("a") + col("b").sum(), Int64, [61, None, 63]),
        (col("a") - col("a"), Int64, [0, None, 0]),
        (col("a").alias("inner").cast(Float64).alias("outer"), Float64, [1.0, None, 3.0]),
    ],
)
def test_numeric_context(expr: Expr, dtype: DType, expected: list[_Scalar]) -> None:
    columns = {"a": Column([1, None, 3], Int64), "b": Column([10, 20, 30], Int64)}
    assert_column(evaluate_expr(expr, columns, height=3), dtype, expected)
    assert_column(columns["a"], Int64, [1, None, 3])
    assert_column(columns["b"], Int64, [10, 20, 30])


@pytest.mark.parametrize("height", [0, 3])
@pytest.mark.parametrize(
    ("expr", "dtype", "expected"),
    [
        (lit(10), Int64, 10),
        (lit(None), None, None),
        (lit(2) + lit(3), Int64, 5),
        (lit(10) / 4, Float64, 2.5),
        (lit(1) / -0.0, Float64, -inf),
        (lit("Kyiv") != "Lviv", Boolean, True),
        (lit(2.0) == lit(2), Boolean, True),
        ((lit(None) == lit(None)).alias("unknown"), Boolean, None),
        (lit(None) >= 1, Boolean, None),
        (lit(None) + 1, Int64, None),
        (lit(2) / None, Float64, None),
        (lit(None) & False, Boolean, False),
        (True | lit(None), Boolean, True),
        (lit(None) | lit(None), Boolean, None),
        (~lit(None), Boolean, None),
        (-lit(3), Int64, -3),
        (lit(None).cast(Categorical), Categorical, None),
        (lit("Kyiv").cast(Categorical), Categorical, "Kyiv"),
        (lit("12").cast(Int64), Int64, 12),
        (lit(None).fill_null(False), Boolean, False),
        (lit(None).fill_null(None), None, None),
        (lit(5).fill_null(0), Int64, 5),
        (lit(None).is_null(), Boolean, True),
        (lit(False).is_not_null(), Boolean, True),
    ],
)
def test_scalar_expressions(
    height: int, expr: Expr, dtype: DType | None, expected: _Scalar
) -> None:
    result = evaluate_expr(expr, {}, height=height)
    assert result == Scalar(expected, dtype)
    assert type(result.value) is type(expected)


@pytest.mark.parametrize("height", [0, 1, 2, 17])
@pytest.mark.parametrize(
    ("scalar", "dtype"),
    [
        (Scalar(1), Int64),
        (Scalar(-0.0), Float64),
        (Scalar(False), Boolean),
        (Scalar("Київ"), String),
        (Scalar("B", Categorical), Categorical),
        *((Scalar(None, dtype), dtype) for dtype in DType),
    ],
)
def test_broadcast_scalar(height: int, scalar: Scalar, dtype: DType) -> None:
    assert_column(broadcast_scalar(scalar, height=height), dtype, [scalar.value] * height)


def test_broadcast_errors() -> None:
    with pytest.raises(SchemaError):
        broadcast_scalar(Scalar(None), height=0)
    with pytest.raises(ShapeError):
        broadcast_scalar(Scalar(2), height=-1)
    with pytest.raises(SchemaError):
        Scalar(None, "Int64")
    with pytest.raises(SchemaError):
        Scalar(True, Int64)
    with pytest.raises(ComputeError):
        Scalar(2**63, Int64)


@pytest.mark.parametrize(
    ("expr", "dtype", "expected"),
    [
        (col("a").sum(), Int64, 4),
        ((col("a") * 2).mean(), Float64, 4.0),
        (col("a").min(), Int64, 1),
        (col("a").max(), Int64, 3),
        (col("a").count(), Int64, 2),
        (col("a").len(), Int64, 3),
        (col("a").n_unique(), Int64, 3),
        (col("a").std(), Float64, sqrt(2)),
        (col("a").var(), Float64, 2.0),
        (col("a").first(), Int64, None),
        (col("a").last().alias("last"), Int64, 3),
        (col("a").mean() + col("a").sum(), Float64, 6.0),
    ],
)
def test_global_reductions(expr: Expr, dtype: DType, expected: _Scalar) -> None:
    result = evaluate_expr(expr, {"a": Column([None, 1, 3], Int64)}, height=3)
    assert result == Scalar(expected, dtype)
    assert type(result.value) is type(expected)


@pytest.mark.parametrize("height", [0, 3])
def test_empty_and_all_null_reductions(height: int) -> None:
    columns = {"a": Column([None] * height, Float64)}
    for op, expected in {
        "sum": Scalar(0.0),
        "mean": Scalar(None, Float64),
        "min": Scalar(None, Float64),
        "max": Scalar(None, Float64),
        "count": Scalar(0),
        "len": Scalar(height),
        "n_unique": Scalar(int(height > 0)),
        "std": Scalar(None, Float64),
        "var": Scalar(None, Float64),
        "first": Scalar(None, Float64),
        "last": Scalar(None, Float64),
    }.items():
        assert evaluate_expr(getattr(col("a"), op)(), columns, height=height) == expected
    assert evaluate_expr(lit(3).sum(), {}, height=height) == Scalar(3 * height)
    assert evaluate_expr(lit(None).cast(String).len(), {}, height=height) == Scalar(height)


@pytest.mark.parametrize("height", [0, 3])
def test_typed_null_operands(height: int) -> None:
    columns = {"ints": Column(range(height), Int64), "nulls": Column([None] * height, Float64)}
    null = col("nulls").mean()
    for expr in (col("ints") + null, null - col("ints"), col("ints") * lit(None).cast(Float64)):
        assert_column(evaluate_expr(expr, columns, height=height), Float64, [None] * height)
    assert_column(
        evaluate_expr(col("ints") == null, columns, height=height), Boolean, [None] * height
    )
    assert evaluate_expr(null + 1, columns, height=height) == Scalar(None, Float64)
    assert evaluate_expr(null.cast(Int64).fill_null(7), columns, height=height) == Scalar(7)
    assert evaluate_expr(1 + null, columns, height=height) == Scalar(None, Float64)
    with pytest.raises(SchemaError):
        evaluate_expr(col("ints") == lit(None).cast(Boolean), columns, height=height)


@pytest.mark.parametrize("dtype", [String, Categorical])
def test_text_and_category_values(dtype: DType) -> None:
    data = ["Kyiv", None, "Lviv", "Kyiv"]
    columns = {"city": Column(data, dtype)}
    city = col("city")
    for expr, expected in [
        (city == "Kyiv", [True, None, False, True]),
        (city >= lit("Lviv").cast(Categorical), [False, None, True, False]),
        ((city == "Kyiv") | (city == "Lviv"), [True, None, True, True]),
    ]:
        assert_column(evaluate_expr(expr, columns, height=4), Boolean, expected)
    assert_column(
        evaluate_expr(city.fill_null("Одеса"), columns, height=4),
        dtype,
        ["Kyiv", "Одеса", "Lviv", "Kyiv"],
    )
    assert evaluate_expr(city.min(), columns, height=4) == Scalar("Kyiv", dtype)
    assert evaluate_expr(city.n_unique(), columns, height=4) == Scalar(3)
    assert_column(columns["city"], dtype, data)


@pytest.mark.parametrize(
    "operation", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]
)
def test_comparison_boundaries(operation: Callable) -> None:
    values = [3, None, 5, 7]
    expr = operation(col("a"), 5)
    result = evaluate_expr(expr, {"a": Column(values, Int64)}, height=len(values))
    assert_column(result, Boolean, [None if v is None else operation(v, 5) for v in values])


def test_ieee_and_strict_cast() -> None:
    columns = {
        "f": Column([nan, None, inf, -0.0], Float64),
        "s": Column(["2", None, "-5", "0"], String),
    }
    f = col("f")
    assert_column(evaluate_expr(f == f, columns, height=4), Boolean, [False, None, True, True])
    assert_column(evaluate_expr(f != f, columns, height=4), Boolean, [True, None, False, False])
    assert_column(evaluate_expr(f > 0, columns, height=4), Boolean, [False, None, True, False])
    assert_column(
        evaluate_expr(f.is_null(), columns, height=4), Boolean, [False, True, False, False]
    )
    assert isnan(evaluate_expr(lit(0) / 0, {}, height=0).value)
    assert_column(evaluate_expr(col("s").cast(Int64), columns, height=4), Int64, [2, None, -5, 0])


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (col("f") & False, [False, False, False]),
        (col("f") | True, [True, True, True]),
        (~col("f"), [False, None, True]),
        (None & col("f"), [None, None, False]),
        (col("f") | None, [True, None, None]),
        (col("f").fill_null(True), [True, True, False]),
        (col("f").fill_null(False), [True, False, False]),
    ],
)
def test_boolean_context(expr: Expr, expected: list[_Scalar]) -> None:
    data = [True, None, False]
    columns = {"f": Column(data, Boolean)}
    assert_column(evaluate_expr(expr, columns, height=3), Boolean, expected)
    assert_column(columns["f"], Boolean, data)


def test_reuse_and_context_validation() -> None:
    expr = (col("a") + col("a")).alias("doubled")
    for values in ([], [1], [2, None, 5]):
        columns = {"a": Column(values, Int64)}
        expected = [None if v is None else 2 * v for v in values]
        assert_column(evaluate_expr(expr, columns, height=len(values)), Int64, expected)
        assert_column(columns["a"], Int64, values)
    missing = defaultdict(lambda: Column([], Int64))
    with pytest.raises(ColumnNotFoundError) as caught:
        evaluate_expr(col("first") + col("second"), missing, height=0)
    assert isinstance(caught.value, SchemaError)
    assert str(caught.value) == "column not found: 'first'"
    assert dict(missing) == {}


@pytest.mark.parametrize(
    ("columns", "height", "error"),
    [
        ({}, -1, ShapeError),
        ({}, True, TypeError),
        ({}, 1.5, TypeError),
        ([], 1, TypeError),
        ({"unused": []}, 0, TypeError),
        ({7: Column([], Int64)}, 0, SchemaError),
        ({"right": Column([], Int64), "wrong": Column([1], Int64)}, 0, ShapeError),
    ],
)
def test_reject_invalid_context_even_for_literal(
    columns: Mapping[str, Column], height: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        evaluate_expr(lit(10), columns, height=height)


@pytest.mark.parametrize(
    ("expr", "error"),
    [
        (lit(INT64_MAX) + 1, ComputeError),
        (-lit(INT64_MIN), ComputeError),
        (col("a") + INT64_MAX, ComputeError),
        (lit("oops").cast(Int64), ComputeError),
        (lit(None).cast(Boolean).cast(String), ComputeError),
        (col("a") + "text", SchemaError),
        (True - col("a"), SchemaError),
        (col("a") | False, SchemaError),
        (~col("a"), SchemaError),
        (col("a") == lit(False), SchemaError),
        (lit(False) < lit(True), SchemaError),
        (lit("x").mean(), SchemaError),
        (lit(None) - None, SchemaError),
        (-lit(None), SchemaError),
        (lit(None).sum(), SchemaError),
        (col("a").fill_null(0.0), SchemaError),
        (lit(True).fill_null(1), SchemaError),
        (lit(0).cast(Boolean), ComputeError),
        (None, TypeError),
    ],
)
def test_invalid_computations(expr: Expr, error: type[Exception]) -> None:
    columns = {"a": Column([1], Int64)}
    with pytest.raises(error):
        evaluate_expr(expr, columns, height=1)
    assert_column(columns["a"], Int64, [1])


@settings(derandomize=True, deadline=None)
@given(data=st.data(), numeric=st.sampled_from([Int64, Float64]), predicate=st.booleans())
def test_generated_trees_match_rows_and_polars(
    data: st.DataObject, numeric: DType, predicate: bool
) -> None:
    dtype = Boolean if predicate else numeric
    tree = data.draw(tree_cases(dtype, numeric, depth=3))
    numbers = st.integers(-8, 8)
    if numeric is Float64:
        numbers = numbers.map(lambda v: v / 4)
    samples = data.draw(
        st.lists(
            st.tuples(st.none() | numbers, st.none() | numbers, st.none() | st.booleans()),
            max_size=24,
        )
    )
    names = ("a", "b", "flag")
    rows = [dict(zip(names, row, strict=True)) for row in samples]
    values = {name: [row[name] for row in rows] for name in names}
    columns = {
        name: Column(xs, Boolean if name == "flag" else numeric) for name, xs in values.items()
    }
    expected = [tree.row_value(row) for row in rows]
    for _ in range(2):
        result = evaluate_expr(tree.expr, columns, height=len(rows))
        if tree.columnar:
            assert_column(result, dtype, expected)
        else:
            assert isinstance(result, Scalar)
            assert result == Scalar(tree.row_value({}), dtype)
            assert_column(broadcast_scalar(result, height=len(rows)), dtype, expected)

    pl_dtype = pl.Int64 if numeric is Int64 else pl.Float64
    frame = pl.DataFrame(values, schema={"a": pl_dtype, "b": pl_dtype, "flag": pl.Boolean})
    oracle = frame.with_columns(tree.polars.alias("result")).get_column("result")
    assert oracle.to_list() == expected
    assert oracle.dtype == (pl.Boolean if predicate else pl_dtype)
    for name, column in columns.items():
        assert_column(column, Boolean if name == "flag" else numeric, values[name])


@settings(derandomize=True)
@given(xs=st.lists(st.none() | st.integers(-10, 10), max_size=40))
def test_composed_kernels(xs: list[int | None]) -> None:
    column = Column(xs, Int64)
    mask = compare.gt(arithmetic.multiply(column, -2), 4)
    direct = logic.or_kleene(mask, compare.eq(column, 0))
    a = col("a")
    result = evaluate_expr((a * -2 > 4) | (a == 0), {"a": column}, height=len(xs))
    assert_column(result, Boolean, direct.to_list())
    assert_column(column, Int64, xs)
