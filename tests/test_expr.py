import operator
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from math import inf, isnan, nan

import pytest

from miniframe import Boolean, ComputeError, Expr, Float64, Int64, SchemaError, String, col, lit
from miniframe.column import Column, _Scalar
from miniframe.dtype import INT64_MAX, INT64_MIN, DType
from miniframe.expr import (
    Scalar,
    _Aggregate,
    _Alias,
    _Binary,
    _Cast,
    _ColumnRef,
    _Fill,
    _Literal,
    _Unary,
    is_root_aggregation,
    output_name,
)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (INT64_MIN, Int64),
        (INT64_MAX, Int64),
        (False, Boolean),
        ("Київ", String),
        (None, None),
        (1.5, Float64),
        (nan, Float64),
        (-inf, Float64),
        (inf, Float64),
    ],
)
def test_literal(value: _Scalar, dtype: DType | None) -> None:
    expr = lit(value)
    assert isinstance(expr, _Literal)
    assert expr.scalar.dtype is dtype
    if isinstance(value, float) and isnan(value):
        assert isnan(expr.scalar.value)
    else:
        assert expr.scalar.value == value


@pytest.mark.parametrize("value", [[], b"bytes", 1j, object()])
def test_reject_unsupported_scalars(value: object) -> None:
    with pytest.raises(TypeError):
        lit(value)
    with pytest.raises(TypeError):
        col("a").fill_null(value)
    with pytest.raises(TypeError):
        col("a") + value


def test_expression_is_not_a_literal_replacement() -> None:
    with pytest.raises(TypeError):
        lit(col("x"))
    with pytest.raises(TypeError):
        col("a").fill_null(col("x"))


@pytest.mark.parametrize("value", [INT64_MIN - 1, INT64_MAX + 1])
def test_validate_literal_bounds(value: int) -> None:
    with pytest.raises(ComputeError):
        lit(value)


def test_names_and_dtypes() -> None:
    assert isinstance(col(""), _ColumnRef)
    assert col("price").name == "price"
    for name in (None, b"price", 42):
        with pytest.raises(SchemaError):
            col(name)
        with pytest.raises(SchemaError):
            col("price").alias(name)
    with pytest.raises(SchemaError):
        col("price").cast("Float64")
    with pytest.raises(TypeError):
        output_name("price")


@pytest.mark.parametrize(
    ("operation", "kind"),
    [
        (operator.add, "add"),
        (operator.sub, "sub"),
        (operator.mul, "mul"),
        (operator.truediv, "div"),
        (operator.eq, "eq"),
        (operator.ne, "ne"),
        (operator.lt, "lt"),
        (operator.le, "le"),
        (operator.gt, "gt"),
        (operator.ge, "ge"),
        (operator.and_, "and"),
        (operator.or_, "or"),
    ],
)
def test_binary_nodes(operation: Callable, kind: str) -> None:
    left, right = col("left"), col("right")
    expr = operation(left, right)
    assert isinstance(expr, _Binary)
    assert expr.op == kind
    assert expr.left is left
    assert expr.right is right


@pytest.mark.parametrize(
    ("operation", "kind", "value"),
    [
        (operator.add, "add", 10),
        (operator.sub, "sub", 10),
        (operator.mul, "mul", 10),
        (operator.truediv, "div", 10),
        (operator.and_, "and", True),
        (operator.or_, "or", False),
    ],
)
def test_reflection_preserves_operand_order(operation: Callable, kind: str, value: _Scalar) -> None:
    source = col("x")
    expr = operation(value, source)
    assert isinstance(expr, _Binary)
    assert expr.op == kind
    assert isinstance(expr.left, _Literal)
    assert expr.left.scalar == Scalar(value)
    assert expr.right is source


def test_nested_lazy_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_allocation(*args: object) -> None:
        pytest.fail("building an expression allocated a Column")

    monkeypatch.setattr(Column, "__init__", reject_allocation)
    source = col("absent")
    product = source * (lit(INT64_MAX) + 1)
    expr = (-product).cast(Float64).fill_null(0.0).alias("answer")
    assert isinstance(expr, _Alias)
    assert expr.name == "answer"
    assert isinstance(expr.operand, _Fill)
    assert expr.operand.replacement == Scalar(0.0, Float64)
    cast = expr.operand.operand
    assert isinstance(cast, _Cast)
    assert cast.dtype is Float64
    assert isinstance(cast.operand, _Unary)
    assert cast.operand.op == "neg"
    assert cast.operand.operand is product
    assert product.left is source
    assert product.right.op == "add"
    assert not hasattr(expr, "evaluate")


@pytest.mark.parametrize(
    ("build", "kind"),
    [(operator.invert, "not"), (Expr.is_null, "is_null"), (Expr.is_not_null, "is_not_null")],
)
def test_unary_nodes(build: Callable[[Expr], Expr], kind: str) -> None:
    source = col("flag")
    result = build(source)
    assert isinstance(result, _Unary)
    assert result.op == kind
    assert result.operand is source


@pytest.mark.parametrize(
    "op", ["sum", "mean", "min", "max", "count", "len", "n_unique", "std", "var", "first", "last"]
)
def test_reduction_metadata(op: str) -> None:
    source = col("v") + 1
    result = getattr(source, op)()
    assert isinstance(result, _Aggregate)
    assert result.op == op
    assert result.operand is source
    assert is_root_aggregation(result)
    assert is_root_aggregation(result.alias("x").alias("y"))
    assert not is_root_aggregation(result + 1)
    assert not is_root_aggregation(result.cast(Float64))
    assert not is_root_aggregation(source.alias("x"))


def test_immutable_nodes_and_ambiguous_truth() -> None:
    expr = col("a") == col("a")
    assert isinstance(expr, Expr)
    with pytest.raises(TypeError):
        bool(expr)
    with pytest.raises(TypeError):
        _ = 0 < col("a") < 10
    with pytest.raises(FrozenInstanceError):
        expr.left = lit(9)
    with pytest.raises(FrozenInstanceError):
        expr.left.name = "b"
    with pytest.raises(FrozenInstanceError):
        lit(1).scalar.value = 5
    assert not hasattr(expr, "__dict__")


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (lit(None), "literal"),
        (lit(2) + 3, "literal"),
        (lit(1).alias("one"), "one"),
        (lit(1) + (col("right") - col("left")), "right"),
        ((col("a").alias("inner") + col("b")).fill_null(0).sum(), "a"),
        (col("a").sum().alias("inner").alias("outer"), "outer"),
        ((lit(0) + col("")) * col("b"), ""),
        (lit(4).cast(Int64) + col("b"), "b"),
    ],
)
def test_output_names(expr: Expr, expected: str) -> None:
    assert output_name(expr) == expected
