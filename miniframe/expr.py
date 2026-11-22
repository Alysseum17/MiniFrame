"""Immutable expression syntax and output metadata."""

from dataclasses import dataclass
from typing import Literal

from .column import _Scalar
from .dtype import Boolean, DType, Float64, Int64, String
from .errors import ComputeError, SchemaError

type _Arithmetic = Literal["add", "sub", "mul", "div"]
type _Comparison = Literal["eq", "ne", "lt", "le", "gt", "ge"]
type _BinaryOp = _Arithmetic | _Comparison | Literal["and", "or"]
type _UnaryOp = Literal["neg", "not", "is_null", "is_not_null"]
type _AggregateOp = Literal[
    "sum", "mean", "min", "max", "count", "len", "n_unique", "std", "var", "first", "last"
]
type _Operand = Expr | _Scalar


@dataclass(frozen=True, slots=True)
class Scalar:
    """Internal evaluation value. Only a bare null may lack a dtype."""

    value: _Scalar
    dtype: DType | None = None

    def __post_init__(self) -> None:
        dtype = self.dtype
        if dtype is None:
            if self.value is None:
                return
            dtype = {int: Int64, float: Float64, bool: Boolean, str: String}.get(type(self.value))
            if dtype is None:
                raise TypeError(f"unsupported scalar type: {type(self.value).__name__}")
            object.__setattr__(self, "dtype", dtype)
        if not isinstance(dtype, DType):
            raise SchemaError(f"unsupported scalar dtype: {dtype!r}")
        if not dtype.accepts(self.value):
            if dtype is Int64 and type(self.value) is int:
                raise ComputeError("Int64 overflow")
            raise SchemaError(f"{dtype.value} cannot store {type(self.value).__name__}")


class Expr:
    """A deferred computation built with col(), lit(), and expression operators."""

    __slots__ = ()

    def __bool__(self) -> bool:
        raise TypeError("the truth value of an Expr is ambiguous")

    def __add__(self, other: _Operand) -> Expr:
        return _Binary("add", self, _as_expr(other))

    def __radd__(self, other: _Scalar) -> Expr:
        return _Binary("add", lit(other), self)

    def __sub__(self, other: _Operand) -> Expr:
        return _Binary("sub", self, _as_expr(other))

    def __rsub__(self, other: _Scalar) -> Expr:
        return _Binary("sub", lit(other), self)

    def __mul__(self, other: _Operand) -> Expr:
        return _Binary("mul", self, _as_expr(other))

    def __rmul__(self, other: _Scalar) -> Expr:
        return _Binary("mul", lit(other), self)

    def __truediv__(self, other: _Operand) -> Expr:
        return _Binary("div", self, _as_expr(other))

    def __rtruediv__(self, other: _Scalar) -> Expr:
        return _Binary("div", lit(other), self)

    def __neg__(self) -> Expr:
        return _Unary("neg", self)

    def __eq__(self, other: _Operand) -> Expr:
        return _Binary("eq", self, _as_expr(other))

    def __ne__(self, other: _Operand) -> Expr:
        return _Binary("ne", self, _as_expr(other))

    def __lt__(self, other: _Operand) -> Expr:
        return _Binary("lt", self, _as_expr(other))

    def __le__(self, other: _Operand) -> Expr:
        return _Binary("le", self, _as_expr(other))

    def __gt__(self, other: _Operand) -> Expr:
        return _Binary("gt", self, _as_expr(other))

    def __ge__(self, other: _Operand) -> Expr:
        return _Binary("ge", self, _as_expr(other))

    def __and__(self, other: _Operand) -> Expr:
        return _Binary("and", self, _as_expr(other))

    def __rand__(self, other: _Scalar) -> Expr:
        return _Binary("and", lit(other), self)

    def __or__(self, other: _Operand) -> Expr:
        return _Binary("or", self, _as_expr(other))

    def __ror__(self, other: _Scalar) -> Expr:
        return _Binary("or", lit(other), self)

    def __invert__(self) -> Expr:
        return _Unary("not", self)

    def is_null(self) -> Expr:
        """Mark null positions with True in a fully valid Boolean result."""
        return _Unary("is_null", self)

    def is_not_null(self) -> Expr:
        """Mark valid positions with True in a fully valid Boolean result."""
        return _Unary("is_not_null", self)

    def fill_null(self, value: _Scalar) -> Expr:
        """Replace nulls with a scalar accepted by the operand's dtype."""
        return _Fill(self, Scalar(value))

    def cast(self, dtype: DType) -> Expr:
        """Request a strict cast. Casting a bare null gives it a dtype."""
        if not isinstance(dtype, DType):
            raise SchemaError(f"unsupported cast dtype: {dtype!r}")
        return _Cast(self, dtype)

    def alias(self, name: str) -> Expr:
        """Name this expression's output in its execution context."""
        return _Alias(self, _name(name))

    def sum(self) -> Expr:
        """Sum valid numeric values, returning a typed zero for an empty sample."""
        return _Aggregate("sum", self)

    def mean(self) -> Expr:
        """Average valid numeric values, returning null for an empty sample."""
        return _Aggregate("mean", self)

    def min(self) -> Expr:
        """Select the least valid numeric or text value, ordering NaN last."""
        return _Aggregate("min", self)

    def max(self) -> Expr:
        """Select the greatest valid numeric or text value, ordering NaN last."""
        return _Aggregate("max", self)

    def count(self) -> Expr:
        """Count valid positions."""
        return _Aggregate("count", self)

    def len(self) -> Expr:
        """Count all positions, including nulls."""
        return _Aggregate("len", self)

    def n_unique(self) -> Expr:
        """Count distinct logical values, counting null once when present."""
        return _Aggregate("n_unique", self)

    def std(self) -> Expr:
        """Compute sample standard deviation with at least two valid observations."""
        return _Aggregate("std", self)

    def var(self) -> Expr:
        """Compute sample variance with at least two valid observations."""
        return _Aggregate("var", self)

    def first(self) -> Expr:
        """Read the first position, including null, or return null for empty input."""
        return _Aggregate("first", self)

    def last(self) -> Expr:
        """Read the last position, including null, or return null for empty input."""
        return _Aggregate("last", self)


@dataclass(frozen=True, slots=True, eq=False)
class _ColumnRef(Expr):
    name: str


@dataclass(frozen=True, slots=True, eq=False)
class _Literal(Expr):
    scalar: Scalar


@dataclass(frozen=True, slots=True, eq=False)
class _Binary(Expr):
    op: _BinaryOp
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True, eq=False)
class _Unary(Expr):
    op: _UnaryOp
    operand: Expr


@dataclass(frozen=True, slots=True, eq=False)
class _Fill(Expr):
    operand: Expr
    replacement: Scalar


@dataclass(frozen=True, slots=True, eq=False)
class _Cast(Expr):
    operand: Expr
    dtype: DType


@dataclass(frozen=True, slots=True, eq=False)
class _Alias(Expr):
    operand: Expr
    name: str


@dataclass(frozen=True, slots=True, eq=False)
class _Aggregate(Expr):
    op: _AggregateOp
    operand: Expr


def _name(name: str) -> str:
    if type(name) is not str:
        raise SchemaError("column name must be a string")
    return name


def col(name: str) -> Expr:
    """Reference a named column in a future execution context."""
    return _ColumnRef(_name(name))


def lit(value: _Scalar) -> Expr:
    """Describe a built-in scalar, including an untyped null."""
    return _Literal(Scalar(value))


def _as_expr(value: _Operand) -> Expr:
    return value if isinstance(value, Expr) else lit(value)


def _first_column(expr: Expr) -> str | None:
    match expr:
        case _ColumnRef(name):
            return name
        case _Literal():
            return None
        case _Binary(_, left, right):
            first = _first_column(left)
            return _first_column(right) if first is None else first
        case _Unary() | _Fill() | _Cast() | _Alias() | _Aggregate():
            return _first_column(expr.operand)
        case _:
            raise TypeError("expected an expression built with col() or lit()")


def output_name(expr: Expr) -> str:
    """Use the outer alias, first source name in operand order, or 'literal'."""
    if isinstance(expr, _Alias):
        return expr.name
    name = _first_column(expr)
    return "literal" if name is None else name


def is_root_aggregation(expr: Expr) -> bool:
    """Whether evaluation ends in a reduction, ignoring surrounding aliases."""
    while isinstance(expr, _Alias):
        expr = expr.operand
    return isinstance(expr, _Aggregate)
