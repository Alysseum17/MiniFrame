from collections.abc import Iterable, Mapping, Sequence
from typing import Self

from .column import Column, _Scalar
from .dtype import Boolean, DType
from .errors import ColumnNotFoundError, SchemaError, ShapeError
from .eval import broadcast_scalar, evaluate_expr
from .expr import Expr, Scalar, _Aggregate, _AggregateOp, _Alias, _name, col, output_name
from .kernels.aggregate import _distinct_indices
from .ops.groupby import hash_aggregate
from .ops.sort import argsort
from .series import Series, _infer_column


def _sequence(value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("expected a sequence")


class DataFrame:
    """Ordered immutable columns with an independent row count.

    Schema entries override inference for named inputs. With no inputs the schema
    creates typed empty columns in schema order.
    """

    __slots__ = ("_columns", "_height")

    def __init__(
        self,
        data: Mapping[str, Iterable[object]] | None = None,
        *,
        schema: Mapping[str, DType] | None = None,
    ) -> None:
        if data is None:
            data = {}
        if schema is None:
            schema = {}
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping")
        if not isinstance(schema, Mapping):
            raise SchemaError("schema must be a mapping")
        for name, dtype in schema.items():
            _name(name)
            if not isinstance(dtype, DType):
                raise SchemaError(f"invalid dtype for {name!r}")
            if data and name not in data:
                raise SchemaError(f"schema column missing from data: {name!r}")
        if not data:
            data = dict.fromkeys(schema, ())
        columns = {}
        for name, values in data.items():
            _name(name)
            columns[name] = (
                Column(values, schema[name]) if name in schema else _infer_column(values)
            )
        self._set_columns(columns)

    def _set_columns(self, columns: dict[str, Column]) -> None:
        height = len(next(iter(columns.values()))) if columns else 0
        if any(len(column) != height for column in columns.values()):
            raise ShapeError("columns must have equal lengths")
        self._columns = columns
        self._height = height

    @classmethod
    def _from_columns(cls, columns: dict[str, Column], height: int) -> Self:
        """Take ownership of an ordered mapping of columns already known to match height."""
        result = cls.__new__(cls)
        result._columns = columns
        result._height = height
        return result

    @classmethod
    def from_series(cls, series: Sequence[Series]) -> Self:
        _sequence(series)
        columns = {}
        for item in series:
            if not isinstance(item, Series):
                raise TypeError("expected a Series")
            name = _name(item.name)
            if name in columns:
                raise SchemaError(f"duplicate column name: {name!r}")
            columns[name] = item._column
        result = cls.__new__(cls)
        result._set_columns(columns)
        return result

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return len(self._columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def schema(self) -> dict[str, DType]:
        return {name: column.dtype for name, column in self._columns.items()}

    def _lookup(self, name: str) -> Column:
        _name(name)
        try:
            return self._columns[name]
        except KeyError:
            raise ColumnNotFoundError(f"column not found: {name!r}") from None

    def column(self, name: str) -> Series:
        return Series._from_column(name, self._lookup(name))

    __getitem__ = column

    def _evaluate(self, expr: Expr) -> Column:
        result = evaluate_expr(expr, self._columns, height=self.height)
        if isinstance(result, Scalar):
            return broadcast_scalar(result, height=self.height)
        if len(result) != self.height:
            raise ShapeError("expression result height differs from frame height")
        return result

    def _project(self, exprs: Sequence[str | Expr]) -> dict[str, Column]:
        columns = {}
        for item in exprs:
            expr = col(item) if isinstance(item, str) else item
            name = output_name(expr)
            if name in columns:
                raise SchemaError(f"duplicate output name: {name!r}")
            columns[name] = self._evaluate(expr)
        return columns

    def select(self, *exprs: str | Expr) -> Self:
        """Project expressions in order, broadcasting scalars to the input height."""
        return self._from_columns(self._project(exprs), self.height)

    def with_columns(self, *exprs: str | Expr) -> Self:
        """Replace or append named expressions, evaluating all against the input."""
        return self._from_columns(self._columns | self._project(exprs), self.height)

    def group_by(self, *keys: str | Expr) -> GroupBy:
        """Define grouping keys for a subsequent eager aggregation."""
        return GroupBy(self, keys)

    def _take(self, indices: Sequence[int]) -> Self:
        columns = {name: column.take(indices) for name, column in self._columns.items()}
        return self._from_columns(columns, len(indices))

    def filter(self, predicate: Expr) -> Self:
        mask = self._evaluate(predicate)
        if mask.dtype is not Boolean:
            raise SchemaError("filter predicate must be Boolean")
        return self._take([i for i in range(self.height) if mask.get(i) is True])

    def sort(
        self,
        by: str | Expr | Sequence[str | Expr],
        *,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
    ) -> Self:
        """Stably sort whole rows by named or computed lexicographic keys."""
        keys = (by,) if isinstance(by, (str, Expr)) else by
        _sequence(keys)
        columns = [
            self._lookup(key) if isinstance(key, str) else self._evaluate(key) for key in keys
        ]
        return self._take(argsort(columns, descending=descending, nulls_last=nulls_last))

    def drop(self, *names: str) -> Self:
        """Remove named columns. Repeated requests raise SchemaError."""
        removed = set()
        for name in names:
            self._lookup(name)
            if name in removed:
                raise SchemaError(f"duplicate drop name: {name!r}")
            removed.add(name)
        kept = {name: column for name, column in self._columns.items() if name not in removed}
        return self._from_columns(kept, self.height)

    def rename(self, mapping: Mapping[str, str]) -> Self:
        if not isinstance(mapping, Mapping):
            raise TypeError("rename requires a mapping")
        for source, destination in mapping.items():
            self._lookup(source)
            _name(destination)
        columns = {mapping.get(name, name): column for name, column in self._columns.items()}
        if len(columns) != self.width:
            raise SchemaError("renaming produces duplicate column names")
        return self._from_columns(columns, self.height)

    def _slice_end(self, n: int, *, tail: bool) -> Self:
        if type(n) is not int:
            raise TypeError("n must be an int")
        if n < 0:
            raise ValueError("n must be nonnegative")
        length = min(n, self.height)
        offset = self.height - length if tail else 0
        columns = {name: column.slice(offset, length) for name, column in self._columns.items()}
        return self._from_columns(columns, length)

    def head(self, n: int = 5) -> Self:
        return self._slice_end(n, tail=False)

    def tail(self, n: int = 5) -> Self:
        return self._slice_end(n, tail=True)

    def _subset(self, subset: Sequence[str] | None) -> tuple[Column, ...]:
        if subset is None:
            return tuple(self._columns.values())
        _sequence(subset)
        return tuple(self._lookup(name) for name in subset)

    def unique(self, subset: Sequence[str] | None = None) -> Self:
        """Keep each key's first row. An empty subset makes all rows tie."""
        return self._take(_distinct_indices(self._subset(subset), self.height))

    def drop_nulls(self, subset: Sequence[str] | None = None) -> Self:
        checks = [column for column in self._subset(subset) if column.null_count]
        if not checks:
            return self._from_columns(self._columns.copy(), self.height)
        return self._take([i for i in range(self.height) if all(c._is_valid(i) for c in checks)])

    def vstack(self, other: DataFrame) -> Self:
        if not isinstance(other, DataFrame):
            raise TypeError("other must be a DataFrame")
        if self.columns != other.columns or self.schema != other.schema:
            raise SchemaError("vstack requires matching ordered schemas")
        columns = {
            name: column._concat(other._columns[name]) for name, column in self._columns.items()
        }
        return self._from_columns(columns, self.height + other.height)

    def rows(self) -> list[tuple[_Scalar, ...]]:
        return [
            tuple(column.get(i) for column in self._columns.values()) for i in range(self.height)
        ]

    def to_dicts(self) -> list[dict[str, _Scalar]]:
        return [
            {name: column.get(i) for name, column in self._columns.items()}
            for i in range(self.height)
        ]

    def equals(self, other: DataFrame, *, null_equal: bool = True) -> bool:
        if not isinstance(other, DataFrame):
            raise TypeError("other must be a DataFrame")
        if type(null_equal) is not bool:
            raise TypeError("null_equal must be a bool")
        return (
            self.shape == other.shape
            and self.columns == other.columns
            and all(self[name].equals(other[name], null_equal=null_equal) for name in self._columns)
        )


class GroupBy:
    """A grouping of a source frame, evaluated on each call to agg."""

    __slots__ = ("_frame", "_keys")

    def __init__(self, frame: DataFrame, keys: tuple[str | Expr, ...]) -> None:
        if not keys:
            raise ValueError("at least one grouping key is required")
        if any(not isinstance(key, (str, Expr)) for key in keys):
            raise TypeError("grouping keys must be strings or expressions")
        self._frame = frame
        self._keys = keys

    def agg(self, *exprs: Expr) -> DataFrame:
        """Reduce each group in first-occurrence order.

        Aggregate operands are evaluated on the full input, including nested reductions.
        """
        keys = self._frame._project(self._keys)
        operands: dict[str, tuple[_AggregateOp, Column]] = {}
        for expr in exprs:
            if not isinstance(expr, Expr):
                raise TypeError("expected an aggregate expression")
            name = output_name(expr)
            while isinstance(expr, _Alias):
                expr = expr.operand
            if not isinstance(expr, _Aggregate):
                raise ShapeError("expected an aggregate at the root of the expression")
            if name in keys or name in operands:
                raise SchemaError(f"duplicate output name: {name!r}")
            operands[name] = expr.op, self._frame._evaluate(expr.operand)
        columns = hash_aggregate(keys, operands, self._frame.height)
        return DataFrame._from_columns(columns, len(next(iter(columns.values()))))
