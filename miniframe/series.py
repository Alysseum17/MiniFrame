from collections.abc import Iterable
from itertools import chain, repeat
from math import isnan
from typing import Self

from .bitmap import Bitmap
from .column import Column, _Scalar
from .dtype import Boolean, DType, Float64, Int64, String
from .errors import SchemaError
from .kernels import arithmetic, compare, logic
from .kernels.cast import cast_column


def _infer_column(values: Iterable[object]) -> Column:
    iterator = iter(values)
    for leading_nulls, value in enumerate(iterator):
        if value is None:
            continue
        dtype = {int: Int64, float: Float64, bool: Boolean, str: String}.get(type(value))
        if dtype is None:
            raise SchemaError(f"cannot infer dtype from {type(value).__name__}")
        return Column(chain(repeat(None, leading_nulls), (value,), iterator), dtype)
    raise SchemaError("empty or all-null Series requires an explicit dtype")


def _operand[T](value: Series | T) -> Column | T:
    return value._column if isinstance(value, Series) else value


class Series:
    """A named, immutable column with positional access and nullable elementwise operators.

    Without an explicit dtype, the first non-null built-in int, float, bool, or str
    determines the dtype. All remaining non-null values must have that same type.
    """

    __slots__ = ("_name", "_column")

    def __init__(self, name: str, values: Iterable[object], dtype: DType | None = None) -> None:
        if not isinstance(name, str):
            raise SchemaError("Series name must be a string")
        self._column = _infer_column(values) if dtype is None else Column(values, dtype)
        self._name = name

    def _with_column(self, column: Column) -> Self:
        result = type(self).__new__(type(self))
        result._name = self._name
        result._column = column
        return result

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> DType:
        return self._column.dtype

    @property
    def null_count(self) -> int:
        return self._column.null_count

    def __len__(self) -> int:
        return len(self._column)

    def __getitem__(self, index: int) -> _Scalar:
        """Read a zero-based position given as a built-in integer."""
        return self._column.get(index)

    def to_list(self) -> list[_Scalar]:
        """Return an independent list of logical values, including None for nulls."""
        return self._column.to_list()

    def slice(self, offset: int, length: int) -> Self:
        """Copy a range wholly inside the Series."""
        return self._with_column(self._column.slice(offset, length))

    def cast(self, dtype: DType) -> Self:
        """Copy into the target dtype using strict conversions."""
        return self._with_column(cast_column(self._column, dtype))

    def _mask_series(self, values: Bitmap) -> Self:
        return self._with_column(
            self._column._with_storage(values, Bitmap(len(self), True), dtype=Boolean)
        )

    def is_null(self) -> Self:
        """Return a fully valid Boolean Series marking null positions."""
        return self._mask_series(~self._column._validity)

    def is_not_null(self) -> Self:
        """Return a fully valid Boolean Series marking valid positions."""
        return self._mask_series(self._column._validity.slice(0, len(self)))

    def fill_null(self, value: object) -> Self:
        """Copy this Series, replacing nulls with a value accepted by its dtype."""
        if value is None or self.null_count == 0:
            Column((value,), self.dtype)
            return self.slice(0, len(self))
        values = (value if (item := self[i]) is None else item for i in range(len(self)))
        return self._with_column(Column(values, self.dtype))

    def equals(self, other: Series, *, null_equal: bool = True) -> bool:
        """Compare name, dtype and logical values. Paired NaNs and signed zeros match.

        Matching nulls compare equal only when null_equal is True.
        """
        if not isinstance(other, Series):
            raise TypeError("other must be a Series")
        if type(null_equal) is not bool:
            raise TypeError("null_equal must be a bool")
        if self.name != other.name or self.dtype is not other.dtype or len(self) != len(other):
            return False
        for i in range(len(self)):
            left, right = self[i], other[i]
            if isinstance(left, float) and isnan(left):
                if not isinstance(right, float) or not isnan(right):
                    return False
            elif left != right or (left is None and not null_equal):
                return False
        return True

    def __bool__(self) -> bool:
        raise TypeError("the truth value of a Series is ambiguous")

    def __add__(self, other: Series | int | float | None) -> Self:
        return self._with_column(arithmetic.add(self._column, _operand(other)))

    def __radd__(self, other: int | float | None) -> Self:
        return self._with_column(arithmetic.add(other, self._column))

    def __sub__(self, other: Series | int | float | None) -> Self:
        return self._with_column(arithmetic.subtract(self._column, _operand(other)))

    def __rsub__(self, other: int | float | None) -> Self:
        return self._with_column(arithmetic.subtract(other, self._column))

    def __mul__(self, other: Series | int | float | None) -> Self:
        return self._with_column(arithmetic.multiply(self._column, _operand(other)))

    def __rmul__(self, other: int | float | None) -> Self:
        return self._with_column(arithmetic.multiply(other, self._column))

    def __truediv__(self, other: Series | int | float | None) -> Self:
        return self._with_column(arithmetic.divide(self._column, _operand(other)))

    def __rtruediv__(self, other: int | float | None) -> Self:
        return self._with_column(arithmetic.divide(other, self._column))

    def __neg__(self) -> Self:
        return self._with_column(arithmetic.negate(self._column))

    def __eq__(self, other: Series | Column | _Scalar) -> Self:
        return self._with_column(compare.eq(self._column, _operand(other)))

    def __ne__(self, other: Series | Column | _Scalar) -> Self:
        return self._with_column(compare.ne(self._column, _operand(other)))

    def __lt__(self, other: Series | Column | _Scalar) -> Self:
        return self._with_column(compare.lt(self._column, _operand(other)))

    def __le__(self, other: Series | Column | _Scalar) -> Self:
        return self._with_column(compare.le(self._column, _operand(other)))

    def __gt__(self, other: Series | Column | _Scalar) -> Self:
        return self._with_column(compare.gt(self._column, _operand(other)))

    def __ge__(self, other: Series | Column | _Scalar) -> Self:
        return self._with_column(compare.ge(self._column, _operand(other)))

    def _boolean_operand(self, other: Series | bool | None) -> Column:
        if isinstance(other, Series):
            return other._column
        if other is not None and type(other) is not bool:
            raise TypeError("expected a Boolean Series, bool or None")
        return self._column._with_storage(
            Bitmap(len(self), other is True), Bitmap(len(self), other is not None), dtype=Boolean
        )

    def __and__(self, other: Series | bool | None) -> Self:
        return self._with_column(logic.and_kleene(self._column, self._boolean_operand(other)))

    __rand__ = __and__

    def __or__(self, other: Series | bool | None) -> Self:
        return self._with_column(logic.or_kleene(self._column, self._boolean_operand(other)))

    __ror__ = __or__

    def __invert__(self) -> Self:
        return self._with_column(logic.not_kleene(self._column))
