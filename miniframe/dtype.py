from enum import Enum, unique
from typing import Final

INT64_MIN: Final = -(2**63)
INT64_MAX: Final = 2**63 - 1


@unique
class DType(Enum):
    """Logical column types with nullable values and a separate validity bitmap.

    Int64 and Float64 use arrays with typecodes q and d. Boolean values use a
    bit-packed bitmap. String uses 32-bit offsets into a UTF-8 byte buffer.
    Categorical uses 32-bit codes into a dictionary of unique strings.
    """

    Int64 = "Int64"
    Float64 = "Float64"
    Boolean = "Boolean"
    String = "String"
    Categorical = "Categorical"

    def __hash__(self) -> int:
        return int.from_bytes(self.value.encode("ascii"))

    @property
    def is_numeric(self) -> bool:
        return self is DType.Int64 or self is DType.Float64

    @property
    def array_typecode(self) -> str | None:
        """Return the array typecode for values, string offsets, or category codes."""
        match self:
            case DType.Int64:
                return "q"
            case DType.Float64:
                return "d"
            case DType.Boolean:
                return None
            case DType.String | DType.Categorical:
                return "i"

    def accepts(self, value: object) -> bool:
        """Check nullable scalar compatibility without coercion or text encoding."""
        if value is None:
            return True

        match self:
            case DType.Int64:
                return type(value) is int and INT64_MIN <= value <= INT64_MAX
            case DType.Float64:
                return type(value) is float
            case DType.Boolean:
                return type(value) is bool
            case DType.String | DType.Categorical:
                return type(value) is str


Int64: Final = DType.Int64
Float64: Final = DType.Float64
Boolean: Final = DType.Boolean
String: Final = DType.String
Categorical: Final = DType.Categorical
