class MiniFrameError(Exception):
    """Base exception for MiniFrame."""


class SchemaError(MiniFrameError):
    """Incompatible schema or types, or duplicate column names."""


class ColumnNotFoundError(SchemaError):
    """Reference to a column that does not exist."""


class ShapeError(MiniFrameError):
    """Incompatible dimensions or lengths."""


class ComputeError(MiniFrameError):
    """Computation failure, including overflow and invalid casts."""


class CorruptedFileError(MiniFrameError):
    """Malformed or corrupted MF1 file."""
