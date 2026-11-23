import pytest

from miniframe import (
    ColumnNotFoundError,
    ComputeError,
    CorruptedFileError,
    MiniFrameError,
    SchemaError,
    ShapeError,
)


def test_exception_hierarchy() -> None:
    assert MiniFrameError.__bases__ == (Exception,)
    assert ColumnNotFoundError.__bases__ == (SchemaError,)

    direct_subclasses = (
        SchemaError,
        ShapeError,
        ComputeError,
        CorruptedFileError,
    )

    for error_type in direct_subclasses:
        assert error_type.__bases__ == (MiniFrameError,)


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (MiniFrameError, "MiniFrame operation failed"),
        (SchemaError, "Duplicate column name: city"),
        (ColumnNotFoundError, "Column not found: region"),
        (ShapeError, "Expected 3 rows, got 2"),
        (ComputeError, "Int64 addition overflowed"),
        (CorruptedFileError, "Invalid MF1 header checksum"),
    ],
)
def test_base_handler_preserves_error_category_and_message(
    error_type: type[MiniFrameError], message: str
) -> None:
    with pytest.raises(MiniFrameError) as caught:
        raise error_type(message)

    assert type(caught.value) is error_type
    assert str(caught.value) == message
