import operator
from collections.abc import Callable
from dataclasses import dataclass
from struct import pack

import polars as pl
from hypothesis import strategies as st

import miniframe as mf
from miniframe import Boolean, DType, Expr, Float64, Int64, col, lit
from miniframe.column import Column, _Scalar

TEXT = st.text(max_size=16) | st.sampled_from(["", "é", "猫", 'q,u"o\nte'])
SCALARS = {
    mf.Int64: st.integers(-(2**63), 2**63 - 1) | st.sampled_from([-(2**63), 2**63 - 1]),
    mf.Float64: st.floats(width=64)
    | st.sampled_from([-0.0, 0.0, float("inf"), float("-inf"), float("nan")]),
    mf.Boolean: st.booleans(),
    mf.String: TEXT,
    mf.Categorical: st.sampled_from(["", "a", "猫"]) | TEXT,
}


type Row = dict[str, _Scalar]


@dataclass(frozen=True, eq=False)
class TreeCase:
    expr: Expr
    polars: pl.Expr
    row_value: Callable[[Row], _Scalar]
    columnar: bool


@st.composite
def tree_cases(draw: st.DrawFn, dtype: DType, numeric: DType, depth: int) -> TreeCase:
    pl_type = {Boolean: pl.Boolean, Int64: pl.Int64, Float64: pl.Float64}[dtype]
    if depth == 0 or draw(st.booleans()):
        if draw(st.booleans()):
            name = "flag" if dtype is Boolean else draw(st.sampled_from(["a", "b"]))
            return TreeCase(col(name), pl.col(name), operator.itemgetter(name), True)
        values = st.booleans() if dtype is Boolean else st.integers(-4, 4)
        literal = draw(st.none() | values)
        if dtype is Float64 and literal is not None:
            literal = float(literal) / 2
        return TreeCase(
            lit(literal).cast(dtype), pl.lit(literal, pl_type), lambda _: literal, False
        )

    if dtype is Boolean:
        operation = draw(st.sampled_from([operator.and_, operator.or_, operator.gt, operator.eq]))
        child_type = numeric if operation in (operator.gt, operator.eq) else Boolean
    else:
        operation = draw(st.sampled_from([operator.add, operator.sub, operator.mul]))
        child_type = numeric
    left = draw(tree_cases(child_type, numeric, depth - 1))
    right = draw(tree_cases(child_type, numeric, depth - 1))

    def row_value(row: Row) -> _Scalar:
        a, b = left.row_value(row), right.row_value(row)
        if operation is operator.and_ and (a is False or b is False):
            return False
        if operation is operator.or_ and (a is True or b is True):
            return True
        return None if a is None or b is None else operation(a, b)

    return TreeCase(
        operation(left.expr, right.expr),
        operation(left.polars, right.polars),
        row_value,
        left.columnar or right.columnar,
    )


@st.composite
def values(draw: st.DrawFn, dtype: DType, size: int) -> list[_Scalar]:
    mode = draw(st.sampled_from(["mixed", "all null", "all valid", "constant", "sparse", "dense"]))
    if mode == "all null":
        return [None] * size
    if mode == "constant":
        return [draw(SCALARS[dtype])] * size
    scalar = SCALARS[dtype]
    if mode == "mixed":
        scalar = st.none() | scalar
    result = draw(st.lists(scalar, min_size=size, max_size=size))
    if mode in ("sparse", "dense") and size:
        index = draw(st.integers(0, size - 1))
        return [None if (i == index) == (mode == "sparse") else x for i, x in enumerate(result)]
    return result


@st.composite
def frame_inputs(draw: st.DrawFn) -> tuple[int, dict[str, DType], dict[str, list[_Scalar]]]:
    height = draw(st.integers(0, 17))
    types = draw(st.lists(st.sampled_from(list(DType)), max_size=7))
    schema = {f"{i}_{dtype.value}": dtype for i, dtype in enumerate(types)}
    return height, schema, {n: draw(values(t, height)) for n, t in schema.items()}


def make_frame(
    height: int, schema: dict[str, DType], data: dict[str, list[_Scalar]]
) -> mf.DataFrame:
    if not schema:
        return mf.DataFrame({"x": range(height)}, schema={"x": Int64}).select()
    return mf.DataFrame(data, schema=schema)


def frames() -> st.SearchStrategy[mf.DataFrame]:
    return frame_inputs().map(lambda inputs: make_frame(*inputs))


@st.composite
def join_tables(
    draw: st.DrawFn,
) -> tuple[list[tuple[float | None, str]], list[tuple[float | None, str]]]:
    shared = draw(st.sampled_from([0.0, -0.0, float("nan"), float("inf")]))

    def side(label: str) -> list[tuple[float | None, str]]:
        key = st.sampled_from([(shared, "both"), (shared, label), (None, "both")])
        return draw(st.lists(key, max_size=12))

    return side("left"), side("right")


def physical_snapshot(column: Column) -> tuple:
    storage = column._values
    if column.dtype is mf.Categorical:
        buffers = (storage.codes.tobytes(), storage.dictionary)
    elif column.dtype is mf.String:
        buffers = (storage.offsets.tobytes(), bytes(storage.data))
    elif column.dtype is mf.Boolean:
        buffers = (storage.to_bytes(),)
    else:
        buffers = (storage.tobytes(),)
    return column.dtype, len(column), column.null_count, column._validity.to_bytes(), buffers


def scalar_bits(value: _Scalar) -> object:
    return pack("<d", value) if type(value) is float else value
