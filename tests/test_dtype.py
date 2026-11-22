from math import inf, nan
from pickle import dumps, loads

import pytest
from hypothesis import given
from hypothesis import strategies as st

from miniframe import Boolean, Categorical, DType, Float64, Int64, String
from miniframe.dtype import INT64_MAX, INT64_MIN


def test_supported_types() -> None:
    assert tuple(DType) == (Int64, Float64, Boolean, String, Categorical)


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("Int64", Int64),
        ("Float64", Float64),
        ("Boolean", Boolean),
        ("String", String),
        ("Categorical", Categorical),
    ],
)
def test_dtype_identity_equality_and_hashing(name: str, dtype: DType) -> None:
    reconstructed = DType(name)
    assert reconstructed is dtype
    assert loads(dumps(dtype)) is dtype
    assert dtype != name
    assert {dtype: name}[reconstructed] == name


@pytest.mark.parametrize("invalid", [None, True, 1, int, "int64", "Null", []])
def test_invalid_dtype_inputs(invalid: object) -> None:
    with pytest.raises(ValueError):
        DType(invalid)


@pytest.mark.parametrize("attribute", ["name", "value"])
def test_dtype_identity_fields_are_read_only(attribute: str) -> None:
    with pytest.raises(AttributeError):
        setattr(Int64, attribute, "Changed")


def test_int64_bounds() -> None:
    assert (INT64_MIN, INT64_MAX) == (-9_223_372_036_854_775_808, 9_223_372_036_854_775_807)


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (-(2**63) - 1, False),
        (-(2**63), True),
        (2**63 - 1, True),
        (2**63, False),
    ],
)
def test_int64_accepts_boundary_values(value: int, accepted: bool) -> None:
    assert Int64.accepts(value) is accepted


@given(st.integers())
def test_int64_accepts_exactly_signed_64_bit_integers(value: int) -> None:
    try:
        value.to_bytes(8, signed=True)
    except OverflowError:
        expected = False
    else:
        expected = True
    assert Int64.accepts(value) is expected


@pytest.mark.parametrize("dtype", tuple(DType))
def test_all_dtypes_are_nullable(dtype: DType) -> None:
    assert dtype.accepts(None)


def test_nan_is_a_float_value() -> None:
    assert {dtype for dtype in DType if dtype.accepts(nan)} == {Float64}


def test_numeric_classification() -> None:
    assert {dtype for dtype in DType if dtype.is_numeric} == {Int64, Float64}


@pytest.mark.parametrize(
    ("dtype", "typecode"),
    [(Int64, "q"), (Float64, "d"), (Boolean, None), (String, "i"), (Categorical, "i")],
)
def test_array_typecode(dtype: DType, typecode: str | None) -> None:
    assert dtype.array_typecode == typecode


@pytest.mark.parametrize(
    ("value", "compatible_dtypes"),
    [
        (False, {Boolean}),
        (True, {Boolean}),
        (0, {Int64}),
        (1, {Int64}),
        (1.0, {Float64}),
        (-0.0, {Float64}),
        (inf, {Float64}),
        (-inf, {Float64}),
        ("", {String, Categorical}),
        ("Київ", {String, Categorical}),
        (b"text", set()),
        (object(), set()),
    ],
)
def test_scalar_compatibility(value: object, compatible_dtypes: set[DType]) -> None:
    assert {dtype for dtype in DType if dtype.accepts(value)} == compatible_dtypes


def test_int_subclass_is_not_an_exact_int_scalar() -> None:
    class CustomInt(int):
        pass

    assert not Int64.accepts(CustomInt(1))
