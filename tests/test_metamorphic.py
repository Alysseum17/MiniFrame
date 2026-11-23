from math import isnan

import duckdb
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import TreeCase, frames, join_tables, physical_snapshot, tree_cases

import miniframe as mf


@settings(derandomize=True, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.none() | st.integers(-8, 8),
            st.none() | st.integers(-8, 8),
            st.none() | st.booleans(),
        ),
        max_size=20,
    ),
    tree_cases(mf.Boolean, mf.Int64, 2),
    tree_cases(mf.Boolean, mf.Int64, 2),
)
def test_filter_partition_and_commutation(rows: list[tuple], p: TreeCase, q: TreeCase) -> None:
    data = {k: [r[i] for r in rows] for i, k in enumerate(("a", "b", "flag"))} | {
        "id": list(range(len(rows)))
    }
    df = mf.DataFrame(
        data, schema={"a": mf.Int64, "b": mf.Int64, "flag": mf.Boolean, "id": mf.Int64}
    )
    pl_df = pl.DataFrame(
        data, schema={"a": pl.Int64, "b": pl.Int64, "flag": pl.Boolean, "id": pl.Int64}
    )
    pq = df.filter(p.expr).filter(q.expr)
    assert pq.equals(df.filter(q.expr).filter(p.expr))
    assert pq.rows() == pl_df.filter(p.polars).filter(q.polars).rows()
    assert pq["id"].to_list() == [
        i for i, r in enumerate(df.to_dicts()) if p.row_value(r) is True and q.row_value(r) is True
    ]
    partitions = [df.filter(expr)["id"].to_list() for expr in (p.expr, ~p.expr, p.expr.is_null())]
    assert sorted(i for part in partitions for i in part) == list(range(len(rows)))


@settings(derandomize=True, deadline=None)
@given(frames(), st.integers(-18, 18))
def test_inverse_shift_retains_surviving_rows(df: mf.DataFrame, distance: int) -> None:
    before = [physical_snapshot(df[n]._column) for n in df.columns]
    shifted = df.select(*(mf.col(n).shift(distance).shift(-distance) for n in df.columns))
    assert shifted.schema == df.schema
    assert shifted.shape == df.shape
    for i in range(df.height):
        for n in df.columns:
            if 0 <= i + distance < df.height:
                assert shifted[n].slice(i, 1).equals(df[n].slice(i, 1))
            else:
                assert shifted[n][i] is None
    assert before == [physical_snapshot(df[n]._column) for n in df.columns]


def test_composite_outer_join_then_grouped_window() -> None:
    def columns(keys: list[tuple], name: str) -> dict[str, list]:
        return {"k": [k for k, _ in keys], "s": [s for _, s in keys], name: list(range(len(keys)))}

    def normalized(rows: list[tuple]) -> list[tuple]:
        return [(("NaN" if isinstance(k, float) and isnan(k) else k), *rest) for k, *rest in rows]

    with duckdb.connect(config={"threads": "1"}) as db:

        @settings(derandomize=True, deadline=None)
        @given(join_tables())
        def exercise(tables: tuple[list, list]) -> None:
            left, right = (columns(keys, n) for keys, n in zip(tables, ("l", "r"), strict=True))
            schema = {"k": mf.Float64, "s": mf.Categorical}
            a = mf.DataFrame(left, schema=schema | {"l": mf.Int64})
            b = mf.DataFrame(right, schema=schema | {"r": mf.Int64})
            result = a.join(b, on=["k", "s"], how="full").with_columns(
                mf.col("l").len().over("k", "s").alias("n")
            )
            pl_schema = {"k": pl.Float64, "s": pl.String}
            pa = pl.DataFrame(left, schema=pl_schema | {"l": pl.Int64})
            pb = pl.DataFrame(right, schema=pl_schema | {"r": pl.Int64})
            order = ["k", "s", "l", "r", "n"]
            expected = (
                pa.join(pb, on=["k", "s"], how="full", coalesce=True)
                .with_columns(pl.len().over("k", "s").alias("n"))
                .sort(["l", "r"], nulls_last=True)
                .select(order)
            )
            db.register("lhs", pa.to_arrow())
            db.register("rhs", pb.to_arrow())
            sql = db.execute("""
                select k, s, l, r, count(*) over (partition by k, s)
                from lhs full join rhs using(k,s) order by l nulls last, r nulls last
            """).fetchall()
            assert result.columns == order
            assert normalized(result.rows()) == normalized(expected.rows()) == normalized(sql)

        exercise()


@settings(derandomize=True, deadline=None)
@given(frames())
def test_failed_operations_preserve_reusable_input(source: mf.DataFrame) -> None:
    df = source.with_columns(mf.lit(True).alias("flag"))
    before = [physical_snapshot(df[n]._column) for n in df.columns]
    with pytest.raises(mf.SchemaError):
        df.join(mf.DataFrame({"flag": [1]}), on="flag")
    with pytest.raises(mf.ShapeError):
        df.group_by("flag").agg(mf.col("flag").alias("result"))
    with pytest.raises(mf.ColumnNotFoundError):
        df.select(mf.col("absent").shift(1))
    with pytest.raises(mf.ComputeError):
        df.select(mf.col("flag").cast(mf.Float64))
    assert before == [physical_snapshot(df[n]._column) for n in df.columns]
    assert df.drop("flag").equals(source)
