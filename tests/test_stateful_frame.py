from collections import Counter
from math import isnan
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule
from strategies import frame_inputs, make_frame, physical_snapshot, scalar_bits, values

import miniframe as mf
from miniframe.column import _Scalar


def key(value: _Scalar) -> tuple:
    if value is None:
        return (0,)
    if type(value) is float and isnan(value):
        return (2,)
    return (1, value)


def snapshot(df: mf.DataFrame) -> tuple:
    return df.shape, tuple((name, physical_snapshot(df[name]._column)) for name in df.columns)


class FrameMachine(RuleBasedStateMachine):
    @initialize(inputs=frame_inputs())
    def start(self, inputs: tuple) -> None:
        height, self.schema, data = inputs
        self.frame = make_frame(height, self.schema, data)
        self.rows = [{n: data[n][i] for n in self.schema} for i in range(height)]
        self.previous: tuple[mf.DataFrame, tuple] | None = None
        self.directory = TemporaryDirectory()
        self.counter = 0

    def name(self) -> str:
        self.counter += 1
        return f"new{self.counter}"

    def choose(self, data: st.DataObject) -> str:
        return data.draw(st.sampled_from(list(self.schema)))

    @precondition(lambda self: bool(self.schema))
    @rule(data=st.data(), keep_null=st.booleans())
    def filter(self, data: st.DataObject, keep_null: bool) -> None:
        n = self.choose(data)
        expr = mf.col(n).is_null() if keep_null else mf.col(n).is_not_null()
        self.frame = self.frame.filter(expr)
        self.rows = [r for r in self.rows if (r[n] is None) == keep_null]

    @rule(data=st.data())
    def select(self, data: st.DataObject) -> None:
        names = list(data.draw(st.permutations(list(self.schema))))[
            : data.draw(st.integers(0, len(self.schema)))
        ]
        self.frame = self.frame.select(*names)
        self.schema = {n: self.schema[n] for n in names}
        self.rows = [{n: r[n] for n in names} for r in self.rows]

    @rule(value=st.integers(-4, 4))
    def with_columns(self, value: int) -> None:
        n = self.name()
        self.frame = self.frame.with_columns(mf.lit(value).alias(n))
        self.schema = self.schema | {n: mf.Int64}
        self.rows = [r | {n: value} for r in self.rows]

    @precondition(lambda self: bool(self.schema))
    @rule(data=st.data(), descending=st.booleans())
    def sort(self, data: st.DataObject, descending: bool) -> None:
        n = self.choose(data)
        self.frame = self.frame.sort(n, descending=descending, nulls_last=descending)
        self.rows = sorted(self.rows, key=lambda r: key(r[n]), reverse=descending)

    @precondition(lambda self: bool(self.schema))
    @rule(data=st.data())
    def drop(self, data: st.DataObject) -> None:
        n = self.choose(data)
        self.frame = self.frame.drop(n)
        self.schema = {k: t for k, t in self.schema.items() if k != n}
        self.rows = [{k: r[k] for k in self.schema} for r in self.rows]

    @precondition(lambda self: bool(self.schema))
    @rule(data=st.data())
    def rename(self, data: st.DataObject) -> None:
        old, new = self.choose(data), self.name()
        self.frame = self.frame.rename({old: new})
        self.schema = {new if n == old else n: t for n, t in self.schema.items()}
        self.rows = [{new if n == old else n: v for n, v in r.items()} for r in self.rows]

    @rule()
    def unique(self) -> None:
        self.frame = self.frame.unique()
        first = {}
        for row in self.rows:
            first.setdefault(tuple(key(row[n]) for n in self.schema), row)
        self.rows = list(first.values())

    @rule()
    def drop_nulls(self) -> None:
        self.frame = self.frame.drop_nulls()
        self.rows = [r for r in self.rows if None not in r.values()]

    @rule(data=st.data(), count=st.integers(1, 4))
    def vstack(self, data: st.DataObject, count: int) -> None:
        addition = {n: data.draw(values(t, count)) for n, t in self.schema.items()}
        side = make_frame(count, self.schema, addition)
        before = snapshot(side)
        self.frame = self.frame.vstack(side)
        self.rows += [{n: addition[n][i] for n in self.schema} for i in range(count)]
        assert snapshot(side) == before

    @precondition(lambda self: bool(self.schema))
    @rule(data=st.data())
    def group(self, data: st.DataObject) -> None:
        n, count = self.choose(data), self.name()
        self.frame = self.frame.group_by(n).agg(mf.col(n).len().alias(count))
        counts = Counter(key(r[n]) for r in self.rows)
        first = {}
        for row in self.rows:
            first.setdefault(key(row[n]), row[n])
        self.rows = [{n: v, count: counts[k]} for k, v in first.items()]
        self.schema = {n: self.schema[n], count: mf.Int64}

    @precondition(lambda self: bool(self.schema) and len(self.rows) <= 24)
    @rule(data=st.data())
    def join(self, data: st.DataObject) -> None:
        n, rid = self.choose(data), self.name()
        keys = data.draw(values(self.schema[n], 2))
        if self.rows:
            keys.append(data.draw(st.sampled_from(self.rows))[n])
        side = mf.DataFrame(
            {n: keys, rid: range(len(keys))}, schema={n: self.schema[n], rid: mf.Int64}
        )
        before = snapshot(side)
        self.frame = self.frame.join(side, on=n, how="left")
        joined = []
        for row in self.rows:
            matches = [i for i, v in enumerate(keys) if v is not None and key(v) == key(row[n])]
            joined.extend(row | {rid: i} for i in matches or [None])
        self.rows = joined
        self.schema = self.schema | {rid: mf.Int64}
        assert snapshot(side) == before

    @rule(corrupt=st.booleans())
    def roundtrip(self, corrupt: bool) -> None:
        path = Path(self.directory.name) / "state.mf"
        mf.write_mf(self.frame, path)
        if corrupt:
            canonical = path.read_bytes()
            for damaged in (canonical[:-1], canonical + b"extra"):
                path.write_bytes(damaged)
                with pytest.raises(mf.CorruptedFileError):
                    mf.read_mf(path)
            mf.write_mf(self.frame, path)
            assert path.read_bytes() == canonical
        self.frame = mf.read_mf(path)

    @invariant()
    def agrees_with_rows(self) -> None:
        if self.previous is not None:
            old, before = self.previous
            assert snapshot(old) == before
        assert self.frame.shape == (len(self.rows), len(self.schema))
        assert list(self.frame.schema.items()) == list(self.schema.items())
        for n in self.schema:
            col = self.frame[n]._column
            assert list(map(scalar_bits, col.to_list())) == [scalar_bits(r[n]) for r in self.rows]
            col.check_invariants()
        self.previous = (self.frame, snapshot(self.frame))

    def teardown(self) -> None:
        self.directory.cleanup()


TestFrameMachine = FrameMachine.TestCase
TestFrameMachine.settings = settings(stateful_step_count=12, derandomize=True, deadline=None)
