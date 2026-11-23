from itertools import accumulate
from math import isfinite

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
from strategies import SCALARS, physical_snapshot, scalar_bits, values

from miniframe import Boolean, Categorical, DType, Float64, Int64, String
from miniframe.column import Column
from miniframe.kernels.cast import cast_column


class ColumnMachine(RuleBasedStateMachine):
    @initialize(
        dtype=st.sampled_from(list(DType)),
        size=st.sampled_from([0, 1, 7, 8, 9, 16, 17]),
        data=st.data(),
    )
    def start(self, dtype: DType, size: int, data: st.DataObject) -> None:
        self.dtype = dtype
        self.model = data.draw(values(dtype, size))
        self.column = Column(self.model, dtype)
        self.previous: tuple[Column, tuple] | None = None

    @rule(data=st.data())
    def take(self, data: st.DataObject) -> None:
        indices = (
            data.draw(st.lists(st.integers(0, len(self.model) - 1), max_size=24))
            if self.model
            else []
        )
        self.column = self.column.take(indices)
        self.model = [self.model[i] for i in indices]

    @rule(data=st.data())
    def slice(self, data: st.DataObject) -> None:
        start = data.draw(st.integers(0, len(self.model)))
        size = data.draw(st.integers(0, len(self.model) - start))
        self.column = self.column.slice(start, size)
        self.model = self.model[start : start + size]

    @rule(data=st.data(), size=st.integers(1, 9))
    def append(self, data: st.DataObject, size: int) -> None:
        tail = data.draw(values(self.dtype, size))
        other = Column(tail, self.dtype)
        old = physical_snapshot(other)
        self.column = self.column._concat(other)
        self.model = self.model + tail
        assert physical_snapshot(other) == old

    @rule(data=st.data())
    def fill(self, data: st.DataObject) -> None:
        replacement = data.draw(SCALARS[self.dtype])
        self.column = self.column.fill_null(replacement)
        self.model = [replacement if x is None else x for x in self.model]

    @rule()
    def cast(self) -> None:
        target = self.dtype
        present = [v for v in self.model if v is not None]
        if target in (Categorical, String):
            target = String if target is Categorical else Categorical
        elif target is Boolean:
            target = Int64
        elif target is Int64 and all(float(v) == v for v in present):
            target = Float64
        elif target is Float64 and all(
            isfinite(v) and v.is_integer() and -(2**63) <= v < 2**63 for v in present
        ):
            target = Int64
        self.column = cast_column(self.column, target)
        if self.dtype is not target and target in (Int64, Float64):
            convert = int if target is Int64 else float
            self.model = [None if v is None else convert(v) for v in self.model]
        self.dtype = target

    @invariant()
    def agrees_with_list(self) -> None:
        if self.previous is not None:
            old, snapshot = self.previous
            assert physical_snapshot(old) == snapshot
        col = self.column
        assert len(col) == len(self.model)
        assert col.dtype is self.dtype
        assert col.null_count == self.model.count(None)
        assert list(map(scalar_bits, col.to_list())) == list(map(scalar_bits, self.model))
        mask = sum(1 << i for i, v in enumerate(self.model) if v is not None)
        assert col._validity.to_bytes() == mask.to_bytes((len(col) + 7) // 8, "little")
        col.check_invariants()
        if self.dtype is String:
            chunks = [(v or "").encode() for v in self.model]
            assert col._values.data == b"".join(chunks)
            assert col._values.offsets.tolist() == list(accumulate(map(len, chunks), initial=0))
        if self.dtype is Categorical:
            dictionary = col._values.dictionary
            assert col._values.codes.tolist() == [
                0 if v is None else dictionary.index(v) for v in self.model
            ]
        self.previous = (col, physical_snapshot(col))


TestColumnMachine = ColumnMachine.TestCase
TestColumnMachine.settings = settings(stateful_step_count=12, derandomize=True, deadline=None)
