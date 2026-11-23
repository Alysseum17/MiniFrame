import sys
from pathlib import Path
from random import Random

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import miniframe as mf
from miniframe.column import Column
from miniframe.ops import sort
from miniframe.ops.join import hash_join_pairs, sort_merge_join_pairs

CASES = [
    "sort",
    "hash_join",
    "merge_join",
    "group_8",
    "group_many",
    "rolling_min_16",
    "rolling_min_256",
    "rolling_sum_256",
    "naive_sum_256",
    "mf1_read",
    "mf1_write",
]


@pytest.mark.parametrize("size", [2048, 4096])
@pytest.mark.parametrize("case", CASES)
def test_scaling(benchmark: BenchmarkFixture, tmp_path: Path, size: int, case: str) -> None:
    rng = Random(17)
    keys = list(range(size // 2)) * 2
    rng.shuffle(keys)
    other = keys.copy()
    rng.shuffle(other)
    values = [rng.randrange(-100, 101) for _ in keys]
    df = mf.DataFrame({"k": keys, "v": values})
    if case == "sort":
        fn, args, kwargs = df.sort, ("k",), {}
    elif case in ("hash_join", "merge_join"):
        fn = hash_join_pairs if case == "hash_join" else sort_merge_join_pairs
        args, kwargs = ([df["k"]._column], [Column(other, mf.Int64)]), {"how": "inner"}
    elif case.startswith("group"):
        groups = [k % 8 for k in keys] if case == "group_8" else keys
        grouped = mf.DataFrame({"g": groups, "v": values}).group_by("g")
        fn, args, kwargs = grouped.agg, (mf.col("v").sum(),), {}
    elif case.startswith("rolling"):
        expr = {
            "rolling_min_16": mf.col("v").rolling_min(16, 1),
            "rolling_min_256": mf.col("v").rolling_min(256, 1),
            "rolling_sum_256": mf.col("v").rolling_sum(256, 1),
        }[case]
        fn, args, kwargs = df.select, (expr,), {}
    elif case == "naive_sum_256":

        def naive() -> list[int]:
            return [sum(values[max(0, i - 255) : i + 1]) for i in range(size)]

        fn, args, kwargs = naive, (), {}
    else:
        path = tmp_path / "frame.mf"
        mf.write_mf(df, path)
        fn, args, kwargs = (
            (mf.read_mf, (path,), {}) if case == "mf1_read" else (mf.write_mf, (df, path), {})
        )
    result = benchmark.pedantic(
        fn, args=args, kwargs=kwargs, rounds=9, iterations=1, warmup_rounds=2
    )
    if case.endswith("join"):
        assert len(result) == 2 * size
    elif case.startswith("group"):
        assert result.height == (8 if case == "group_8" else size // 2)
    elif case == "naive_sum_256":
        assert result[-1] == sum(values[-256:])
    elif case == "mf1_write":
        assert mf.read_mf(path).equals(df)
    else:
        assert result.height == size


def test_resident_int64_storage() -> None:
    values = list(range(1000, 101000))
    column = Column(values, mf.Int64)
    packed = sum(
        sys.getsizeof(x) for x in (column, column._values, column._validity, column._validity._data)
    )
    boxed = sys.getsizeof(values) + sum(map(sys.getsizeof, values))
    assert column.to_list() == values
    print(f"100000 unique integers: packed={packed} list={boxed}")


def test_sort_comparison_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    compare = sort._compare_rows
    calls = 0

    def counted(keys: list[sort._Key], left: int, right: int) -> int:
        nonlocal calls
        calls += 1
        return compare(keys, left, right)

    monkeypatch.setattr(sort, "_compare_rows", counted)
    for size in (2048, 4096):
        values = list(range(size))
        Random(17).shuffle(values)
        calls = 0
        assert sort.argsort([Column(values, mf.Int64)]) == sorted(
            range(size), key=values.__getitem__
        )
        assert 0 < calls <= size * (size - 1).bit_length()
        print(f"sort rows={size} comparisons={calls}")
