# MiniFrame Stage 1 quality audit

**Stage 1 engineering milestone reached. Strict design-document MVP incomplete.** The working engine contains 2,573 conventional Python code lines. The stricter exclusions give a conservative estimate of about 1,450 algorithmic lines. Branch coverage exceeds 90%. The full mutation gate, CSV API and 3,000-line algorithmic threshold remain open.

## Engine and tests

The engine implements the five nullable physical DTypes, packed validity and Boolean values, typed kernels, expression evaluation and eager frames. Relational operators include stable multi-key merge sort, first-seen hash grouping, hash and sort-merge joins, and grouped and ordered windows. MF1 stores validated column buffers with deterministic encoding and checksums.

| Layer | Evidence |
|---|---|
| Unit and negative | Exact truth tables, strict casts, Int64 overflow, physical normalization, schema and shape errors, and zero-width frames. Generated failure sequences now check that subsequent operations can reuse the original inputs. |
| Property and metamorphic | Existing tests compare sorting to stable permutations, join pairs to nested loops, group partitions to Python and windows to naive scans. New properties check filter commutation and the true/false/null partition, inverse shifts on retained positions, and physical immutability. |
| Differential | Random composite full joins followed by a grouped count compare every ordered output row to **both** Polars and DuckDB. Generated predicate trees compare a two-filter pipeline to Polars and a scalar row evaluator. Existing tests also use Arrow buffers/kernels and exact rational sample-variance oracles. |
| Frame state machine | Twelve transitions exercise filter, select, with_columns, sort, drop, rename, unique, drop_nulls, vstack, grouping, left joins and MF1 persistence. An independent Python row list preserves multiplicity, row order and zero-column height. Ordered schema, float bits, invariants and the previous frame's physical bytes are checked after each transition. Failed MF1 reads may intervene before the same file is rewritten and read successfully. |
| Column state machine | A list model follows takes, slices, concatenations, strict casts and null fills. Initial lengths include 7, 8, 9, 16 and 17. Checks include physical validity, canonical placeholders, UTF-8 offsets, category code interpretation and prior-buffer snapshots. Both machines use 12 stateful steps and inherit the active example budget. |
| Fuzz | The fixed MF1 fixture exercises every byte replacement and truncated prefixes. Other tests cover arbitrary bytes, insertion, deletion, reordered payload, checksummed structural damage and tracemalloc bounds on extreme claims. These are bounded samples of the input space. |
| Mutation, fault injection and performance | Generated mutants, hand-injected faults and separate opt-in benchmarks have distinct denominators and results below. |

**Defect found and fixed.** Section 4.5 names an expression after its first referenced column unless it has an outer alias. In `lit(1).sum().over("key")`, the original name traversal ignored the grouping keys and produced `literal`. This could let a duplicate name through a projection. The regression failed before the fix. The traversal now checks the reduction followed by its keys in order, including when an earlier key is a literal. Expressions made entirely of literals retain the fallback name.

**Oracle conventions.** Join comparisons keep both row IDs, sort the oracle by left ID then right ID with nulls last, and coalesce the keys for right-only rows. Repeated matches remain separate rows. Composite keys containing null never match. NaNs match and group together, signed zeros compare equal as keys, and float value-preservation checks compare raw bits. SQL and Polars window counts include every row. GroupBy checks use first-seen order. Integer overflow, IEEE elementwise NaN comparison and SQL's empty sum follow the deviations in design section 12. Exact scalar checks cover the passing pipelines here. Existing approximate comparisons are confined to numerical reductions that require rounding.

## Execution

Linux x86-64, CPython 3.14.10, uv 0.13.8, pytest 9.2.0, Ruff 0.17.6, Hypothesis 6.170.17, coverage.py 7.17.3, mutmut 3.8.1 and pytest-benchmark 5.3.1. Installed oracle versions were Polars 2.1.0, DuckDB 2.0.0 and PyArrow 26.0.1. pandas 3.1.1 was installed. Runtime dependencies are empty. The Python pin, packaging constraint, Ruff target, lockfile and CI Python selector already target 3.14. Offline locked sync and lockfile verification succeeded.

| Command | Result |
|---|---|
| `uv run ruff check .` | Passed |
| `uv run ruff format --check .` | All 52 files already formatted |
| `uv run pytest` | 1,088 passed in 31.93 s |
| `uv run pytest --cov=miniframe --cov-branch --cov-report=term-missing` | 1,088 passed in 32.22 s |
| `uv run pytest -q tests/test_stateful_{frame,column}.py tests/test_metamorphic.py tests/test_regressions.py` | 7 passed in 7.50 s |
| `HYPOTHESIS_PROFILE=ci uv run pytest` | 1,088 passed in 142.34 s |
| Separate benchmarks | 23 passed in 5.86 s on the first run. After adding the comparison-count check, all 24 passed in 5.10 s. |

The development, coverage and CI-profile runs had zero skips or xfails. The registered dev, ci and mutation profiles were inspected. Their example counts are 100, 500 and 30. Mutation sets derandomize to true and deadline to None. The existing join property retains its 120-example development budget and inherits the CI and mutation budgets. Benchmarks require explicit selection. The CI workflow uses the CI profile. For a longer standalone MF1 fuzz run use `HYPOTHESIS_PROFILE=ci uv run pytest tests/test_mf_format.py`. The development checks include the exhaustive small-fixture corruption test.

## Coverage and source audit

Branch-only coverage is **860/862 (99.77%)**. All 2,260 executable statements ran. The only missing arcs are exits at lines 40 and 55 of dtype.py, where exhaustive matches over the five declared Enum members could fall through for a hypothetical additional member. The audit found no uncovered reachable branches in the eleven named critical modules. No coverage exclusions were added.

| Module under miniframe | Branches hit / total | cloc code | Conservative algorithmic proxy |
|---|---:|---:|---:|
| bitmap.py | 30/30 | 109 | 68 |
| column.py | 80/80 | 275 | 178 |
| dtype.py | 16/18 | 44 | 11 |
| errors.py | 0/0 | 6 | 0 |
| eval.py | 82/82 | 189 | 116 |
| expr.py | 42/42 | 243 | 42 |
| frame.py | 80/80 | 258 | 153 |
| series.py | 28/28 | 156 | 43 |
| kernels/aggregate.py | 50/50 | 159 | 102 |
| kernels/arithmetic.py | 34/34 | 80 | 53 |
| kernels/cast.py | 34/34 | 69 | 55 |
| kernels/compare.py | 26/26 | 61 | 37 |
| kernels/logic.py | 4/4 | 26 | 17 |
| ops/groupby.py | 48/48 | 146 | 58 |
| ops/join.py | 78/78 | 182 | 127 |
| ops/sort.py | 40/40 | 80 | 63 |
| ops/window.py | 82/82 | 172 | 126 |
| io/mf_format.py | 106/106 | 282 | 217 |
| Package initializers (root 34 code lines, io 2, kernels and ops 0) | 0/0 | 36 | 0 |
| **Total** | **860/862** | **2,573** | **1,466** |

`cloc --by-file miniframe` and `cloc tests` used cloc 2.10 with its default Python docstring classification. The package contains **3,328 physical lines** (608 blank, 147 comment/docstring, 2,573 code). Tests including benchmarks contain **5,767 physical lines** (740 blank, 12 comment/docstring, 5,015 code). Across the whole project there are **11,029 physical lines across 58 text files**, including 1,767 blank lines. cloc recognizes 55 files in that inventory and counts 1,728 blank, 159 comment, 8,495 code lines. It omits the lockfile and two dotfiles from its language totals. This whole-tree tally includes documentation, CI, configuration and the lockfile, and excludes binary images, caches, the environment, Git metadata and mutation output.

The last table column is a reproducible lower-end proxy for a subjective classification. Using the Python AST, take the union of physical body lines of functions with at least two non-docstring statements, excluding bodies consisting solely of assignments. Exclude definitions and signatures, decorators, imports, type aliases, docstrings and lines containing no alphanumeric characters. Count a nested body line once. This omits forwarding, most expression builders and even some substantial one-expression implementations. Counting only executable statement-start lines under the same function selection gives 1,339. Neither count can establish the strict 3,000-line gate. Even the entire package before algorithmic exclusions contains fewer than 3,000 cloc code lines.

## Mutation evidence

mutmut 3.8.1 generated **2,026** mutants across the eight configured modules. The experiment completed all mutants in two modules and all generated mutants of one selected function in each other module. The raw score counts only pytest failures as kills and includes equivalent survivors in the denominator. Unexecuted mutants have no result.

| Module and measured scope | Generated in module | Executed | Killed | Survived | Equivalent among survivors | Final timeout/error | Raw score |
|---|---:|---:|---:|---:|---:|---:|---:|
| kernels/logic, whole module | 52 | 52 | 51 | 1 | 1 | 0 | 51/52 = 98.08% |
| kernels/compare, whole module | 119 | 119 | 119 | 0 | 0 | 0 | 119/119 = 100% |
| bitmap, Bitmap._position | 213 | 17 | 17 | 0 | 0 | 0 | 17/17 = 100% |
| ops/sort, _compare_rows | 166 | 38 | 29 | 9 | 9 | 0 | 29/38 = 76.32% |
| ops/join, _hash_key | 357 | 7 | 7 | 0 | 0 | 0 | 7/7 = 100% |
| ops/groupby, group_ids | 216 | 10 | 10 | 0 | 0 | 0 | 10/10 = 100% |
| ops/window, _dominates | 353 | 15 | 12 | 3 | 3 | 0 | 12/15 = 80% |
| io/mf_format, _validate_layout | 550 | 34 | 31 | 3 | 3 | 0 | 31/34 = 91.18% |
| **Measured population** | **2,026** | **292** | **276** | **16** | **16** | **0** | **276/292 = 94.52%** |

The first comparison run killed 104/119. All fifteen survivors changed error diagnostics, including which operand type was reported. Exact assertions on existing negative cases killed all fifteen on rerun. Three bitmap bounds-message mutations also fell to strengthened assertions. In the initial extended run fifteen mutants timed out under the default fork runner. Replaying those IDs with forkserver isolation and focused test nodes produced three kills and twelve survivors. Timeouts receive no kill credit. Excluding the sixteen reviewed equivalents gives 276/276 for the measured subset. **1,734 mutants remain unexecuted.** The eight-module 80% gate is unverified.

| Surviving function and mutmut suffixes | Reason for equivalence on supported inputs |
|---|---|
| logic._boolean_buffers 6 | The annotation argument to typing.cast changes. The identical bitmap object is returned at runtime. |
| sort._compare_rows 7, 9, 14, 15, 22, 27, 32, 34 | Each changes the magnitude of a positive or negative result from one to two. The sole consumer tests its sign with `<= 0`. Direction reversal preserves that sign distinction. |
| sort._compare_rows 33 | Changes less-than to less-than-or-equal after equal keys have already continued. Nulls and NaNs have their own preceding branches. |
| window._dominates 4 | The first argument of typing.cast changes. The previous floating-point value passed to the comparator is identical. |
| window._dominates 14, 15 | The non-float comparison removes an older equal candidate instead of retaining it. Either candidate emits the same Int64 or text value. The newer equal candidate expires later and remains available whenever the older one would have contributed. Valid-count tracking is separate. Float ties, including signed zero, use a different branch. |
| mf_format._validate_layout 10, 13, 14 | Relax the strict zip. The parser derives the descriptor count from the same dtype, flag and row count as the expected lengths. Both lists therefore have equal length on every path from the public reader, including malformed file inputs reaching this function. |

To reproduce the selected production scopes with the final configuration and the current suite, run the following from the repository root on Linux. mutmut writes its generated copies under the ignored mutants directory.

```sh
export HYPOTHESIS_PROFILE=mutation
uv run mutmut run --max-children 3 'miniframe.kernels.logic.*' 'miniframe.kernels.compare.*'
uv run mutmut run --max-children 3 'miniframe.bitmap.*_position__mutmut_*' \
  'miniframe.ops.sort.x__compare_rows__mutmut_*' 'miniframe.ops.join.x__hash_key__mutmut_*' \
  'miniframe.ops.groupby.x_group_ids__mutmut_*' 'miniframe.ops.window.x__dominates__mutmut_*' \
  'miniframe.io.mf_format.x__validate_layout__mutmut_*'
uv run mutmut results
```

The measured runs used disposable source copies with pytest selection narrowed to test_logic, test_compare and test_bitmap for the first batch, and test_bitmap, test_sort, test_join, test_groupby, test_window and test_mf_format for the second. The fifteen timeout retries selected the invalid-index, stable null-placement, IEEE ties, multi-key flags, trailing-window, signed-zero extrema and expiring-text extrema tests. The final configuration uses forkserver with no pre-fork test collection. Only the mutation runner omits the exhaustive byte-replacement sweep. The initial extended run reported 0.12 mutants per second. For the full gate run `HYPOTHESIS_PROFILE=mutation uv run mutmut run --max-children 3` without name filters, then inspect survivors with `uv run mutmut show MUTANT_NAME`.

All eleven applicable design faults were killed by their paired control/changed-source experiments. Scenario 12 is **N/A — LazyFrame not implemented**. The fault definitions, pytest nodes and observed failures are recorded separately. [Injected-bug evidence](injected-bugs.md)

## Performance observations

The opt-in suite uses seeded inputs, nine single-call rounds after two warmups and median wall time. Both inputs to the joins have two rows per key, so the inner join emits exactly 2n pairs. GroupBy uses eight or n/2 groups. Frame sort includes row gathering. Rolling windows use a minimum of one valid observation. File timings include local filesystem operations and are sensitive to caching.

| Operation | 2,048 rows, ms | 4,096 rows, ms | 2n/n | Repeat 2n/n |
|---|---:|---:|---:|---:|
| Stable frame sort | 17.448 | 80.389 | 4.61 | 4.03 |
| Hash join pairs | 8.050 | 18.884 | 2.35 | 2.05 |
| Sort-merge join pairs | 32.679 | 69.699 | 2.13 | 2.13 |
| GroupBy, eight groups | 6.210 | 12.078 | 1.95 | 1.89 |
| GroupBy, n/2 groups | 9.448 | 18.562 | 1.96 | 1.85 |
| Rolling minimum, width 16 | 8.348 | 28.588 | 3.42 | 1.72 |
| Rolling minimum, width 256 | 14.325 | 30.556 | 2.13 | 1.69 |
| Rolling sum, width 256 | 12.719 | 13.558 | 1.07 | 2.00 |
| Naive sum of each 256-row slice | 11.612 | 27.181 | 2.34 | 2.09 |
| MF1 read | 1.993 | 3.829 | 1.92 | 2.21 |
| MF1 write | 2.511 | 3.856 | 1.54 | 1.98 |

GroupBy and fixed-multiplicity hash join timings are broadly consistent with linear work. The frame sort exceeded 4× on both sequential benchmark runs. The short-window timing ratio varied from 3.42× to 1.72×. A separate instrumented sort counted 19,909 and 43,961 comparisons (2.21×), both below n × ceil(log₂ n). The existing large-window regression checks a linear bound of five column reads per input row. These counts support the expected work on these data. A diagnostic shuffled the order of argsort, gather and whole-frame measurements at 2,048, 4,096 and 8,192 rows (seed 17, two warmups, eleven measured cycles). Median wall times in milliseconds were 11.04/23.77/50.74 for argsort, 6.41/11.49/22.46 for gather, and 15.20/34.16/83.55 for frame sort. The interleaved frame ratios were 2.25× and 2.45×. The sequential timing anomaly remains unexplained. These finite measurements cannot prove asymptotic complexity. No wall-time assertion runs in CI.

For 100,000 distinct integers above the small-int cache, resident object sizes were **829,497 bytes** for the Column and its value/validity storage and **3,600,056 bytes** for the list and its elements (4.34×). This is a resident-size comparison on the tested CPython build. It excludes allocator/process overhead and construction peaks. Reproduce timing and memory measurements with `uv run pytest tests/benchmarks/test_complexity.py -qs --benchmark-json=/tmp/miniframe-bench.json`.

## Remaining design requirements

CSV reading, writing and the corresponding CSV oracle/fuzz coverage remain absent from the mandatory Section 3.1 API. Some public operations still lack individual contract docstrings. The full mutation population, larger scale and skew benchmarks from Section 11, and the strict algorithmic LOC target remain open. LazyFrame is optional and outside Stage 1. The eager engine and measured quality gates support the original approximately 2,000-line working-code milestone, with the stricter MVP gates tracked independently.
