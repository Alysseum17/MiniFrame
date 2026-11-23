# Stage 1 fault injection

All eleven applicable faults were experimentally detected. Each row below had a passing control on the original code and a failing pytest assertion after changing a single behavior in an isolated source copy. Each run used a fresh Python process, the local source tree, the mutation Hypothesis profile and `pytest -xq`. The runner restored the source after each experiment. No injected implementation is included in the package.

| # | Temporary change | Detecting test in `tests/` | Observed failure |
|---|---|---|---|
| 1 | Address the preceding bit for positive indices divisible by eight | `test_bitmap.py::test_lsb_first_layout_across_byte_boundaries` | Packed bytes became `85 80 00`, expected `85 81 01`. |
| 2 | Require left validity before a false right operand can determine AND | `test_logic.py::test_binary_truth_tables[None-False-False-None]` | The null count became 1 for a result that must be valid False. |
| 3 | Choose the right merge run on ties by changing the comparison from `<= 0` to `< 0` | `test_sort.py::test_stable_indices_and_null_placement` | Tied row IDs 1 and 5 changed order. |
| 4 | Implement descending order by reversing an ascending permutation, adjusting null placement | `test_sort.py::test_stable_indices_and_null_placement` | Ties reversed even though nulls were placed at the requested end. |
| 5 | Include null components in hash keys | `test_join.py::test_exact_order_and_schema` | Inner join emitted the extra pair (2, 2). |
| 6 | Skip each left row with more than one right match in a left join | `test_join.py::test_exact_order_and_schema` | Only the unmatched rows (2, None) and (3, None) remained. All four matched pairs disappeared. |
| 7 | Remove groups with any null key component from the output | `test_groupby.py::test_composite_keys_and_representatives` | The ordered result differed from the complete expected frame. |
| 8 | Accumulate squared old-mean deltas in Welford's second moment | `test_aggregate.py::test_welford_against_exact_rational_oracle` | Sample variance was about 8.51 × 10³⁷, expected 0.5. |
| 9 | Leave expired front indices in the rolling extrema deque | `test_window.py::test_text_extrema_expire_in_logical_order` | At row 6, the expired value “a” replaced the expected “猫”. |
| 10 | Remove the absolute-offset check while retaining length limits | `test_mf_format.py::test_header_structure` | The reader accepted a checksum-valid descriptor with offset 18 inside the header. |
| 11 | Bypass the per-buffer CRC check | `test_mf_format.py::test_checksums_footer_and_extra_bytes` | The reader accepted a flipped Boolean value bit that still satisfied every physical invariant. |
| 12 | Push a right-side predicate across a left join in a lazy plan | N/A — LazyFrame not implemented | No experiment applies to the eager engine. |

These targeted experiments use separate faults from the generated mutmut population. They contribute zero observations to the reported mutation score.
