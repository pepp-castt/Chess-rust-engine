# EXP-Y1: node-identical speedups (Agent Y, systems lane)

**Hypothesis.** Part of the NPS gap to Reckless-class engines is plain systems work,
and it can be gained with *zero* search/eval change, so no SPRT is needed for the
change to be strength-neutral per node; every extra node per second is free depth.

**Rule.** A patch only counts if `bench 8` (178914) and `bench 10` (469049) and a
fixed-depth search stay **node-identical**. Elo is then purely NPS.

## Measurements (same box, 1 thread, Hash 512, interleaved runs, depth 20, middlegame FEN)

| build | nodes | time | Mnps |
| --- | --- | --- | --- |
| upstream `7793ad7` | 8,742,867 | 6.14 s / 6.05 s | 1.42 / 1.45 |
| this PR | 8,742,867 | 5.22 s / 5.16 s | 1.68 / 1.69 |

**+17.5 % NPS, identical nodes.**  `bench 8`: 178914 = 178914. `bench 10`: 469049 = 469049.

Attribution (each measured separately on the earlier base, same direction):

| change | effect |
| --- | --- |
| eval: attack set computed once (was twice: mobility + king safety), bitboard doubled/isolated pawns | ~ +8-10 % |
| TT prefetch right after `make` | ~ +3 % (larger with big Hash) |
| `.cargo/config.toml` target-cpu=x86-64-v2 (hardware POPCNT) | ~ +4-5 % |

Equivalence proof: `eval::tests::optimized_hce_is_bit_identical_to_reference` keeps the
original `evaluate_hce_with` as `evaluate_hce_with_reference` and compares it on
>50k random-walk positions (8 openings/endgames x 60 walks x 120 plies), with both
`DEFAULT_PARAMS` and a randomly perturbed parameter vector (what the Texel tuner feeds).

## Tried and REVERTED (negative result, kept on record)

* **Lazy move selection** (score once, pick best remaining per iteration, same stable
  order as the insertion sort, nodes identical): no gain, within noise of slightly
  slower. Most nodes visit many moves; the shift cost cancels the saved sorting.
  Not included.

## Also tried and NOT shipped

* **PGO** (`-Cprofile-generate` on bench 11 + 4 depth-14 searches, then `-Cprofile-use`):
  node-identical, but A/B runs were within noise (5.94 / 5.48 s vs 5.60 / 5.54 s, depth 20).
  The code is already branch-light and LTO + codegen-units=1; not worth CI complexity.

## Workload note (honest numbers)

The depth-20 single-position test above is TT-heavy and shows +17.5 %. The 12-position
`bench 11` (16 MB TT, cleared per position) shows a smaller gain for this PR alone:
interleaved medians main 1.86 -> 1.99 Mnps (**+7.2 %**). Real games sit between the two.
Stacked with PR for EXP-Y2 (captures-only quiescence generator): 2.15 Mnps (**+15.6 %**).

## What this is NOT

A strength claim vs Stockfish. +17 % NPS is on the order of +10-15 Elo at this level.
It is a small, safe, merge-now gain. The large gains remain the NNUE / data work in
other lanes; this lane just stops leaving nodes on the table.

## Next in this lane (not done)

* PGO build in CI (`cargo pgo` / llvm-profdata); expected another 5-10 % NPS.
* Incremental NNUE accumulator (`nnue::evaluate` recomputes both accumulators from
  scratch every node; any net that ever beats HCE will be NPS-starved without this).
