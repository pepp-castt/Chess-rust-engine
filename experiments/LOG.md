# Experiment log

One entry per experiment: hypothesis, setup, result with uncertainty, decision.
Never record an Elo number that the sample size cannot support.

## EXP-001  Baseline vs Stockfish 19 MAX + weakness attribution  (Agent D, 2026-11-09)

**Engine:** `6ce2055` (main: HCE + singular ext + pawn corrhist, no net). **Teacher/opponent:** official
Stockfish 19 (`sha256 0f83d24c...da19`, `stockfish-linux-x86-64-universal`), `UCI_LimitStrength=false`,
Threads=1, Hash=64 for play; Threads=3, Hash=128, MultiPV 4, 1.0 s/position for labelling.
**Hardware:** 4 vCPU x86_64 sandbox, 2 games in parallel (2 procs each, equal for both engines).
**Games:** 24, 10+0.1 real clocks, 12 openings from `tools/openings_book.epd`, each played with both colours,
`tools/play_max.py --seed 11`.

### Result
| | W | D | L | score |
|---|---|---|---|---|
| NextGen vs SF19 MAX | 0 | 0 | 24 | 0.0% |

95% Wilson upper bound on the true score is 13.8% (about -318 Elo vs SF19 MAX at this TC). That is a
*ceiling*, not an estimate; 0/24 does not say how far below. Do not quote an absolute Elo from this.

### Where the losses come from (74 costly mistakes >= 80 cp, 362 labelled positions, decided positions excluded)
| class | n | share | mean loss |
|---|---|---|---|
| eval (static eval prefers played move, never converges to SF-acceptable move to +8 plies) | 22 | 30% | 143 cp |
| history-variance (clean-FEN replay at same budget plays an acceptable move) | 18 | 24% | 167 cp |
| search-pruning (acceptable move stable only at +3..+8 plies) | 13 | 18% | 148 cp |
| search-horizon (eval agrees with SF, tactic beyond +8) | 13 | 18% | 167 cp |
| search-depth (stable within +2 plies) | 8 | 11% | 173 cp |

58/74 mistakes are in the opening (fullmove <= 12) and 18 of the 22 eval-class ones too.

### Self-correction (kept in the log on purpose)
The first classifier counted a position as "search-depth" if ANY rung of the depth ladder hit an acceptable
move. That gave 27% search-depth and said ~51% of the mistakes were speed/time. A controlled check disagreed:
the same 74 positions at 0.5 s vs 3.0 s per move (6x time, about +2.6 plies) solved 22/74 = 29.7%
[20.5, 40.9] vs 27/74 = 36.5% [26.4, 47.9]: overlapping CIs, +5 positions. Flip-flopping between rungs had been
counted as convergence. The classifier now requires a *stable* solve (acceptable at depth d and every deeper
rung); the table above is the corrected one. Raw first-pass data is not kept; `--reclassify` reproduces the table.

### Conclusion / decision
At this strength the bottleneck is **knowledge, not speed**: 30% eval + 36% pruning/horizon vs 11% pure depth.
More NPS or time management will not close the gap to SF19; a net trained on >= 500 ms SF19 labels and better
pruning/ordering will. This supports the NNUE-first plan already agreed in CRE#1.

### Measurable follow-ups (each is keep/revert by evidence)
| id | hypothesis | metric | keep if |
|---|---|---|---|
| H1 | Oversampling `hard_positions.tsv` (opening-heavy) in the stage-2 net fixes eval-class errors | hard-suite solve rate @0.5 s | >= +10 pts AND SPRT vs HCE passes |
| H2 | The 24% non-reproducible mistakes come from TT/history carry-over or time allocation, not eval | replay *with full move history* vs clean FEN; compare reproduction rate | rate differs -> investigate `tt.rs` replacement / `time.rs` |
| H3 | Pruning/horizon cases improve with SPRT-validated LMR/NMP/futility changes | solved at +3..+8 plies class shrinks; SPRT | SPRT H1 accepted |
| H4 | Time-management change targets the 11% only | n/a | low priority |

### Reproduce
```bash
cargo build --release
python3 tools/play_max.py --stockfish ./stockfish --games 24 --workers 2 --tc 10+0.1 --seed 11 --pgn results/max.pgn
python3 tools/diagnose_weakness.py results/max.pgn --stockfish ./stockfish --ours target/release/chess-rust-engine \
        --movetime 1.0 --multipv 4 --threads 3 --out-dir results/diagnosis
python3 tools/epd_suite.py results/diagnosis/hard_suite.epd --engine ng=target/release/chess-rust-engine --movetime 0.5
```
Or: Actions -> **Weakness loop** -> workflow_dispatch.

Caveats: single engine revision, 24 games, replay starts from FEN (no repetition history), the EPD suite is
built from this engine's own losses (biased towards its weaknesses; a smoke alarm, never a merge gate).
