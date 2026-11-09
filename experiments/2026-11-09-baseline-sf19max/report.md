# NextGen weakness diagnosis

- teacher: **Stockfish 19** (full strength, 1.0 s/position, MultiPV 4)
- engine: Chess-rust-engine 0.1.0 @ `6ce2055`
- games analysed: 24, our positions labelled: 362, costly mistakes (>= 80 cp): **74**
- host: 4 cpus x86_64, wall 1700.7 s

## Attribution

| class | n | share | mean loss (cp) | what to do |
|---|---|---|---|---|
| eval | 22 | 30% | 143 | Static eval prefers the played move even after deeper search: train/tune the evaluation (NNUE data should include these positions). |
| history-variance | 18 | 24% | 167 | Same budget finds an acceptable move from a clean state: look at TT/history pollution, time management (soft/hard limits), instability handling. |
| search-pruning | 13 | 18% | 148 | Needs 3-8 more plies: ordering / LMR / NMP / futility are cutting the line. Candidate for SPRT on reduction tweaks + corrhist. |
| search-horizon | 13 | 18% | 167 | Eval agrees with SF but the tactic is beyond +8 plies: extensions (singular/check), quiescence, SEE. |
| search-depth | 8 | 11% | 173 | Engine finds it 1-2 plies deeper: NPS, time allocation, or too-aggressive stopping when the score drops (extend time on fail-low). |

## By phase

| phase | class | n |
|---|---|---|
| middlegame | eval | 4 |
| middlegame | history-variance | 4 |
| middlegame | search-depth | 2 |
| middlegame | search-horizon | 3 |
| middlegame | search-pruning | 3 |
| opening | eval | 18 |
| opening | history-variance | 14 |
| opening | search-depth | 6 |
| opening | search-horizon | 10 |
| opening | search-pruning | 10 |

## Ten costliest

| loss | class | phase | played | SF best | solve depth (orig) | FEN |
|---|---|---|---|---|---|---|
| 368 | history-variance | opening | h8f8 | d7d6 | 16 (16) | `r1bqk2r/pp1pppbp/5np1/8/2PQ4/2N3P1/PP2PPBP/R1B1K2R b KQkq - 0 4` |
| 303 | eval | opening | a1b1 | f3h4 | None (14) | `r4rk1/pp2ppbp/2np1np1/2p2q2/P1PP4/1PN1PNPb/5PB1/R1BQ1RK1 w - - 0 9` |
| 275 | search-horizon | middlegame | d1c1 | d2g5 | None (14) | `1r1q1rk1/3n1pbp/b2pp1p1/8/pnN1P3/3P1NP1/1P1BQPBP/3R1RK1 w - - 4 14` |
| 270 | search-depth | opening | g2f3 | d1f3 | 18 (16) | `r2qk2r/ppp2ppp/1bnp1n2/4p3/1PP1P3/P1NB1b1P/3P1PP1/R1BQK2R w KQkq - 0 7` |
| 269 | search-depth | opening | d6d5 | a7a6 | 15 (13) | `2rqkb1r/pp1b1p2/3ppnpp/8/4P1PP/1PN1BP2/1PPQ4/2KR1B1R b k - 0 9` |
| 258 | history-variance | opening | h1f1 | c2c3 | None (14) | `2kr3r/pp1n2pp/2p1b3/2bqppN1/8/3B1P2/PPPPQ1PP/R1B1K2R w KQ - 5 10` |
| 253 | eval | opening | c6c5 | f6h4 | None (15) | `rnbq1rk1/pp3p1p/2p2b2/3p4/3P4/2NQ4/PPP1NPPP/R1B1K2R b KQ - 1 8` |
| 235 | search-pruning | opening | d8f6 | g7h6 | 20 (14) | `r1bq1rk1/pp1p2b1/6pB/2pP4/3p4/3P2P1/PPP2PB1/R2QK2R b KQ - 0 9` |
| 234 | search-horizon | middlegame | a1b1 | c4e2 | None (12) | `r2r2k1/1pq1bppp/p1b1pn2/8/P1Q5/2N1PP1P/1P1N2P1/R1BR2K1 w - - 1 14` |
| 232 | search-horizon | opening | f4f5 | e1g1 | None (14) | `r1b1k2r/p3ppbp/2p1q1p1/3np3/2B2P2/1PN4P/P1PB2P1/R2QK2R w KQkq - 2 10` |

> Small-n warning: classes are hypotheses. A change is kept only if the EPD suite AND an SPRT agree.
