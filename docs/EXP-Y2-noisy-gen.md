# EXP-Y2: captures-only generator for quiescence (Agent Y, systems lane)

**Finding.** `noisy_moves` (called at every quiescence node, i.e. most nodes) ran the
*full* `pseudo_legal` generator and then filtered to captures/promotions: quiet
moves, double pushes and castling (with its `in_check` probe) were generated and
thrown away. Microbenchmark on 5 positions: `noisy_moves` **128 ns** vs full
`pseudo_legal` **84 ns** (filter pass included).

**Change.** One generic generator `gen::<NOISY>`; `NOISY = true` skips quiet pushes,
double pushes and castling, and masks piece destinations to enemy pieces. Moves are
emitted in the *same relative order* as the filter produced, so the stable sort sees
identical input and search results cannot change. Microbenchmark: **24 ns**.

**Evidence it is a pure speed change.**
* `movegen::tests::noisy_gen_equals_filtered_pseudo_legal`: exact vector equality with
  `pseudo_legal` filtered to captures/promos on >40k random-walk positions (promotions
  and en passant both exercised; asserted by the test).
* Nodes identical: `bench 8` 178914, `bench 10` 469049, `bench 11` 836287,
  depth-20 middlegame FEN 8,742,867.
* perft suite unchanged.

**Speed (interleaved medians of 6 runs, `bench 11`, 1 thread, Hash 16):**
main 1.86 Mnps -> this branch alone ~ +8 %; stacked on PR #25: 2.15 Mnps (+15.6 % vs main).

Not a strength claim; ~+15-20 % NPS is on the order of +10-15 Elo.
