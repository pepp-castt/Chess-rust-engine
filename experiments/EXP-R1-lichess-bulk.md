# EXP-R1: bulk Stockfish labels (Lichess CC0 evaluation DB) for the NNUE wall

Agent R. Status: **pre-registered, result pending.** Nothing here is an Elo claim.

## Question
Do nets lose to the HCE because of *data volume*, or because of something structural
(trainer, features, search integration)? Every net so far (E, H, G, A) used <= 1.12M positions.
A's H=128 net had ~3 samples/parameter. Nobody has run the same pipeline at >= 10M.

## Why this data
`Lichess/chess-position-evaluations` (CC0-1.0): 416M unique positions, Stockfish MultiPV evals,
typically depth 20-60+ (much deeper than a 500 ms single-thread label), free to download in bulk.
The 500 ms floor protects *games* from bad moves. A label is never played; it needs to be
accurate and plentiful. (Same argument as P's price list, taken to its end: 10M labels from the
SF19 pipeline would be ~1,400 core-hours, which we do not have.)

Honest limits, stated up front:
* labels are **not** from the pinned SF19 (various SF versions, browser-run, varying depth);
* distribution is human-game positions from Lichess analysis (opening/middlegame heavy; median
  28 pieces in the first shard), not our engine's search distribution;
* no game result -> `wdl` unknown -> `--lam 1.0` (pure eval target);
* castling/ep not in the 28-byte record (same as all sources here).

## Pipeline (all in this PR)
`tools/nnue/lichess_to_records.py` -> `tools/nnue/train.py --lam 1.0` (unchanged) ->
`tools/gate_net.sh` (unchanged). Workflow: `.github/workflows/lichess-bulk.yml`.

Verified before opening the PR:
* `--selftest`: packer is **byte-identical** to `datagen_movetime.pack` and `in_check` agrees
  with python-chess on 17,644 random positions.
* **Score perspective is empirical, not assumed:** on 200k rows with |material| >= 5 and
  |cp| >= 150, sign(cp) agrees with material from *White's* view 86.1% vs 55.1% from the mover's
  view -> `cp` is White-POV (`lichess_pov_check.py`, also a CI guard in the workflow).
* Filter stats on the first 3.24M rows (603,787 FENs): mate 16%, depth<20 11%, |cp|>1500 1.7%,
  side-to-move in check 4.3%, best move capture/promotion 16.7%; 300k kept in 11 s (~27k pos/s/core).
  Kept set: 52.6% |score|<50, 15.9% |score|>300, mover-material vs score correlation 0.37.

## Pre-registered protocol
* Arm R-10M: shards 0,1 (up to 8M each), min depth 20, H=128 trainer, 150 min.
* Gate: `gate_net.sh`, net vs HCE, 100 games, 10+0.1, same binary, book, colours reversed.
* Validation: trainer's position-hash split (position-disjoint). Lichess FENs are independent
  positions (no game adjacency), so the block-leak that H/I found does not apply here.
* **KEEP** the direction if net scores >= 40% vs HCE (a large improvement over the 0-10%
  everyone saw; a *merge* still needs net >= 50% and a passing SPRT on `main`).
* **If still < 15%**: volume is not the explanation. Next suspects, in order: (1) HCE-anchored
  residual net (Agent I's EXP-I1 direction), (2) feature set (king buckets / HalfKA), (3) the
  eval-scale mismatch between net output (cp scale 400) and the search's pruning margins.
* Either result gets posted in the channel as measured.

## Next if it works
Pretrain here, then fine-tune on SF19 MAX labels (`nnue-finetune.yml`) and self-play positions
with search-score relabelling; grow H and add king buckets only after the small net clears the gate.
