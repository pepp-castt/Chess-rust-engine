# EXP-M1: datagen shards were byte-identical; own-engine throughput measured

Agent M. Base: `main` @ 6ce2055. Box: 4 vCPU sandbox, release build, `bench 10` = 469049 nodes @ 2.74 Mnps.

## Finding 1 (bug): `datagen` is deterministic and hard-seeded
`datagen::run` used `Rng(0xC0FF_EE12_3456_789A)` unconditionally and the search is single-threaded and deterministic.
Four parallel processes (`datagen 20000 6 out_i.jsonl`) wrote **four byte-identical files** (same md5 `8aa75dd9...`).
A matrix of N shards therefore yields one shard's worth of unique data, not N.

Fix: optional `CRE_DATAGEN_SEED=<n>`. Unset = historical seed, **output bit-identical to before** (md5 unchanged, verified).
With seeds 1..4: four different md5s, **77,403 unique of 80,000** lines (3.2% duplicates from shared book prefixes / transpositions).
`cargo test --release`: all pass.

## Finding 2 (measurement): own-engine data throughput
`datagen 20000 6`: 18,855 positions kept in the first stage, **~793 positions/s per process** (depth 6 HCE self-play, same limits as H's stage-0).
Teacher comparison from the thread: SF19 MAX at 500 ms = ~1.2 labelled pos/s/proc (T, CRE#1).
Ratio ~660x per process. 20 shards x 4 procs x 1 h at 793/s would be ~228M positions; the 500 ms Stockfish path gives ~0.35M for the same compute.

## What this does and does not show
- It does NOT show a net trained on that data beats HCE. H's 1.12M-position stage-0 net lost (0-2-52 at 54 games).
- It does show the "10^7-10^8 positions" regime that T says is needed is reachable with own-engine data in hours, not weeks,
  *provided shards are seeded differently*. Before this PR that regime would have silently produced duplicates.
- Hypothesis for the next experiment (not run): H's net showed no train/val divergence after 15 min, which indicates underfitting,
  not "data volume is irrelevant". Test = same trainer, 20M+ unique positions, equal wall-clock per epoch, same 100-game gate vs HCE.
  Keep/revert on the gate result only.
