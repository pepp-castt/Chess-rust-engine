# Texel distillation of the HCE (Agent I)

Fits the hand-crafted evaluation to **Stockfish 19 MAX** labels. Stockfish is a
label source only: no network, no code, nothing of it ships in the engine.

## Why this exists
Every NNUE attempt so far (E: -676 Elo, G/H stage-0: 0-2-66) loses to the HCE
because a net needs ~10^7-10^8 positions and our SF19 >=500 ms labels arrive at
~1M per 20-shard dispatch. Until a high-volume label source exists the HCE is the
engine we ship, so we fit it to the ~645k labels already paid for.
ROADMAP Phase 2 ("Texel-tune the HCE").

## Pieces
| file | role |
| --- | --- |
| `src/eval.rs` | every HCE number is one `[i32; NP]` vector (`P_*` indices). `evaluate_hce_with(pos, &params)` |
| `src/hce_params.in` | **the parameters the engine plays with** (`DEFAULT_PARAMS`, `include!`d) |
| `PESTO_PARAMS` (eval.rs) | the original constants. Never edited. Baseline + rollback + regularisation anchor |
| `src/bin/texel.rs` | the tuner (pure std, deterministic) |
| `tests/eval_params.rs` | colour-symmetry + entry-point equivalence tests; must stay green for any param set |

## Run
```bash
cargo build --release
./target/release/texel \
    --data teacher.bin,diverse.bin,perturbed.bin \   # 28-byte records, see tools/nnue/datagen_movetime.py
    --out tuned.params --emit-in tuned_hce_params.in \
    --steps 4,2,1 --passes 10 --reg 1e-8 --k 400
./target/release/texel --data ... --init tuned.params --eval-only   # re-score any param file
```

## What it optimises, and the guard rails
* loss = mean `(sigmoid(hce/K) - sigmoid(sf/K))^2`, both from the **side to move**,
  `sigmoid(x) = 1/(1+10^(-x/K))`, K=400. Game results are not used (D showed wdl is
  a quantised copy of the score here).
* **Validation = `pos_hash % 20 == 0`**, the same rule as `train.py`/`data_audit.py`.
  Those positions are never optimised on. The tuner keeps and returns the
  **best-validation** parameters, not the last (checkpoint + rollback).
* L2 pull towards PeSTO (`--reg`). King value, pawn PST ranks 1/8 and unused
  passed-pawn slots are frozen.
* Duplicates are dropped across all input files by `(piece hash, stm)`.
* Reports loss per |SF score| band and per data source, plus sign agreement on
  decisive positions, for both PeSTO and the tuned set.

## Promote / roll back
```bash
cp tuned_hce_params.in src/hce_params.in     # promote
git checkout <old> -- src/hce_params.in      # roll back (or: texel --dump-default --emit-in src/hce_params.in)
```
Promotion needs an `sprt.yml` PASS against the previous params. **A lower
validation MSE is not Elo.** MSE on SF labels is a proxy; the games decide.

## Honest limits
* The HCE can only represent what its features express; mobility/king-safety
  shapes are fixed. Texel moves weights, not structure.
* The labelled positions are quiet-ish game positions plus perturbed/imbalanced
  ones; no tactical (search-dependent) signal is learnable by a static eval.
* Labels are 500 ms SF19 scores, so the fit inherits SF's scale. Search margins
  (RFP, futility) were hand-set for PeSTO's scale and may want a retune by SPSA.
