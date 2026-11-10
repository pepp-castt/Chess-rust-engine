# EXP-I1: HCE-anchored residual net (`eval = HCE + net`) - NEGATIVE as a shipping eval

Agent I (residual). Branch `agentI/residual-net`, stacked on PR #8 (`agentB/nnue-h128`). Pre-registered rule
(posted on issue #1 before any run): **KEEP** iff residual-vs-HCE >= 50% over >= 100 games (10+0.1, 595/116-position
book, colours reversed) *and* it then survives `sprt.yml [0,10]`. Otherwise report negative.

## Hypothesis
From-scratch nets lose 90%+ to HCE (E: -676 Elo, H: 0-2-66) because they must re-learn what HCE already knows from ~1M
positions. If the net only predicts `target - HCE_static` with a zero-initialised output layer, step 0 is exactly HCE,
the cliff disappears, and the net spends capacity on where HCE is wrong.

## What was built (all opt-in; version-1 nets and HCE-only behaviour are unchanged)
- CRN1 header `version = 2` => residual net. `eval::evaluate` returns `HCE + clamp(net, +-cap)`.
- UCI `ResidualCap` (spin 0..3000, default 400; `CRE_DELTA_CAP` sets the initial value) so `sprt.yml` can set it via `test_opts`.
- `tools/nnue/hce_labels.py` (HCE static eval sidecar per record, via `hce-dump`), `train_residual.py` (reuses B's
  `decode`/`pos_hash`/held-out split; reports **HCE-only held-out loss as the baseline the net must beat**; rolls back
  if it does not), `diagnose_residual.py`, `gate_vs_hce.sh` (time-control or fixed-nodes gate, same binary).

## Verified properties (measured)
| check | result |
|---|---|
| `bench 8`, no net | 178914 nodes, identical to PR #8 -> default behaviour unchanged |
| zero **v2** net | **178914 nodes = HCE exactly** (the safety claim holds) |
| zero **v1** (replace-style) net | 64455 nodes: the engine is blind. A zero-init replace net *is* a cliff |
| trained v2, `ResidualCap=0` | 178914 nodes = HCE |
| NPS (same loaded box) | HCE ~2.3M, residual ~0.56-1.0M: full accumulator refresh + HCE per node |

## Data and training
`teacher-data/data/teacher-42595442579.bin`: 348,714 unique SF19 MAX >=500 ms positions, 74% draws. HCE vs label:
MAE 161 cp, corr 0.762. 341,738 train / 6,976 held-out (`hash % 50`). 8-minute budget, best-val checkpoint kept.

Held-out result (looks great, and is the trap):
- loss HCE 0.01178 -> **0.00331 (-71.9%)**, best at ~1 min, then val rises while train keeps falling (overfit).
- corr(label, eval) 0.762 -> 0.817; sign agreement on |SF|>200: 96.3% -> 99.6%. Mean |delta| 152 cp (p95 522).

## Game results (same binary, `UseNNUE` true vs false, same book, colours reversed)
Time control 10+0.1 (4 concurrent games on 4 vCPU, so absolute speed is noisy; both sides share the noise):

| run | cap | W-D-L | score |
|---|---|---|---|
| T1 (stopped early for futility at 52 games) | 400 | 6-6-40 | 17.3%, -272 Elo (95%: -421..-177) |
| cap-100 / cap-40 (stopped, far too few games) | 100 / 40 | 4-5-5 / 5-3-6 of 14 | not an Elo claim |

**Fixed nodes = 25,000/move, 60 rounds x 2 = 120 games per cap** (removes the NPS confound; both sides search the same budget):

| `ResidualCap` | W-D-L | score | Elo (95% CI) |
|---|---|---|---|
| 0 | - | 50% by construction (identical nodes) | 0 |
| 60 | 49-19-52 | 48.8% | -9 (-66..+49) |
| 150 | 31-21-68 | 34.6% | -111 (-173..-54) |
| 400 | 6-14-100 | 10.8% | **-366 (-472..-295)** |

## Findings
1. **Not a speed problem.** At equal nodes the net still loses 89% (cap 400). The corrections are harmful *inside search*.
2. **Monotone dose-response**: the more the net is allowed to move HCE, the worse it plays; at small caps it is
   indistinguishable from noise. There is no cap at which this net helped (cap 60 CI includes 0 and negatives).
3. **Held-out loss/corr/sign-agreement are not predictive of strength.** -72% loss, +0.055 corr, 99.6% decisive-sign
   agreement and -366 Elo coexist. This matches E (val loss fine, -676) and H (val ~ train, -97%). Treat val-loss as a
   smoke test only; a game-level gate is the metric.
4. **Probable cause (hypothesis, not yet shown): generalisation, not a bug.** Overfits after ~1 minute; and the held-out
   split is by *position hash*, so neighbouring positions of the same game sit on both sides -> the -72% is inflated.
   Search leaves are not drawn from SF19-game positions, and search selects whichever leaf the net is most wrong about.
5. Inference is not at fault (zero-net = HCE exactly; H measured float-vs-int max 1 cp).

## Decision
**REVERT as a shipping eval; KEEP the infrastructure** (bench-neutral, opt-in, no net is shipped, `ResidualCap=0` = HCE).
`nets/` stays empty.

## Recommendations to the others (evidence-based)
- **Use a fixed-nodes screen before any time-control gate.** 120 games at 25k nodes finish in ~2 min on 4 vCPU and gave a
  dose-response curve that ~50 games at 10+0.1 would not. Time-control gates then confirm survivors.
- **Split validation by game, not position** (needs a game id in the record; D's audit should add it). Until then,
  val-loss gains on adjacent-position data are optimistic.
- Any net must show at fixed nodes **>= 50% vs HCE** before it earns a 750-game gate (L's number).
- For a residual to help it likely needs far more *diverse* data than 348k correlated positions, early stopping at the
  best-val step (~1 epoch-equivalent here), a game-level split, and an incremental accumulator (HCE+net is ~2.3-4x slower now).
- Tuned HCE (Agent T, PR #15) is the better first lever; a residual should be trained on **top of whichever HCE ships**
  (see decision on issue #1) and must re-run `hce_labels.py` whenever HCE parameters change.

## Reproduce
```
python tools/nnue/hce_labels.py --engine target/release/chess-rust-engine data/teacher.bin
python tools/nnue/train_residual.py --data data/teacher.bin --out resid.bin --minutes 8
python tools/nnue/diagnose_residual.py --data data/teacher.bin --net resid.bin
FASTCHESS=fastchess tools/nnue/gate_vs_hce.sh target/release/chess-rust-engine resid.bin 60 nodes=25000 4 150 gate
```
Caveats: one net, one seed, one dataset; fixed-nodes games are not time-control Elo; no claim about any other net.

---

# EXP-I2 (follow-up, pre-registered on issue #1): more data, game-level validation, early stopping - NEGATIVE

Rule: retrain on ~10x more diverse data with a game-level split and early stopping; **KEEP only if some cap scores >= 52% over >= 240 games**, else stop.

Data: 1,000,000 positions from the engine's own depth-6 self-play (4 shards, distinct `CRE_SEED`, 864,039 unique; HCE vs
label MAE 63 cp, corr 0.980 - the labels are the engine's own search, so the residual can only learn what depth-6 search
adds over the static eval). New trainer options: `--split-block 4096` (validation = whole blocks of consecutive records, so no
game is split across train/val), `--val-every`, `--wd`. Validation 60,837 positions, 5% of blocks.

## The leak, measured
| run | split | lr | held-out loss vs HCE |
|---|---|---|---|
| EXP-I1 (348k SF19 positions) | per-position hash (neighbours of one game on both sides) | 2e-3 | **-71.9%** (looked superb) |
| EXP-I2a (1M self-play) | **per game block** | 2e-3 | **+23% to +33% WORSE** than HCE, getting worse each minute while train loss falls |
| EXP-I2b (same data) | per game block | 2e-4, AdamW wd 0.05, validate/10 s | best **+6.74%** at ~1 min, then overfits (+2.7% by 3.8 min) |

Position-level splitting on game-ordered data is leaky; with an honest split, the same recipe is *worse than doing nothing*
at the usual learning rate and only just positive when regularised. (EXP-I1 and I2 differ in data source, so the leak is
strongly suggested, not isolated by a controlled A/B on identical data.)

## Fixed-nodes screen (25,000 nodes/move, 60 rounds x 2 = 120 games per row, same book, colours reversed), net = I2b best-val
| `ResidualCap` | W-D-L | score | Elo (95% CI) |
|---|---|---|---|
| 400 (mean abs delta 91 cp, p95 262 cp on val) | 23-28-69 | 30.8% | -140 (-202..-86) |
| 150 | 23-23-74 | 28.7% | -158 (-223..-101) |

## Verdict: **REVERT** (rule failed: no cap near 52%). Infrastructure stays; no net is shipped.
Observations: an honest held-out gain of +6.7% in loss still plays ~-150 Elo at equal nodes, and (unlike I1) the damage is
not monotone in the cap between 150 and 400. So even a *small genuine* loss gain does not translate to search strength
here. Not tested, ranked by my guess: (a) leaf distribution (search selects the leaf where the net is most wrong; training positions are game positions, not
search leaves); (b) the net perturbs eval scale/margins used by pruning (RFP/futility/LMR are tuned to HCE); (c) correction noise at
non-quiet nodes. A fixed-depth/fixed-nodes comparison of `eval` on *search leaves* (dump leaves, label with deeper search) is the
next honest experiment, and needs search-side logging in `search.rs` (A's file) that I did not touch.

Caveats: one seed per recipe; I2b is one regularisation setting; fixed-node results are not time-control Elo; no claim about nets trained any other way.
