# EXP-P1 — NNUE train/inference parity, and what a Stockfish label costs (Agent P)

No net trained, no Elo claimed, `src/` untouched, `bench` unchanged (178914 nodes at d8 on `main` 7580714).

## Question 1 — can a train/inference mismatch explain "good val loss, loses to HCE"?

Every net so far (B, E, G, H, I) reported a good held-out loss and still lost to the HCE.
A mismatch between what `tools/nnue/train.py` optimises and what `src/nnue.rs` computes would produce
exactly that, and nobody had measured it (`train.py` writes `net.bin.parity.txt` but nothing compares it).

**Method** (`tools/nnue/parity_check.py`, runs in seconds, no Stockfish, no torch):
random CRN1 net in the trainer's byte layout, 400 python-chess positions (213 white / 187 black to
move, all 8 material buckets covered: 78/76/52/27/31/34/38/64), float reference from the documented
feature convention using the *dequantised* weights, then
(a) the real engine via `CRE_NNUE_FILE` + UCI `eval`, and (b) the trainer's real `decode()` extracted from
`train.py` with `ast`, fed through the same forward pass.

**Result:** engine vs float reference worst |diff| **1.01 cp** (integer rounding), 0/400 over 3 cp;
trainer `decode()` vs reference worst |diff| **0.0000 cp**. Eval std 173 cp, nothing at the +-3000 clamp,
so the test could see a perspective, bucket or index error. **PARITY OK.**

**Conclusion:** feature index, perspective, bucket, scale (QA/QB/SCALE), sign and clamping agree
between trainer data path and engine. A train/inference bug is **not** why nets lose. That removes one
hypothesis and moves the weight onto the others already on the table in #1 (position-level validation
leak, search leaves vs game positions, HCE-tuned pruning margins, **data volume**).
Not covered: `Net.forward` (6 lines, read by eye; `--trainer` checks it wherever torch is installed),
and search-time behaviour of the net. The check is wired into `ci.yml`.

## Question 2 — is the >=500 ms label floor the reason we have 348k positions?

`teacher-42595442579.bin` = 348,714 labels from 8 runners x 150 min x 4 procs (~80 core-hours, i.e. ~4.4k
labels per core-hour including game play). The ">=500 ms" rule was written for *games* ("120 ms games teach bad
moves"). A label is not a game move: nobody plays it. So I measured what a cheaper *label* gives up.

**Method** (`tools/nnue/label_cost.py`, data in `EXP-P1-data/`): 300 quiet NextGen self-play positions
(`datagen 3000 6`, seed 101, every 10th), Stockfish 19 (`UCI_LimitStrength` off, 1 thread, 64 MB hash)
at depth 8/10/12/14/16, 500 ms, 2000 ms. Error = MSE in `sigmoid(cp/400)` (the trainer's space) vs the 2000 ms label.

| label | ms/pos | MSE vs 2 s | MSE / label variance | positions / core-hour |
|---|---:|---:|---:|---:|
| d8  | 7   | 0.00604 | 6.9% | 524,043 |
| d10 | 10  | 0.00573 | 6.6% | 366,113 |
| **d12** | **21** | 0.00450 | **5.2%** | **171,330** |
| d14 | 59  | 0.00307 | 3.5% | 61,458 |
| d16 | 179 | 0.00175 | 2.0% | 20,143 |
| t500 | 499 | 0.00034 | 0.4% (**biased low**, see below) | 7,219 |
| t2000 (reference) | 1980 | 0 | 0 | 1,817 |

* d12 labels cost **24x less** than 500 ms labels and have **0 sign flips beyond +-100 cp** against the 2 s label (n=300).
* The 500 ms row is **not** a fair noise estimate: same engine, same position, shared search path. All rows are
  lower bounds on true label noise. N=300 -> every MSE has a wide interval; I report no CIs because I did not bootstrap them.
* **Negative result:** a cheap disagreement filter (drop positions where d8 and d12, or d10 and d14, disagree) does
  **not** find the bad labels (d12 MSE on kept 0.00450 -> 0.00456 at thr 0.10, 0.00417 at thr 0.03 while dropping 13%).
  Don't build a quiet-ness filter on this signal.

**What this does and does not show.** It shows the *price list*: ~24x more positions per core-hour for ~5% of
label variance unexplained instead of ~0.4-1%. It does **not** show that the net gets better. Label noise is
roughly independent of the net's error and shrinks as 1/N under averaging; whether 24x data beats 24x cleaner
labels is an empirical question for this net size and the only way to answer it is the experiment below.

## Pre-registered follow-up (not run; needs runner time)

Same trainer, same shard-disjoint validation (Agent M's seeds, never trained on), same wall-clock budget:

| arm | data | label |
|---|---|---|
| T500 | existing 348k | SF19 MAX 500 ms |
| D12 | >=5M self-play positions (`datagen` x 20 seeds) | SF19 MAX depth 12 (`relabel_depth.py`) |
| MIX | D12 + the 348k 500 ms set upweighted | both |

Gate (from I's EXP-I1): fixed 25k nodes/move, >=120 games vs HCE on C's book, KEEP only if >=52%; a time-control
gate only after that. If D12 does not beat T500 on the fixed-nodes screen, the volume hypothesis is dead and
I say so here. Cost estimate: 5M labels = ~29 core-hours at d12 = one 4-vCPU runner for ~7 h (two 3.5 h matrix legs).

## Files
`tools/nnue/parity_check.py` (CI gate) | `tools/nnue/label_cost.py` | `tools/nnue/relabel_depth.py`
(order-preserving, 28-byte records identical to `train.py`, writes `<out>.provenance.json`; refuses depth < 10;
round-trip verified: 400/400 records decode back to the source FEN, stm and WDL) | `EXP-P1-data/`.
