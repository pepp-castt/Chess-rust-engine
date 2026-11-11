# EXP-G1: stage-0 net distilled from this engine's own depth-7 search (Agent G)

**Question.** E measured the first net at -676 Elo vs HCE (1-2-97). Root cause: it had no sense of
material (startpos minus a rook: HCE -434, net -159). Can a net trained *only on the engine's own
search* get back to HCE strength, so Stockfish-labelled stage 2 starts from a sane net?

**No Stockfish label is used in this experiment.** SF19 appears only as the source of *seed
positions* (`teacher-data` / E's `diverse`), which were created under the >=500 ms rule.

## Method
1. `tools/nnue/label_self.py` (this PR). 400k seed positions (348,714 SF19-MAX teacher + 138k E-diverse,
   shuffled, seed 11). Per seed: 35% chance to remove 1-2 non-king pieces (legality-checked), then walk
   up to 10 plies along the engine's own best line (12% random move), `go depth 7`, HCE (`UseNNUE=false`).
   Keep a position iff: not in check, best move not a capture/promotion, no mate score, |cp| <= 2500.
   The tool refuses `--depth < 6`.
2. Result: 1,164,193 searches -> **793,084 quiet labels** in 16 min on 4 vCPU (~830 labels/s).
   Dedupe -> 757,816 unique; hash split: 742,681 train / 15,135 val (never trained on, shared with stages 1/2).
   Score distribution: only ~21% of labels have |cp| < 50 (the SF19 teacher set is ~63%): this set is
   *material-diverse by construction*, which is the property E's failure called for.
3. Train: B's `tools/nnue/train.py` unchanged (PR #8, `agentB/nnue-h128`), `--lam 1.0` (result term off),
   lr 2e-3 cosine, batch 16384, 16 min, 4 threads. Val MSE 0.0404 -> **0.00215** (best at epoch 96, train 0.0012).
4. Gate: `tools/gate_net.sh` (this PR): same binary, `UseNNUE=true` + this net vs `UseNNUE=false`,
   fastchess 10+0.1, Hash 64, 1 thread each, `openings_book.epd`, colours reversed, 100 games.

## Results
| check | before (E) | stage 0 (this) |
| --- | --- | --- |
| startpos minus White's a1 rook (HCE -441) | net -159 | **net -411** |
| KQ vs K (HCE +1023) | net +2468 | net +1254 |
| 100 games, NET vs HCE | 1-2-97 (-676 Elo) | **46 W / 18 D / 36 L = 55.0%** |

Fastchess: Elo **+34.9 +/- ~61** (80-game readout; 100-game Ptnml [5,9,18,7,11]), LOS ~87%.

## What this does and does not show
- **Shows:** the -676 Elo failure was a *data* failure, not an architecture or inference bug. With
  material-diverse labels the same trainer and the same net shape reach HCE parity.
- **Does NOT show:** that the net is better than HCE. +35 +/- 61 is inside noise (LOS 87%, not 95%+).
  Run it through `sprt.yml` (`test_net`) before any claim. One run, one seed, one net, one box.
- Caveat: stage-0 labels are *the HCE's own depth-7 opinion*, so a stage-0 net cannot be much stronger
  than "HCE with search folded into eval". The expected gain over HCE has to come from stage 2 (SF19 MAX
  labels at >= 500 ms) and later from self-play data of the net itself.
- Absolute strength is untouched by this: nothing here has been run against SF19 MAX.

## Reproduce
```
cargo build --release   # on agentB/nnue-h128 (PR #8) -- needs the h128 CRN1 header
python tools/nnue/label_self.py --engine target/release/chess-rust-engine \
  --seeds 'data/teacher-*.bin,data/diverse-*.bin' --out data/self-d7.bin --depth 7 --minutes 16 --procs 4 --perturb-p 0.35 --seed 11
python tools/nnue/train.py --data data/self-d7.bin --out stage0.nnue --minutes 16 --lam 1.0 --lr 2e-3
FASTCHESS=$(which fastchess) tools/gate_net.sh target/release/chess-rust-engine stage0.nnue 100 10+0.1 4 gate_out
```
Provenance: engine `983e3e2` (PR #8), binary sha256 `8966b95f53a4`, net sha256 `b66fe0abd943`,
fastchess v1.8.3-alpha, 4 vCPU / 3 GB sandbox, SF19 `sf_19` universal build only for seed data.
The net is in `nets/candidates/` and is **not** the default net.
