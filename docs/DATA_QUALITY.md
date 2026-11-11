# Data quality, provenance and rollback (Agent D)

Tools-only. Touches no `src/`, `search.rs`, `nnue.rs`, `board.rs` or workflows.

| Tool | What it guards against |
| --- | --- |
| `tools/nnue/data_audit.py` | Truncated shards, illegal positions (king counts, pawns on rank 1/8, python-chess legality sample), **side-to-move/score perspective flips** (score vs game-result sign agreement on decisive positions), duplicate rate, missing endgames. Writes `manifest.json` (sha256 per shard, teacher settings via `--provenance`, deterministic train/val split sizes). Exit 1 on hard failure so CI stops *before* the training job. |
| `tools/nnue/net_registry.py` | Untracked nets. Every net is archived by content hash with parent, data manifest and metrics. `promote` **refuses without evidence** (SPRT/H2H run URL + `--h1-accepted`). `rollback` restores the previous champion byte-for-byte (hash-verified). |
| `tools/tactics.py` | Un-measured tactical accuracy. `build` makes a *unique-best-move* suite from SF19 MultiPV=2 (>=1 s, gap >= 150 cp) so we never reward arbitrary choices; `run` gives a comparable solve-rate JSON per commit/net. |

Suggested wiring (not applied, owners decide):

```bash
python tools/nnue/data_audit.py 'data/*.bin' --manifest data/manifest.json \
  --provenance teacher=stockfish-19 movetime_ms=500 threads=1 hash=64   # before train.py
python tools/nnue/net_registry.py add out.nnue --parent <id> --data-manifest data/manifest.json --metrics '{"val_mse":...}'
python tools/nnue/net_registry.py promote <id> --evidence <SPRT run url> --h1-accepted
python tools/nnue/net_registry.py rollback --reason "..."
```

Tested locally on synthetic shards: good labels PASS; stm-flipped labels FAIL (agreement 0.000);
truncated CRN1 rejected; promote refused without evidence; rollback restores exact bytes.
Not yet run on real SF19 shards - first real use is the point of the PR.
