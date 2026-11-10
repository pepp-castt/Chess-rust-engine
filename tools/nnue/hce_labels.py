#!/usr/bin/env python3
"""Add the engine's own HCE static eval to every 28-byte training record.

    python tools/nnue/hce_labels.py --engine target/release/chess-rust-engine data/teacher-*.bin

For each `X.bin` writes `X.bin.hce` = int16[N], HCE static eval in centipawns from the
side to move (same POV as the record's `score`). Order is the record order, so the sidecar can
be indexed with the same permutation as the records. Used by `train_residual.py`: the residual
net learns `target - HCE`, i.e. only what HCE gets wrong.
"""
import argparse, subprocess, sys
import numpy as np

REC = np.dtype([("occ", "<u8"), ("pcs", "u1", (16,)), ("stm", "u1"), ("score", "<i2"), ("wdl", "u1")])
SYM = "PNBRQKpnbrqk"  # nibble = color*6 + type (white 0..5, black 6..11)


def to_fens(recs):
    out = []
    for r in recs:
        occ = int(r["occ"])
        pcs = r["pcs"]
        board = [None] * 64
        k = 0
        for sq in range(64):
            if occ >> sq & 1:
                nib = (int(pcs[k >> 1]) >> (4 * (k & 1))) & 15
                board[sq] = SYM[nib]
                k += 1
        rows = []
        for rank in range(7, -1, -1):
            s, e = "", 0
            for f in range(8):
                c = board[rank * 8 + f]
                if c is None:
                    e += 1
                else:
                    if e:
                        s += str(e); e = 0
                    s += c
            if e:
                s += str(e)
            rows.append(s)
        out.append("/".join(rows) + (" w" if r["stm"] == 0 else " b") + " - - 0 1")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for f in a.files:
        recs = np.fromfile(f, dtype=REC)
        fens = to_fens(recs)
        res = subprocess.run([a.engine, "hce-dump"], input="\n".join(fens) + "\n", capture_output=True, text=True, check=True)
        vals = [int(l.rsplit("|", 1)[1]) for l in res.stdout.splitlines() if "|" in l]
        if len(vals) != len(recs):
            sys.exit(f"{f}: engine returned {len(vals)} evals for {len(recs)} records (unparseable FEN?)")
        arr = np.clip(np.array(vals), -32000, 32000).astype("<i2")
        arr.tofile(f + ".hce")
        d = arr.astype(np.float64) - recs["score"]
        print(f"{f}: n={len(recs)} HCE-vs-label mean_err={d.mean():.1f} mae={np.abs(d).mean():.1f} corr={np.corrcoef(arr, recs['score'])[0,1]:.3f}")


if __name__ == "__main__":
    main()
