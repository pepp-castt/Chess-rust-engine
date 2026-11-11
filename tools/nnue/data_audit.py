#!/usr/bin/env python3
"""Data-quality audit + provenance manifest for 28-byte teacher shards (Agent D).

Record (shared by datagen.py / datagen_movetime.py / train.py):
    occ u64 | 16 bytes of piece nibbles (color*6+type, ascending square) | stm u8 |
    score i16 (cp, side-to-move POV) | wdl u8 (0 loss, 1 draw, 2 win, stm POV)

Why this exists: a wrong side-to-move flip, an illegal king count or a score/result
sign mismatch silently ruins an NNUE and looks like "the net just isn't good yet".
Hard failures exit 1 so CI can stop BEFORE a 30-minute training job is burned.

Usage:
  python tools/nnue/data_audit.py 'data/*.bin' --manifest data/manifest.json \
      --provenance teacher=stockfish-19 movetime_ms=500 threads=1 hash=64 \
      --fen-sample 2000 --split-mod 20
"""
import argparse, glob, hashlib, json, os, platform, sys, time
import numpy as np

REC = np.dtype([("occ", "<u8"), ("pcs", "u1", (16,)), ("stm", "u1"), ("score", "<i2"), ("wdl", "u1")])
assert REC.itemsize == 28
PIECE = "PNBRQKpnbrqk"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pos_hash(recs):
    """Identical to train.py:pos_hash (position identity for dedupe + deterministic split)."""
    M = np.uint64
    pcs = np.ascontiguousarray(recs["pcs"]).view("<u8").reshape(len(recs), 2)
    h = recs["occ"].astype(M) * M(0x9E3779B97F4A7C15)
    h ^= pcs[:, 0] * M(0xC2B2AE3D27D4EB4F)
    h ^= pcs[:, 1] * M(0x165667B19E3779F9)
    h ^= recs["stm"].astype(M) * M(0x27D4EB2F165667C5)
    h ^= h >> M(31)
    h *= M(0xBF58476D1CE4E5B9)
    h ^= h >> M(29)
    return h


def unpack(recs):
    """-> flat (rows, squares, pieces) arrays + per-record piece counts."""
    n = len(recs)
    occ = np.ascontiguousarray(recs["occ"]).view(np.uint8).reshape(n, 8)
    mask = np.unpackbits(occ, axis=1, bitorder="little").astype(bool)
    cnt = mask.sum(1)
    nib = np.empty((n, 32), np.uint8)
    nib[:, 0::2] = recs["pcs"] & 15
    nib[:, 1::2] = recs["pcs"] >> 4
    valid = np.arange(32)[None, :] < np.minimum(cnt, 32)[:, None]
    rows, cols = np.nonzero(mask)
    pieces = nib[valid].astype(np.int64)
    return rows, cols, pieces, cnt


def to_board(rec):
    import chess
    occ = int(rec["occ"])
    nib = []
    for b in bytes(rec["pcs"]):
        nib += [b & 15, b >> 4]
    b = chess.Board(None)
    i = 0
    for sq in range(64):
        if occ >> sq & 1:
            b.set_piece_at(sq, chess.Piece.from_symbol(PIECE[nib[i]]))
            i += 1
    b.turn = chess.WHITE if rec["stm"] == 0 else chess.BLACK
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--manifest", help="write JSON manifest here")
    ap.add_argument("--provenance", nargs="*", default=[], help="key=value pairs recorded verbatim")
    ap.add_argument("--fen-sample", type=int, default=2000, help="records cross-checked with python-chess legality")
    ap.add_argument("--split-mod", type=int, default=20, help="validation = pos_hash %% split_mod == 0 (same rule as train.py)")
    ap.add_argument("--min-sign-agreement", type=float, default=0.65,
                    help="hard-fail if among |score|>=300 the score sign agrees with the game result less often than this")
    ap.add_argument("--max-dup-rate", type=float, default=0.30)
    a = ap.parse_args()

    files = sorted({f for p in a.patterns for f in glob.glob(p, recursive=True)})
    if not files:
        print("no input files", file=sys.stderr)
        return 2
    fails, warns, arrs, finfo = [], [], [], []
    for f in files:
        sz = os.path.getsize(f)
        if sz % 28:
            fails.append(f"{f}: size {sz} not a multiple of 28 (truncated shard?)")
        arrs.append(np.fromfile(f, dtype=REC, count=sz // 28))
        finfo.append({"file": f, "bytes": sz, "records": int(len(arrs[-1])), "sha256": sha256(f)})
    recs = np.concatenate(arrs)
    n = len(recs)
    if n == 0:
        print("empty dataset", file=sys.stderr)
        return 1

    # ---- structural checks (vectorised) ----
    bad_stm = int((recs["stm"] > 1).sum())
    bad_wdl = int((recs["wdl"] > 2).sum())
    score = recs["score"].astype(np.int32)
    rows, cols, pieces, cnt = unpack(recs)
    if (cnt > 32).any() or (cnt < 2).any():
        fails.append(f"{int(((cnt > 32) | (cnt < 2)).sum())} records with impossible piece count")
    wk = np.bincount(rows, weights=(pieces == 5), minlength=n) if len(rows) == len(pieces) else np.zeros(n)
    bk = np.bincount(rows, weights=(pieces == 11), minlength=n) if len(rows) == len(pieces) else np.zeros(n)
    kings_ok = (wk == 1) & (bk == 1)
    is_pawn = (pieces == 0) | (pieces == 6)
    rank = cols // 8
    pawn_bad = np.bincount(rows, weights=(is_pawn & ((rank == 0) | (rank == 7))), minlength=n) > 0
    if bad_stm:
        fails.append(f"{bad_stm} records with stm>1")
    if bad_wdl:
        fails.append(f"{bad_wdl} records with wdl>2")
    if (~kings_ok).any():
        fails.append(f"{int((~kings_ok).sum())} records without exactly one king per side")
    if pawn_bad.any():
        fails.append(f"{int(pawn_bad.sum())} records with pawns on rank 1/8")

    # ---- label consistency: detects side-to-move / perspective flip bugs ----
    decisive = (np.abs(score) >= 300) & (recs["wdl"] != 1)
    sign_agree = None
    if decisive.sum() >= 200:
        sign_agree = float(((score[decisive] > 0) == (recs["wdl"][decisive] == 2)).mean())
        if sign_agree < a.min_sign_agreement:
            fails.append(f"score/result sign agreement {sign_agree:.3f} < {a.min_sign_agreement} "
                         f"(side-to-move perspective bug in datagen?)")
    else:
        warns.append("too few decisive+strong records to test score/result agreement")
    stm_black = float((recs["stm"] == 1).mean())
    if not 0.40 <= stm_black <= 0.60:
        warns.append(f"stm imbalance: {stm_black:.2%} black-to-move")

    # ---- duplicates / deterministic split ----
    h = pos_hash(recs)
    uniq = int(len(np.unique(h)))
    dup_rate = 1 - uniq / n
    if dup_rate > a.max_dup_rate:
        warns.append(f"duplicate rate {dup_rate:.1%} > {a.max_dup_rate:.0%}: run merge_dedupe.py")
    val_mask = (h % np.uint64(a.split_mod)) == 0

    # ---- distributions ----
    sat = float((np.abs(score) >= 2900).mean())
    hist_score = np.histogram(np.clip(score, -1500, 1500), bins=[-1500, -600, -300, -100, 100, 300, 600, 1501])[0].tolist()
    bucket_hist = np.bincount(np.minimum(np.maximum(cnt - 2, 0) // 4, 7), minlength=8).tolist()
    wdl_hist = np.bincount(recs["wdl"].clip(0, 2), minlength=3).tolist()
    if (bucket_hist[0] + bucket_hist[1]) / n < 0.01:
        warns.append("<1% endgame positions (<=9 pieces): net will mis-evaluate simple endings")

    # ---- python-chess legality on a deterministic random sample ----
    illegal = checked = 0
    if a.fen_sample:
        try:
            idx = np.random.default_rng(12345).choice(n, size=min(a.fen_sample, n), replace=False)
            for i in idx:
                checked += 1
                if not to_board(recs[i]).is_valid():
                    illegal += 1
            if illegal / max(checked, 1) > 0.001:
                fails.append(f"{illegal}/{checked} sampled positions are not legal chess positions")
        except ImportError:
            warns.append("python-chess missing: skipped legality sample")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit_host": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
        "provenance": dict(kv.split("=", 1) for kv in a.provenance if "=" in kv),
        "records": int(n), "unique": uniq, "dup_rate": round(dup_rate, 4),
        "split": {"rule": f"pos_hash % {a.split_mod} == 0 -> validation", "val": int(val_mask.sum()), "train": int(n - val_mask.sum())},
        "wdl_loss_draw_win": wdl_hist, "score_hist[-1500..1500]": hist_score,
        "saturated_frac": round(sat, 4), "sign_agreement_decisive": sign_agree,
        "piece_bucket_hist": bucket_hist, "stm_black_frac": round(stm_black, 4),
        "legality_sample": {"checked": checked, "illegal": illegal},
        "failures": fails, "warnings": warns, "files": finfo,
    }
    if a.manifest:
        with open(a.manifest, "w") as fh:
            json.dump(manifest, fh, indent=1)
    print(f"records={n} unique={uniq} dup={dup_rate:.1%} val={int(val_mask.sum())} wdl L/D/W={wdl_hist}")
    print(f"score hist={hist_score} saturated={sat:.1%} sign_agree={sign_agree}")
    print(f"piece-bucket hist (0=endgame..7=full)={bucket_hist}")
    for w in warns:
        print("WARN:", w)
    for f_ in fails:
        print("FAIL:", f_)
    print("AUDIT", "FAILED" if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
