#!/usr/bin/env python3
"""Bulk Stockfish-labelled positions -> the trainer's 28-byte records (Agent R).

Source: https://huggingface.co/datasets/Lichess/chess-position-evaluations  (licence: CC0-1.0)
  416M unique positions / 1.0B rows, produced by Stockfish (various versions, depth ~20-60+) in
  Lichess users' browsers. One row per MultiPV line; rows of one FEN are adjacent.
  Columns: fen, line (PV, UCI), depth, knodes, cp, mate.

Why this exists: every net trained so far in this repo used <= 1M positions (348k SF19-MAX labels
is ~3 samples/parameter for H=128) and lost to the HCE. A label is never *played*, so the 500 ms
floor that protects GAMES does not protect the label; what matters is label quality and volume.
Lichess labels are at depth 20+ (typically far deeper than 500 ms of one thread) and free in bulk.
This is a *teacher dataset*, not a Stockfish network: no SF weights are used anywhere.

What it does, per FEN group (all counted in the stats, nothing silent):
  * keeps the best PV for the side to move (MultiPV rows are not ranked, we choose by score)
  * converts the score to side-to-move POV (the trainer's convention), see --pov
  * drops: mate scores, |cp| > --max-cp, depth < --min-depth, side to move in check,
    best move is a capture / promotion (standard "quiet" filter), >32 pieces, no king pair
  * optional --keep-rate random subsample (seeded) so a truncated run is not biased by file order
Record: occ u64 | 16B nibbles (colour*6+type, ascending square) | stm u8 | score i16 | wdl u8
  wdl is UNKNOWN for this data (no game result) and is written as 1 (draw) -> train with --lam 1.0.
  The `lichess-bulk.yml` workflow enforces that.

Castling rights / en-passant are not part of the 28-byte record (same as every other source here).

Usage:
  python3 tools/nnue/lichess_to_records.py --in data_0000.parquet --out lichess_0000.bin \
      --min-depth 20 --max-cp 1500 --keep-rate 0.5 --max-out 20000000
  python3 tools/nnue/lichess_to_records.py --selftest        # needs python-chess; no network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import time

REC = struct.Struct("<Q16sBhB")  # 28 bytes, identical to datagen_movetime.py / train.py
assert REC.size == 28

# piece char -> nibble (white 0..5 = P N B R Q K, black 6..11)
NIB = {c: i for i, c in enumerate("PNBRQK")}
NIB.update({c: 6 + i for i, c in enumerate("pnbrqk")})

KN = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KG = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def parse_fen_board(fen: str):
    """-> (board[64] of piece chars or None, white_to_move) ; sq = rank*8+file, a1 = 0."""
    parts = fen.split(" ")
    rows = parts[0].split("/")
    if len(rows) != 8:
        return None, True
    b = [None] * 64
    for ri, row in enumerate(rows):
        r = 7 - ri
        f = 0
        for ch in row:
            if "1" <= ch <= "8":
                f += ord(ch) - 48
            else:
                if f > 7 or ch not in NIB:
                    return None, True
                b[r * 8 + f] = ch
                f += 1
        if f != 8:
            return None, True
    return b, (len(parts) < 2 or parts[1] == "w")


def in_check(b, white_to_move: bool) -> bool:
    """Is the side to move's king attacked?  (Plain ray / jump scan, no move generation.)"""
    king = "K" if white_to_move else "k"
    try:
        ksq = b.index(king)
    except ValueError:
        return True  # no king: treat as invalid
    kr, kf = divmod(ksq, 8)
    en = (lambda c: c.islower()) if white_to_move else (lambda c: c.isupper())
    # a white king is attacked by black pawns one rank above it (and vice versa)
    pr = kr + (1 if white_to_move else -1)
    pawn = "p" if white_to_move else "P"
    if 0 <= pr < 8:
        for df in (-1, 1):
            f = kf + df
            if 0 <= f < 8 and b[pr * 8 + f] == pawn:
                return True
    nite = "n" if white_to_move else "N"
    for dr, df in KN:
        r, f = kr + dr, kf + df
        if 0 <= r < 8 and 0 <= f < 8 and b[r * 8 + f] == nite:
            return True
    ek = "k" if white_to_move else "K"
    for dr, df in KG:
        r, f = kr + dr, kf + df
        if 0 <= r < 8 and 0 <= f < 8 and b[r * 8 + f] == ek:
            return True
    for dirs, sl in ((ORTH, "rq"), (DIAG, "bq")):
        for dr, df in dirs:
            r, f = kr + dr, kf + df
            while 0 <= r < 8 and 0 <= f < 8:
                p = b[r * 8 + f]
                if p is not None:
                    if en(p) and p.lower() in sl:
                        return True
                    break
                r += dr
                f += df
    return False


def pack_board(b, white_to_move: bool, score_stm: int, wdl: int = 1):
    occ = 0
    nib = []
    for sq in range(64):
        p = b[sq]
        if p is not None:
            occ |= 1 << sq
            nib.append(NIB[p])
    if len(nib) > 32:
        return None
    nib += [0] * (32 - len(nib))
    pcs = bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(16))
    return REC.pack(occ, pcs, 0 if white_to_move else 1, score_stm, wdl)


def sq_of(s: str) -> int:
    return (ord(s[1]) - 49) * 8 + (ord(s[0]) - 97)


STAT_KEYS = ["rows", "fens", "kept", "bad_fen", "no_line", "mate", "depth", "max_cp", "in_check",
             "capture_or_promo", "too_many_pieces", "subsampled_out"]


def choose_best(rows, white_to_move: bool, pov: str):
    """rows: list of (line, depth, cp, mate). Best row for the side to move (non-mate rows only)."""
    best = None
    bkey = None
    for r in rows:
        if r[3] is not None or r[2] is None:
            continue
        key = r[2] if (pov == "stm" or white_to_move) else -r[2]  # larger = better for STM
        if bkey is None or key > bkey or (key == bkey and (r[1] or 0) > (best[1] or 0)):
            best, bkey = r, key
    return best


def process_group(fen, rows, a, st, rng):
    """rows of one FEN -> packed record or None (reason counted in st)."""
    st["fens"] += 1
    if a.keep_rate < 1.0 and rng.random() >= a.keep_rate:
        st["subsampled_out"] += 1
        return None
    b, wtm = parse_fen_board(fen)
    if b is None:
        st["bad_fen"] += 1
        return None
    best = choose_best(rows, wtm, a.pov)
    if best is None:
        st["mate"] += 1
        return None
    line, depth, cp, _ = best
    if depth is None or depth < a.min_depth:
        st["depth"] += 1
        return None
    if not line:
        st["no_line"] += 1
        return None
    stm_cp = cp if (a.pov == "stm" or wtm) else -cp
    if abs(stm_cp) > a.max_cp:
        st["max_cp"] += 1
        return None
    if in_check(b, wtm):
        st["in_check"] += 1
        return None
    mv = line.split(" ", 1)[0]
    if len(mv) != 4:  # promotion (5 chars) or malformed
        st["capture_or_promo"] += 1
        return None
    dst, src = sq_of(mv[2:4]), sq_of(mv[0:2])
    if b[dst] is not None or (b[src] in ("P", "p") and mv[0] != mv[2]):
        st["capture_or_promo"] += 1  # capture, or en-passant (pawn changes file onto an empty square)
        return None
    rec = pack_board(b, wtm, int(stm_cp), 1)
    if rec is None:
        st["too_many_pieces"] += 1
        return None
    st["kept"] += 1
    return rec


def convert(a):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(a.inp)
    st = {k: 0 for k in STAT_KEYS}
    rng = random.Random(a.seed)
    t0 = time.time()
    h = hashlib.sha256()
    out = open(a.out, "wb")
    state = {"fen": None, "rows": [], "written": 0}

    def flush():
        if state["fen"] is None:
            return
        rec = process_group(state["fen"], state["rows"], a, st, rng)
        if rec is not None:
            out.write(rec)
            h.update(rec)
            state["written"] += 1
        state["fen"], state["rows"] = None, []

    done = False
    last_print = 0
    for batch in pf.iter_batches(batch_size=1 << 18, columns=["fen", "line", "depth", "cp", "mate"]):
        fens = batch.column(0).to_pylist()
        lines = batch.column(1).to_pylist()
        depths = batch.column(2).to_pylist()
        cps = batch.column(3).to_pylist()
        mates = batch.column(4).to_pylist()
        for i in range(len(fens)):
            st["rows"] += 1
            f = fens[i]
            if f != state["fen"]:
                flush()
                if a.max_out and state["written"] >= a.max_out:
                    done = True
                    break
                state["fen"] = f
            state["rows"].append((lines[i], depths[i], cps[i], mates[i]))
        if done:
            break
        if st["rows"] - last_print >= (1 << 22):
            last_print = st["rows"]
            el = time.time() - t0
            print(f"  rows={st['rows']:,} kept={state['written']:,} {st['rows']/max(el,1e-9):,.0f} rows/s", flush=True)
    if not done:
        flush()
    out.close()
    el = time.time() - t0
    with open(a.inp, "rb") as fh:
        head_sha = hashlib.sha256(fh.read(1 << 20)).hexdigest()
    prov = {
        "tool": "tools/nnue/lichess_to_records.py (Agent R)",
        "source": "Lichess/chess-position-evaluations",
        "source_licence": "CC0-1.0",
        "source_file": os.path.basename(a.inp),
        "source_file_sha256_first_1MiB": head_sha,
        "teacher": "Stockfish, various versions/depths run in Lichess browsers; NOT the pinned SF19 binary",
        "score_perspective_in_source": a.pov,
        "score_perspective_in_records": "side-to-move",
        "wdl_field": "unknown for this source; written as 1; train with --lam 1.0",
        "filters": {"min_depth": a.min_depth, "max_cp": a.max_cp, "quiet_best_move": True,
                    "not_in_check": True, "mate_dropped": True, "keep_rate": a.keep_rate,
                    "seed": a.seed, "max_out": a.max_out},
        "stats": st,
        "records": state["written"],
        "records_sha256": h.hexdigest(),
        "seconds": round(el, 1),
    }
    with open(a.out + ".json", "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps(prov, indent=1))
    return prov


def selftest():
    """Byte-compare our packer / check detector against python-chess on random positions."""
    import chess

    rnd = random.Random(7)
    n_ok = 0
    for _ in range(300):
        bd = chess.Board()
        for _ply in range(rnd.randint(4, 120)):
            mv = list(bd.legal_moves)
            if not mv:
                break
            bd.push(rnd.choice(mv))
            fen = bd.fen()
            b, wtm = parse_fen_board(fen)
            assert wtm == bd.turn
            if in_check(b, wtm) != bd.is_check():
                raise SystemExit(f"in_check mismatch on {fen}")
            # reference pack, copied from datagen_movetime.py::pack
            occ = 0
            nib = []
            for sq in range(64):
                p = bd.piece_at(sq)
                if p is not None:
                    occ |= 1 << sq
                    nib.append((0 if p.color else 6) + p.piece_type - 1)
            nib += [0] * (32 - len(nib))
            ref = REC.pack(occ, bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(16)),
                           0 if bd.turn else 1, 123, 1)
            if ref != pack_board(b, wtm, 123, 1):
                raise SystemExit(f"pack mismatch on {fen}")
            n_ok += 1
    print(f"SELFTEST OK: {n_ok} positions; pack == datagen_movetime.pack byte-for-byte; "
          f"in_check == python-chess")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", default="lichess.bin")
    ap.add_argument("--min-depth", type=int, default=20)
    ap.add_argument("--max-cp", type=int, default=1500)
    ap.add_argument("--keep-rate", type=float, default=1.0)
    ap.add_argument("--max-out", type=int, default=0, help="stop after this many records (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pov", choices=["white", "stm"], default="white",
                    help="perspective of `cp` in the source (verify with tools/nnue/lichess_pov_check.py)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.inp:
        ap.error("--in required")
    convert(a)


if __name__ == "__main__":
    main()
