#!/usr/bin/env python3
"""Bulk teacher labels from the Lichess evaluation database -> 28-byte CRN1 training records.

Why this exists (Agent X, lane: bulk public Stockfish labels).
    The thread's measured bottleneck is label VOLUME: ~350k SF19 500 ms labels vs the 1e7-1e9
    positions where small NNUEs start beating a hand-crafted eval. ~350M positions of
    Stockfish analysis (CC0, median depth ~36, many at 1M+ nodes) already exist at
    https://database.lichess.org/#evals . Labels are never played, so the ">=500 ms for games"
    rule does not apply, and these labels are *deeper* than anything we can generate on runners.

Record layout (identical to tools/nnue/train.py `REC`, 28 bytes, little endian):
    occ u64 | 16 B piece nibbles (color*6+type, ascending square, low nibble first) |
    stm u8 (0 white, 1 black) | score i16 (cp, SIDE-TO-MOVE POV) | wdl u8 (0/1/2, STM POV)

!! PERSPECTIVE: Lichess `cp` / `mate` are from WHITE's point of view (verified by playing out
   mating PVs: mate>0 <=> white delivers mate, independent of side to move; see
   tools/nnue/test_lichess_eval.py). The trainer wants side-to-move POV, so we negate for black.
!! WDL: the database has no game results. wdl is written as 1 (draw) and is MEANINGLESS.
   Train with `--lam 1.0` (score only). meta.json says `"wdl_valid": false`.

Quality filters (all on by default; every drop reason is counted in the meta file):
    depth >= --min-depth   highest-depth eval entry, first PV (Lichess' own recommendation)
    mate scores            dropped (not a static-eval target)
    |cp| > --max-abs-cp    dropped (sigmoid(cp/400) saturates; decided positions teach nothing)
    side to move in check  dropped (static eval undefined)
    best move is capture / promotion / en passant  dropped (non-quiet: label is a search result)
    illegal / malformed    dropped (king counts, pawns on rank 1/8, mover can capture king)

Usage:
    python tools/nnue/lichess_eval_to_bin.py --out data/lichess-s0.bin --max-positions 10000000
    python tools/nnue/lichess_eval_to_bin.py --out data/lichess-s3.bin --skip-lines 30000000 \
        --max-positions 10000000                      # disjoint slice (shard 3)
    python tools/nnue/lichess_eval_to_bin.py --input local.jsonl.zst --out x.bin --max-positions 1000000
"""
import argparse, collections, hashlib, json, multiprocessing as mp, os, struct, sys, time, urllib.request

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
REC = struct.Struct("<Q16sBhB")
assert REC.size == 28

PIECE = {c: (0 if c.isupper() else 6) + "pnbrqk".index(c.lower()) for c in "PNBRQKpnbrqk"}
W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING = 0, 1, 2, 3, 4, 5


def _tables():
    kn, kg, rays_d, rays_o = [], [], [], []
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        kn.append([(r + dr) * 8 + f + df for dr, df in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
                   if 0 <= r + dr < 8 and 0 <= f + df < 8])
        kg.append([(r + dr) * 8 + f + df for dr in (-1, 0, 1) for df in (-1, 0, 1)
                   if (dr or df) and 0 <= r + dr < 8 and 0 <= f + df < 8])
        d, o = [], []
        for dr, df, tgt in ((1, 1, d), (1, -1, d), (-1, 1, d), (-1, -1, d), (1, 0, o), (-1, 0, o), (0, 1, o), (0, -1, o)):
            ray, rr, ff = [], r + dr, f + df
            while 0 <= rr < 8 and 0 <= ff < 8:
                ray.append(rr * 8 + ff)
                rr += dr
                ff += df
            tgt.append(ray)
        rays_d.append(d)
        rays_o.append(o)
    return kn, kg, rays_d, rays_o


KN, KG, RD, RO = _tables()


def attacked(b, sq, by):
    """Is `sq` attacked by colour `by` (0 white / 1 black)? `b` = list of 64 piece codes or -1."""
    o = 6 * by
    for t in KN[sq]:
        if b[t] == o + W_KNIGHT:
            return True
    for t in KG[sq]:
        if b[t] == o + W_KING:
            return True
    f, r = sq & 7, sq >> 3
    pr = r - 1 if by == 0 else r + 1  # a white pawn attacking sq sits one rank below it
    if 0 <= pr < 8:
        for pf in (f - 1, f + 1):
            if 0 <= pf < 8 and b[pr * 8 + pf] == o + W_PAWN:
                return True
    for ray in RD[sq]:
        for t in ray:
            p = b[t]
            if p >= 0:
                if p == o + W_BISHOP or p == o + W_QUEEN:
                    return True
                break
    for ray in RO[sq]:
        for t in ray:
            p = b[t]
            if p >= 0:
                if p == o + W_ROOK or p == o + W_QUEEN:
                    return True
                break
    return False


def parse_fen(fen):
    """-> (board[64], stm, ep_sq or -1) or None if malformed."""
    parts = fen.split()
    if len(parts) < 2:
        return None
    rows = parts[0].split("/")
    if len(rows) != 8:
        return None
    b = [-1] * 64
    for i, row in enumerate(rows):
        f = 0
        rank = 7 - i
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                p = PIECE.get(ch)
                if p is None or f > 7:
                    return None
                b[rank * 8 + f] = p
                f += 1
        if f != 8:
            return None
    if parts[1] not in ("w", "b"):
        return None
    ep = -1
    if len(parts) > 3 and parts[3] != "-" and len(parts[3]) == 2:
        ep = (int(parts[3][1]) - 1) * 8 + (ord(parts[3][0]) - 97)
    return b, (0 if parts[1] == "w" else 1), ep


def sq_of(s):
    return (int(s[1]) - 1) * 8 + (ord(s[0]) - 97)


def pack(b, stm, score_stm, wdl=1):
    occ, nib = 0, []
    for sq in range(64):
        if b[sq] >= 0:
            occ |= 1 << sq
            nib.append(b[sq])
    nib += [0] * (32 - len(nib))
    return REC.pack(occ, bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(16)), stm, score_stm, wdl)


def convert_line(line, min_depth, max_cp, keep_noisy):
    """-> (record bytes | None, drop_reason | 'ok', piece_count, depth)"""
    try:
        j = json.loads(line)
        e = max(j["evals"], key=lambda x: (x["depth"], x["knodes"]))
        pv = e["pvs"][0]
    except Exception:
        return None, "bad_json", 0, 0
    depth = e["depth"]
    if depth < min_depth:
        return None, "shallow", 0, depth
    if "cp" not in pv:
        return None, "mate", 0, depth
    cp = pv["cp"]
    if abs(cp) > max_cp:
        return None, "decided", 0, depth
    parsed = parse_fen(j["fen"])
    if parsed is None:
        return None, "bad_fen", 0, depth
    b, stm, ep = parsed
    kings = ([i for i, p in enumerate(b) if p == W_KING], [i for i, p in enumerate(b) if p == 6 + W_KING])
    if len(kings[0]) != 1 or len(kings[1]) != 1:
        return None, "illegal", 0, depth
    n = sum(1 for p in b if p >= 0)
    # the record has 32 nibble slots; corrupt rows with 33/36/62 "pieces" exist in the source
    if n > 32 or sum(1 for p in b if 0 <= p < 6) > 16 or sum(1 for p in b if p >= 6) > 16 \
            or sum(1 for p in b if p == W_PAWN) > 8 or sum(1 for p in b if p == 6 + W_PAWN) > 8:
        return None, "illegal", n, depth
    for sq in list(range(8)) + list(range(56, 64)):
        if b[sq] >= 0 and b[sq] % 6 == W_PAWN:
            return None, "illegal", n, depth
    if attacked(b, kings[1 - stm][0], stm):  # side NOT to move is in check => illegal
        return None, "illegal", n, depth
    if attacked(b, kings[stm][0], 1 - stm):
        return None, "in_check", n, depth
    if not keep_noisy:
        line_mv = (pv.get("line") or "").split()
        if line_mv:
            mv = line_mv[0]
            if len(mv) >= 5:
                return None, "noisy_promo", n, depth
            to = sq_of(mv[2:4])
            frm = sq_of(mv[0:2])
            if b[to] >= 0:
                # castling is encoded king-takes-own-rook in Chess960 notation; not a capture
                if b[frm] % 6 == W_KING and (b[to] >= 6) == (b[frm] >= 6):
                    pass
                else:
                    return None, "noisy_capture", n, depth
            elif b[frm] % 6 == W_PAWN and ep == to and (frm & 7) != (to & 7):
                return None, "noisy_ep", n, depth
    score_stm = cp if stm == 0 else -cp  # Lichess cp is WHITE POV -> side-to-move POV
    return pack(b, stm, score_stm), "ok", n, depth


def work(args):
    lines, min_depth, max_cp, keep_noisy = args
    out, reasons, hist, dsum = [], collections.Counter(), collections.Counter(), 0
    sc = collections.Counter()
    for ln in lines:
        rec, why, n, depth = convert_line(ln, min_depth, max_cp, keep_noisy)
        reasons[why] += 1
        if rec is not None:
            out.append(rec)
            hist[n] += 1
            dsum += depth
            sc[max(-20, min(20, REC.unpack(rec)[3] // 100))] += 1
    return b"".join(out), reasons, hist, dsum, sc


def open_stream(a):
    import zstandard
    if a.input:
        raw = open(a.input, "rb")
        info = {"source": os.path.abspath(a.input), "size": os.path.getsize(a.input)}
    else:
        req = urllib.request.Request(a.url, headers={"User-Agent": "chess-rust-engine-datagen/1.0"})
        raw = urllib.request.urlopen(req, timeout=120)
        info = {"source": a.url, "etag": raw.headers.get("ETag"), "last_modified": raw.headers.get("Last-Modified"),
                "content_length": raw.headers.get("Content-Length")}
    return zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw), info


def lines_after_skip(stream, skip, stride, chunk=16 << 20):
    """Yield raw JSON lines, first discarding `skip` lines by counting newlines only (fast)."""
    carry, idx = b"", 0
    try:
        while True:
            data = stream.read(chunk)
            if not data:
                break
            data = carry + data
            if skip > 0:
                c = data.count(b"\n")
                if c <= skip:
                    skip -= c
                    carry = data[data.rfind(b"\n") + 1:]
                    continue
                pos = -1
                for _ in range(skip):
                    pos = data.find(b"\n", pos + 1)
                data, skip = data[pos + 1:], 0
            parts = data.split(b"\n")
            carry = parts.pop()
            for ln in parts:
                if ln:
                    if idx % stride == 0:
                        yield ln
                    idx += 1
    except Exception as ex:  # truncated / dropped connection: keep what we have
        print(f"[warn] stream ended early: {type(ex).__name__}: {ex}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--input", help="local lichess_db_eval.jsonl.zst (else streamed from --url)")
    ap.add_argument("--max-positions", type=int, default=1_000_000, help="stop after this many KEPT records")
    ap.add_argument("--skip-lines", type=int, default=0, help="skip this many database lines first (disjoint shards)")
    ap.add_argument("--max-lines", type=int, default=0, help="read at most this many database lines after --skip-lines "
                    "(0 = unlimited). Shards use skip=k*L, max-lines=L so ranges are provably disjoint")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth database line")
    ap.add_argument("--min-depth", type=int, default=24)
    ap.add_argument("--max-abs-cp", type=int, default=2000)
    ap.add_argument("--keep-noisy", action="store_true", help="keep positions whose best move is a capture/promotion")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--batch", type=int, default=4000)
    a = ap.parse_args()
    t0 = time.time()
    stream, src = open_stream(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    reasons, hist, sc = collections.Counter(), collections.Counter(), collections.Counter()
    kept = dsum = seen = nb = 0
    sha = hashlib.sha256()

    def batches():
        buf, taken = [], 0
        for ln in lines_after_skip(stream, a.skip_lines, a.stride):
            if a.max_lines and taken >= a.max_lines:
                break
            taken += 1
            buf.append(ln)
            if len(buf) >= a.batch:
                yield (buf, a.min_depth, a.max_abs_cp, a.keep_noisy)
                buf = []
        if buf:
            yield (buf, a.min_depth, a.max_abs_cp, a.keep_noisy)

    with open(a.out, "wb") as fo, mp.Pool(a.workers) as pool:
        for blob, rs, h, ds, s in pool.imap(work, batches(), chunksize=1):
            nb += 1
            seen += sum(rs.values())
            reasons.update(rs)
            if kept + len(blob) // 28 > a.max_positions:
                blob = blob[: (a.max_positions - kept) * 28]
            fo.write(blob)
            sha.update(blob)
            kept += len(blob) // 28
            hist.update(h)
            sc.update(s)
            dsum += ds
            if nb % 25 == 0:
                print(f"  seen={seen:,} kept={kept:,} ({seen/(time.time()-t0):,.0f} lines/s)", flush=True)
            if kept >= a.max_positions:
                pool.terminate()
                break
    stream.close()
    meta = {
        "tool": "tools/nnue/lichess_eval_to_bin.py", "teacher": "Stockfish via Lichess analysis-board (mixed versions, crowdsourced; CC0)",
        "source": src, "args": vars(a), "lines_seen": seen, "records": kept, "sha256": sha.hexdigest(),
        "score_pov": "side-to-move (converted from Lichess white-POV)", "wdl_valid": False, "train_with": "--lam 1.0",
        "drop_reasons": dict(reasons), "mean_depth": round(dsum / max(kept, 1), 2), "seconds": round(time.time() - t0, 1),
        "piece_count_hist": {str(k): hist[k] for k in sorted(hist)},
        "score_hist_pawns_stm": {str(k): sc[k] for k in sorted(sc)},
    }
    json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("records", "lines_seen", "drop_reasons", "mean_depth", "seconds")}))


if __name__ == "__main__":
    main()
