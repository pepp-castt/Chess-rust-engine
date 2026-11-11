#!/usr/bin/env python3
"""High-volume Stockfish 19 MAX labelling at a FIXED DEPTH (Agent P).

Why: the >=500 ms/position floor yields ~7k labels per core-hour (EXP-P1). A fixed-depth label
at depth 12 costs ~21 ms (~170k/core-hour) and explains ~95% of the label variance of a 2 s
label (depth 14: ~96.5%, depth 16: ~98%). This tool exists so that depth-vs-volume can be
tested as an experiment instead of being argued. It does NOT replace the 500 ms teacher data
(that stays the validation / tactical set); it makes the *bulk* of the training set affordable.

Input : lines `FEN | cp | wdl` as written by `chess-rust-engine datagen N DEPTH out` (game order kept).
Output: 28-byte records identical to tools/nnue/train.py (occ u64 | 16B nibbles | stm u8 | score i16 |
        wdl u8), written IN INPUT ORDER (so `--split-block` style game-disjoint validation stays valid),
        plus `<out>.provenance.json` (SF version, depth, hash, threads, host, input sha256, counts).
Labels: Stockfish 19, UCI_LimitStrength off, score from the side to move, mates clipped to +-3000.
Refuses depth < 10 (EXP-P1: d8/d10 leave ~7% of label variance unexplained and cost only 10 ms less).

Example:
  CRE_DATAGEN_SEED=7 ./target/release/chess-rust-engine datagen 200000 6 /tmp/sp7.txt
  python3 tools/nnue/relabel_depth.py --stockfish ./stockfish --in /tmp/sp7.txt --out data/d12-s7.bin --depth 12
"""
import argparse, hashlib, json, multiprocessing as mp, os, platform, struct, sys, time
import chess, chess.engine

REC = struct.Struct("<Q16sBhB")  # 28 bytes
assert REC.size == 28
MIN_DEPTH = 10


def pack(board, score, wdl):
    occ, nib = 0, []
    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None:
            occ |= 1 << sq
            nib.append((0 if p.color else 6) + p.piece_type - 1)
    nib += [0] * (32 - len(nib))
    b = bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(16))
    return REC.pack(occ, b, 0 if board.turn else 1, max(-3000, min(3000, score)), wdl)


def work(args):
    sf, depth, hash_mb, chunk = args
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    eng.configure({"Threads": 1, "Hash": hash_mb, "UCI_LimitStrength": False})
    out, dropped = [], 0
    for fen, wdl in chunk:
        b = chess.Board(fen)
        if b.is_game_over() or b.is_check():
            dropped += 1
            continue
        info = eng.analyse(b, chess.engine.Limit(depth=depth))
        out.append(pack(b, info["score"].relative.score(mate_score=3000), wdl))
    ver = eng.id.get("name", "?")
    eng.quit()
    return b"".join(out), len(out), dropped, ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--hash", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000, help="positions per task; order of chunks is preserved")
    a = ap.parse_args()
    if a.depth < MIN_DEPTH:
        sys.exit(f"refusing depth {a.depth} < {MIN_DEPTH}: label noise too high (see experiments/EXP-P1)")
    rows, raw = [], open(a.inp, "rb").read()
    for line in raw.decode().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            rows.append((parts[0], int(round(float(parts[2]) * 2))))  # wdl 0/0.5/1 (stm) -> 0/1/2
    chunks = [(a.stockfish, a.depth, a.hash, rows[i : i + a.chunk]) for i in range(0, len(rows), a.chunk)]
    t0 = time.time()
    n = dropped = 0
    ver = "?"
    with mp.Pool(a.procs) as pool, open(a.out, "wb") as f:
        for data, k, d, ver in pool.imap(work, chunks):  # imap keeps input order
            f.write(data)
            n += k
            dropped += d
            print(f"{n}/{len(rows)} labelled {n/(time.time()-t0):.0f} pos/s", file=sys.stderr, flush=True)
    prov = {
        "tool": "tools/nnue/relabel_depth.py", "teacher": ver, "uci_limit_strength": False, "depth": a.depth,
        "threads_per_engine": 1, "procs": a.procs, "hash_mb": a.hash, "score_pov": "side_to_move",
        "mate_cp": 3000, "wdl_pov": "side_to_move (0 loss,1 draw,2 win)", "records": n, "dropped_check_or_over": dropped,
        "input": os.path.basename(a.inp), "input_sha256": hashlib.sha256(raw).hexdigest(), "order": "input order preserved",
        "host": platform.platform(), "cpus": os.cpu_count(), "seconds": round(time.time() - t0, 1),
        "note": "fixed-depth bulk labels; NOT the >=500ms teacher set. History/game-id not recorded.",
    }
    json.dump(prov, open(a.out + ".provenance.json", "w"), indent=2)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
