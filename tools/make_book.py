#!/usr/bin/env python3
"""Build a real, balanced opening book (EPD) for SPRT / ladder matches.

Why: tools/openings.epd is random-legal junk (1.a4 f5 ..., pieces in odd places)
- engines play a different game from it than from real chess positions. This starts
from the ~40 theory mainlines already embedded in tools/nnue/datagen_movetime.py,
cuts them at 6/8/10 plies, optionally extends them by 0/2/4 plies of *sound*
Stockfish moves (sampled from MultiPV, within --band cp of the best), then keeps
only positions Stockfish 19 evaluates as balanced at >= 0.5 s.

    python3 tools/make_book.py --stockfish ./stockfish --out tools/openings_book.epd

Written by Agent C. Needs: pip install chess
"""
import argparse
import multiprocessing as mp
import random
import re
import sys

import chess
import chess.engine

SF = None


def init(path, hash_mb):
    global SF
    SF = chess.engine.SimpleEngine.popen_uci(path)
    SF.configure({"Threads": 1, "Hash": hash_mb})


def load_lines(src):
    s = open(src).read()
    m = re.search(r'OPENINGS\s*=\s*"""(.*?)"""', s, re.S)
    return [ln.split() for ln in m.group(1).strip().splitlines() if ln.strip()]


def prefix(line, plies):
    b = chess.Board()
    try:
        for san in line[:plies]:
            b.push_san(san)
    except ValueError:
        return None
    return b


def work(job):
    fen, ext, seed, band, ext_ms, bal_ms, max_cp = job
    rnd = random.Random(seed)
    b = chess.Board(fen)
    for _ in range(ext):
        if b.is_game_over():
            return None
        infos = SF.analyse(b, chess.engine.Limit(time=ext_ms), multipv=4)
        scored = []
        for i in infos:
            sc = i["score"].pov(b.turn)
            if sc.is_mate() or "pv" not in i:
                continue
            scored.append((sc.score(), i["pv"][0]))
        if not scored:
            return None
        best = max(s for s, _ in scored)
        pool = [m for s, m in scored if s >= best - band]
        b.push(rnd.choice(pool))
    if b.is_game_over() or b.is_check():
        return None
    info = SF.analyse(b, chess.engine.Limit(time=bal_ms))
    sc = info["score"].white()
    if sc.is_mate() or abs(sc.score()) > max_cp:
        return None
    return b.epd(), sc.score()


def key(epd):
    return " ".join(epd.split()[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--src", default="tools/nnue/datagen_movetime.py")
    ap.add_argument("--out", default="tools/openings_book.epd")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--samples", type=int, default=3, help="extension samples per (line,cut,ext)")
    ap.add_argument("--band", type=int, default=35, help="cp below best still allowed when extending")
    ap.add_argument("--max-cp", type=int, default=45, help="keep only |eval| <= this (white POV)")
    ap.add_argument("--ext-ms", type=float, default=0.3)
    ap.add_argument("--bal-ms", type=float, default=0.5, help="balance check; floor 0.5 s")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    if a.bal_ms < 0.5:
        sys.exit("refusing balance check below 0.5 s")

    lines = load_lines(a.src)
    jobs, seed = [], a.seed
    for ln in lines:
        for cut in (6, 8, 10):
            b = prefix(ln, cut)
            if b is None:
                continue
            for ext in (0, 2, 4):
                n = 1 if ext == 0 else a.samples
                for _ in range(n):
                    seed += 1
                    jobs.append((b.fen(), ext, seed, a.band, a.ext_ms, a.bal_ms, a.max_cp))
    print(f"{len(lines)} mainlines -> {len(jobs)} candidates", file=sys.stderr)

    seen, out, rejected = set(), [], 0
    with mp.Pool(a.procs, initializer=init, initargs=(a.stockfish, a.hash)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            if r is None:
                rejected += 1
            elif key(r[0]) not in seen:
                seen.add(key(r[0]))
                out.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} kept={len(out)}", file=sys.stderr)
    random.Random(a.seed).shuffle(out)
    with open(a.out, "w") as f:
        for epd, _ in out:
            f.write(epd + "\n")
    cps = sorted(abs(c) for _, c in out)
    print(f"wrote {len(out)} unique balanced positions to {a.out} "
          f"(rejected {rejected}; median |eval| {cps[len(cps)//2] if cps else 0} cp)", file=sys.stderr)


if __name__ == "__main__":
    main()
