#!/usr/bin/env python3
"""EPD regression suite runner: a FAST, low-variance strength signal next to SPRT.

Reads EPD files with `bm` (best/acceptable moves) and/or `am` (avoid moves), runs one or
more engines at fixed movetime or depth and reports solved %, with a Wilson 95% interval.
Compare two engines (BASE vs TEST) on the SAME suite:

  python3 tools/epd_suite.py results/diagnosis/hard_suite.epd \
      --engine base=./base/chess-rust-engine --engine test=./target/release/chess-rust-engine \
      --movetime 1.0 --fail-below base

Exit code 1 if --fail-below NAME is given and any other engine solves fewer than NAME
minus the CI slack (regression guard). This does NOT replace SPRT: a suite built from one
engine's own losses is biased toward that engine's weaknesses. It is a smoke alarm for
tactical regressions and a measuring stick for the specific weaknesses diagnosed.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import chess
import chess.engine


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def load(paths):
    items = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            b = chess.Board()
            try:
                ops = b.set_epd(line)
            except ValueError as e:
                print(f"skip bad EPD: {e}", file=sys.stderr)
                continue
            items.append((b.copy(), ops.get("bm", []), ops.get("am", []), ops.get("id", "?")))
    return items


def solved(board, move, bm, am):
    if move is None or move not in board.legal_moves:
        return False
    if bm and move not in bm:
        return False
    if am and move in am:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("epd", nargs="+")
    ap.add_argument("--engine", action="append", required=True, help="NAME=PATH (repeatable)")
    ap.add_argument("--movetime", type=float, default=1.0)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--fail-below", default="", help="NAME of the reference engine (regression guard)")
    a = ap.parse_args()
    items = load(a.epd)
    if not items:
        print("no positions", file=sys.stderr)
        return 2
    lim = chess.engine.Limit(depth=a.depth, time=a.movetime) if a.depth else chess.engine.Limit(time=a.movetime)
    results = {}
    for spec in a.engine:
        name, _, path = spec.partition("=")
        eng = chess.engine.SimpleEngine.popen_uci(path)
        try:
            eng.configure({"Hash": a.hash})
        except chess.engine.EngineError:
            pass
        ok = 0
        for b, bm, am, pid in items:
            r = eng.play(b, lim)
            ok += solved(b, r.move, bm, am)
        eng.quit()
        lo, hi = wilson(ok, len(items))
        results[name] = (ok, lo, hi)
        print(f"{name:12s} solved {ok}/{len(items)} = {100 * ok / len(items):5.1f}%  95% CI [{100 * lo:.1f}, {100 * hi:.1f}]")
    if a.fail_below and a.fail_below in results:
        ref = results[a.fail_below]
        bad = [n for n, v in results.items() if n != a.fail_below and v[2] < ref[1]]
        if bad:
            print(f"REGRESSION: {bad} significantly below {a.fail_below}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
