#!/usr/bin/env python3
"""Parallel NextGen-vs-Stockfish runner with REAL clocks (local / one-box use).

Why this exists next to scripts/stockfish_match.py and the fastchess workflow:
  * scripts/stockfish_match.py replays the same 10 openings in the same order and, in
    teacher mode, runs an extra SF analysis after every move (slow).
  * fastchess (sf19-match.yml) is the right tool on CI. This script is for a laptop /
    sandbox without fastchess, and it writes per-move `score/depth time` comments into
    the PGN, which tools/diagnose_weakness.py uses to attribute mistakes.

Rules enforced (shared owner rules):
  * SF19 runs with UCI_LimitStrength OFF unless --sf-elo is given (ladder).
  * Time control base must be >= 5 s (no 100-120 ms games).
  * Each opening is played with BOTH colours (pairs), openings come from an EPD book.

Example:
  python3 tools/play_max.py --stockfish ./stockfish --games 16 --workers 2 \
      --tc 10+0.1 --book tools/openings_book.epd --pgn results/max.pgn
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

MIN_BASE_S = 5.0


def parse_tc(s: str) -> tuple[float, float]:
    base, _, inc = s.partition("+")
    return float(base), float(inc or 0)


def load_book(path: str) -> list[str]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split(";")[0].split()
        if len(f) >= 4:
            out.append(" ".join(f[:4]) + " 0 1")
    return out


def fmt_score(info) -> str:
    sc = info.get("score")
    if sc is None:
        return "?"
    s = sc.white()
    if s.is_mate():
        return f"M{s.mate()}"
    return f"{s.score() / 100:+.2f}"


def play_one(task):
    (idx, fen, ours_path, sf_path, tc, hash_mb, sf_elo, max_ply) = task
    base, inc = parse_tc(tc)
    our_white = idx % 2 == 0
    board = chess.Board(fen)
    game = chess.pgn.Game()
    game.setup(board)
    game.headers.update(
        {
            "Event": "NextGen vs Stockfish 19 " + ("MAX" if not sf_elo else f"UCI_Elo {sf_elo}"),
            "White": "NextGen" if our_white else "Stockfish 19",
            "Black": "Stockfish 19" if our_white else "NextGen",
            "TimeControl": tc,
            "Round": str(idx + 1),
        }
    )
    ours = chess.engine.SimpleEngine.popen_uci(ours_path)
    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        ours.configure({"Hash": hash_mb, "Threads": 1})
        opts = {"Hash": hash_mb, "Threads": 1}
        if sf_elo:
            opts.update({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
        else:
            opts["UCI_LimitStrength"] = False
        sf.configure(opts)
        clock = {chess.WHITE: base, chess.BLACK: base}
        node = game
        term = None
        forced = None
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_ply:
            side = board.turn
            is_ours = (side == chess.WHITE) == our_white
            eng = ours if is_ours else sf
            lim = chess.engine.Limit(
                white_clock=max(0.01, clock[chess.WHITE]),
                black_clock=max(0.01, clock[chess.BLACK]),
                white_inc=inc,
                black_inc=inc,
            )
            t0 = time.perf_counter()
            try:
                r = eng.play(board, lim, info=chess.engine.INFO_BASIC | chess.engine.INFO_SCORE)
            except chess.engine.EngineError as e:
                term = f"engine error: {e}"
                break
            dt = time.perf_counter() - t0
            clock[side] = clock[side] - dt + inc
            if clock[side] < 0:
                term = "time forfeit"
                forced = "0-1" if side == chess.WHITE else "1-0"
                break
            if r.move is None or r.move not in board.legal_moves:
                term = "illegal move"
                forced = "0-1" if side == chess.WHITE else "1-0"
                break
            node = node.add_variation(r.move)
            node.comment = f"{fmt_score(r.info)}/{r.info.get('depth', 0)} {dt:.2f}s"
            board.push(r.move)
        res = forced or board.result(claim_draw=True)
        if res == "*":
            res = "1/2-1/2"
            term = "ply cap"
        game.headers["Result"] = res
        if term:
            game.headers["Termination"] = term
        return str(game) + "\n\n", game.headers["Result"], our_white
    finally:
        for e in (ours, sf):
            try:
                e.quit()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="target/release/chess-rust-engine")
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2, help="keep 2*workers <= physical cores")
    ap.add_argument("--tc", default="10+0.1")
    ap.add_argument("--hash-mb", type=int, default=64)
    ap.add_argument("--sf-elo", type=int, default=0, help="0 = full strength (default)")
    ap.add_argument("--book", default="tools/openings_book.epd")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ply", type=int, default=300)
    ap.add_argument("--pgn", default="results/max.pgn")
    a = ap.parse_args()
    if parse_tc(a.tc)[0] < MIN_BASE_S:
        print(f"refusing TC base < {MIN_BASE_S}s: short games teach junk", file=sys.stderr)
        return 2
    book = load_book(a.book)
    rng = random.Random(a.seed)
    rng.shuffle(book)
    tasks = []
    for i in range(a.games):
        fen = book[(i // 2) % len(book)]
        tasks.append((i, fen, a.ours, a.stockfish, a.tc, a.hash_mb, a.sf_elo, a.max_ply))
    Path(a.pgn).parent.mkdir(parents=True, exist_ok=True)
    w = d = l = 0
    with open(a.pgn, "w") as out, mp.Pool(a.workers) as pool:
        for text, res, our_white in pool.imap_unordered(play_one, tasks):
            out.write(text)
            out.flush()
            if res == "1/2-1/2":
                d += 1
            elif (res == "1-0") == our_white:
                w += 1
            else:
                l += 1
            print(f"W-D-L {w}-{d}-{l}", flush=True)
    n = w + d + l
    print(f"summary W-D-L {w}-{d}-{l} score {(w + d / 2) / max(1, n):.3f} n={n} pgn={a.pgn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
