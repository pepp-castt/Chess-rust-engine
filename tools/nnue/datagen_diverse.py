#!/usr/bin/env python3
"""Diversified SF19-MAX teacher data (Agent E).

Why this exists: Agent D's audit of the first 500 ms shard (teacher-42595442579) found 62.6% of
records with |score| < 50 cp, 19.6% exactly 0, and only 15.1% with |score| >= 300. Stockfish playing
its single best move from ~40 balanced mainlines produces long balanced games, so the net sees few of
the imbalanced positions that decide games.

What this does differently (labels are still SF19 MAX, full strength, >= 500 ms per position):
  * MultiPV sampling: on a fraction of plies the side to move plays the 2nd/3rd best move *if it is
    within --window cp of the best*. The resulting position is realistic (SF itself rates the move as
    playable) but off the beaten path and often imbalanced. SF then has to find its way, and every
    position along the way is labelled by a full >= 500 ms search.
  * The LABEL of a position is always SF's best-line score for that position (PV1), never the score
    of the sampled move.
  * Optional imbalance-biased seeds (--seed-eval-min/max): accept seed positions whose 500 ms eval is
    in a chosen |cp| band instead of near 0, so some games start from a real advantage.
  * Mate scores are saturated to +-3000 exactly as datagen_movetime.py does.

Output: same 28-byte records as datagen_movetime.py / train.py (occ u64 | 16 piece nibbles | stm |
score i16 | wdl u8).  SF is a teacher of labels only; nothing of it is used at runtime.
"""
import argparse, os, random, sys, time, multiprocessing as mp
import chess, chess.engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datagen_movetime import pack, book_lines, random_endgame, MIN_MS  # noqa: E402


def cp_of(score, turn):
    s = score.pov(turn)
    if s.is_mate():
        return 32000 if s.mate() > 0 else -32000
    return s.score()


def worker(args):
    wid, sf, minutes, mt_ms, out, seed, book, p_sample, window, seed_lo, seed_hi, sample_from, sample_to = args
    rnd = random.Random(seed * 1000 + wid)
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    eng.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": False})
    t_end = time.time() + minutes * 60
    games = npos = nsampled = 0
    f = open(out, "ab")
    while time.time() < t_end:
        endgame = rnd.random() < 0.10
        if endgame:
            board = random_endgame(rnd)
            if board is None:
                continue
        else:
            board = chess.Board(rnd.choice(book)) if book else chess.Board()
            for _ in range(rnd.randint(0, 2)):
                board.push(rnd.choice(list(board.legal_moves)))
                if board.is_game_over():
                    break
            if board.is_game_over():
                continue
            info = eng.analyse(board, chess.engine.Limit(time=mt_ms / 1000.0))
            wcp = cp_of(info["score"], chess.WHITE)
            if abs(wcp) >= 32000 or not (seed_lo <= abs(wcp) <= seed_hi):
                continue
        recs = []
        win_run, win_side, draw_run, result = 0, None, 0, None
        while True:
            if board.is_game_over(claim_draw=True) or board.ply() > 400:
                break
            t = max(MIN_MS, int(mt_ms * rnd.uniform(0.9, 1.3))) / 1000.0
            sample_now = (not endgame and sample_from <= board.ply() <= sample_to
                          and rnd.random() < p_sample and board.legal_moves.count() > 3)
            infos = eng.analyse(board, chess.engine.Limit(time=t), multipv=3 if sample_now else 1)
            if not isinstance(infos, list):
                infos = [infos]
            best = infos[0]
            if not best.get("pv"):
                break
            cp = cp_of(best["score"], board.turn)
            best_move = best["pv"][0]
            move = best_move
            if sample_now and len(infos) > 1 and abs(cp) < 20000:
                cands = []
                for inf in infos[1:]:
                    if inf.get("pv"):
                        c2 = cp_of(inf["score"], board.turn)
                        if abs(c2) < 20000 and cp - c2 <= window:
                            cands.append(inf["pv"][0])
                if cands:
                    move = rnd.choice(cands)
                    nsampled += 1
            # the label belongs to the position and to SF's best line, whatever move we then play
            quiet = (not board.is_check()) and (not board.is_capture(best_move)) and best_move.promotion is None
            if quiet:
                recs.append((pack(board, max(-3000, min(3000, cp)), 1)[:-1], board.turn))
            white_cp = cp if board.turn else -cp
            if abs(white_cp) >= 1500:
                side = white_cp > 0
                win_run = win_run + 1 if win_side == side else 1
                win_side = side
                if win_run >= (12 if endgame else 6):
                    result = 1.0 if side else 0.0
                    break
            else:
                win_run, win_side = 0, None
            if board.ply() > 90 and abs(white_cp) <= 8:
                draw_run += 1
                if draw_run >= 14:
                    result = 0.5
                    break
            else:
                draw_run = 0
            board.push(move)
        if result is None:
            if board.is_game_over(claim_draw=True):
                o = board.outcome(claim_draw=True)
                result = 0.5 if (o is None or o.winner is None) else (1.0 if o.winner else 0.0)
            else:
                result = 0.5
        for body, turn in recs:
            w = result if turn else 1.0 - result
            f.write(body + bytes([0 if w == 0.0 else (1 if w == 0.5 else 2)]))
        f.flush()
        games += 1
        npos += len(recs)
        if games % 10 == 0:
            print(f"[w{wid}] games={games} positions={npos} sampled_moves={nsampled}", flush=True)
    eng.quit()
    f.close()
    return games, npos, nsampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--movetime-ms", type=int, default=500, help="per move, FLOOR 500")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", default="data/diverse.bin")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p-sample", type=float, default=0.18, help="chance a ply plays a MultiPV runner-up")
    ap.add_argument("--window", type=int, default=70, help="runner-up must be within this many cp of best")
    ap.add_argument("--sample-from", type=int, default=6)
    ap.add_argument("--sample-to", type=int, default=60)
    ap.add_argument("--seed-eval-min", type=int, default=0, help="min |eval| (cp) of an accepted opening seed")
    ap.add_argument("--seed-eval-max", type=int, default=250, help="max |eval| (cp) of an accepted opening seed")
    a = ap.parse_args()
    if a.movetime_ms < MIN_MS:
        print(f"refusing movetime {a.movetime_ms} ms < {MIN_MS} ms (owner rule: no short games)", file=sys.stderr)
        sys.exit(2)
    book = book_lines()
    print(f"book lines {len(book)} movetime {a.movetime_ms} ms p_sample {a.p_sample} window {a.window} cp", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    jobs = [(i, a.stockfish, a.minutes, a.movetime_ms, f"{a.out}.{i}", a.seed, book, a.p_sample, a.window,
             a.seed_eval_min, a.seed_eval_max, a.sample_from, a.sample_to) for i in range(a.procs)]
    with mp.Pool(a.procs) as pool:
        res = pool.map(worker, jobs)
    with open(a.out, "wb") as out:
        for i in range(a.procs):
            p = f"{a.out}.{i}"
            if os.path.exists(p):
                out.write(open(p, "rb").read())
                os.remove(p)
    print(f"DONE games={sum(r[0] for r in res)} positions={sum(r[1] for r in res)} "
          f"sampled_moves={sum(r[2] for r in res)} bytes={os.path.getsize(a.out)}")


if __name__ == "__main__":
    main()
