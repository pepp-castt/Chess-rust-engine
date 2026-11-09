#!/usr/bin/env python3
"""Teacher data: Stockfish 19 (full strength, UCI_LimitStrength off) plays itself with a
real node budget per move (default 60k nodes ~ depth 12-14), NOT 100-120 ms blitz.
We store quiet positions as packed 28-byte records:
    occ u64 | 16 bytes of piece nibbles (color*6+type, ascending square) | stm u8 |
    score i16 (SF cp, side-to-move POV) | wdl u8 (0 loss, 1 draw, 2 win, stm POV)
Stockfish is only a *teacher of labels*; no Stockfish network is copied anywhere.
"""
import argparse, os, random, struct, sys, time, multiprocessing as mp
import chess, chess.engine

REC = struct.Struct("<Q16sBhB")  # 28 bytes
assert REC.size == 28


def pack(board: chess.Board, score: int, wdl: int) -> bytes:
    occ = 0
    nib = []
    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None:
            occ |= 1 << sq
            nib.append((0 if p.color else 6) + p.piece_type - 1)  # white=0..5 black=6..11
    nib += [0] * (32 - len(nib))
    b = bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(16))
    return REC.pack(occ, b, 0 if board.turn else 1, score, wdl)


def worker(args):
    wid, sf, minutes, nodes, out, seed = args
    rnd = random.Random(seed * 1000 + wid)
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    eng.configure({"Threads": 1, "Hash": 32})
    t_end = time.time() + minutes * 60
    games = npos = 0
    f = open(out, "ab")
    while time.time() < t_end:
        board = chess.Board()
        # random opening: 8-12 plies, then require a balanced position
        for _ in range(rnd.randint(8, 12)):
            mv = rnd.choice(list(board.legal_moves))
            board.push(mv)
            if board.is_game_over():
                break
        if board.is_game_over():
            continue
        info = eng.analyse(board, chess.engine.Limit(nodes=nodes // 2))
        sc = info["score"].pov(chess.WHITE)
        if sc.is_mate() or abs(sc.score()) > 250:
            continue
        recs = []  # (board copy, score_stm)
        win_run = 0  # consecutive plies with |score|>1200 favouring the same side
        win_side = None
        draw_run = 0
        result = None
        while True:
            if board.is_game_over(claim_draw=True) or board.ply() > 400:
                break
            n = max(5000, int(nodes * rnd.uniform(0.75, 1.25)))
            r = eng.play(board, chess.engine.Limit(nodes=n), info=chess.engine.INFO_SCORE)
            if r.move is None:
                break
            s = r.info["score"].pov(board.turn)
            if s.is_mate():
                cp = 32000 if s.mate() > 0 else -32000
            else:
                cp = s.score()
            quiet = (not board.is_check()) and (not board.is_capture(r.move)) and r.move.promotion is None
            if quiet and abs(cp) < 3000:
                recs.append((pack(board, cp, 1)[:-1], board.turn))
            # adjudication by SF's own score (teacher), generous thresholds
            white_cp = cp if board.turn else -cp
            if abs(white_cp) >= 1500:
                side = white_cp > 0
                win_run = win_run + 1 if win_side == side else 1
                win_side = side
                if win_run >= 6:
                    result = 1.0 if side else 0.0
                    break
            else:
                win_run = 0
                win_side = None
            if board.ply() > 90 and abs(white_cp) <= 8:
                draw_run += 1
                if draw_run >= 14:
                    result = 0.5
                    break
            else:
                draw_run = 0
            board.push(r.move)
        if result is None:
            if board.is_game_over(claim_draw=True):
                o = board.outcome(claim_draw=True)
                result = 0.5 if (o is None or o.winner is None) else (1.0 if o.winner else 0.0)
            else:
                result = 0.5
        for body, turn in recs:
            # body is pack() minus the wdl byte; patch score/wdl POV
            w = result if turn else 1.0 - result
            f.write(body + bytes([0 if w == 0.0 else (1 if w == 0.5 else 2)]))
        f.flush()
        games += 1
        npos += len(recs)
        if games % 10 == 0:
            print(f"[w{wid}] games={games} positions={npos}", flush=True)
    eng.quit()
    f.close()
    return games, npos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--nodes", type=int, default=60000)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", default="data/shard.bin")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    jobs = [(i, a.stockfish, a.minutes, a.nodes, f"{a.out}.{i}", a.seed) for i in range(a.procs)]
    with mp.Pool(a.procs) as pool:
        res = pool.map(worker, jobs)
    with open(a.out, "wb") as out:
        for i in range(a.procs):
            p = f"{a.out}.{i}"
            if os.path.exists(p):
                out.write(open(p, "rb").read())
                os.remove(p)
    g = sum(r[0] for r in res)
    n = sum(r[1] for r in res)
    print(f"DONE games={g} positions={n} bytes={os.path.getsize(a.out)}")


if __name__ == "__main__":
    main()
