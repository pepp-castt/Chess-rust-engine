#!/usr/bin/env python3
"""Material-imbalance teacher data (Agent E).

Problem (measured, see Chess-rust-engine#1): a (768->128)x2->1 net trained on 348k SF19-MAX self-play
labels reached val MSE 0.003 but scored 1-2-97 against the hand-crafted eval, because it had almost
never seen material imbalance (it rated "White is missing a rook" as -159 cp, HCE -434).  SF self-play
is balanced by construction, so the data cannot teach what a rook is worth.

Fix: take real, quiet positions from the existing teacher shards, remove 1-2 non-king pieces, and ask
Stockfish 19 at FULL strength (UCI_LimitStrength off, >= 500 ms) what the new position is worth.  The
label is therefore Stockfish's own verdict, including compensation, fortress and tempo effects,
not a hand-written piece-value table.

Rules enforced here (the same ones the team signed):
  * movetime floor 500 ms (exit 2 below it);
  * label perspective = side to move, mate scores saturate to +-3000;
  * positions where the side to move is in check or SF's best move is a capture/promotion are skipped
    (static eval is not meaningful there - same rule as datagen_movetime.py);
  * the perturbed position must be legal (python-chess is_valid) and not game-over.

wdl byte: there is no game, so it is derived from SF's score (>=+200 win, <=-200 loss, else draw).
D showed the wdl byte of the existing data is already a near-deterministic function of the score, so
train with --lam 1.0 on these records (the result term carries no independent signal here).

Output: same 28-byte records as datagen_movetime.py / train.py.
"""
import argparse, os, random, struct, sys, time, multiprocessing as mp
import chess, chess.engine

REC = struct.Struct("<Q16sBhB")
MIN_MS = 500
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def unpack(raw: bytes):
    occ, pcs, stm, score, wdl = REC.unpack(raw)
    nib = []
    for b in pcs:
        nib += [b & 15, b >> 4]
    board = chess.Board(None)
    i = 0
    for sq in range(64):
        if occ >> sq & 1:
            p = nib[i]
            i += 1
            board.set_piece_at(sq, chess.Piece(p % 6 + 1, chess.WHITE if p < 6 else chess.BLACK))
    board.turn = chess.WHITE if stm == 0 else chess.BLACK
    return board, score


def pack(board: chess.Board, score: int, wdl: int) -> bytes:
    occ = 0
    nib = []
    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None:
            occ |= 1 << sq
            nib.append((0 if p.color else 6) + p.piece_type - 1)
    nib += [0] * (32 - len(nib))
    b = bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(16))
    return REC.pack(occ, b, 0 if board.turn else 1, score, wdl)


def perturb(board: chess.Board, rnd: random.Random):
    """Remove 1 (70%) or 2 (30%) non-king pieces. Returns a new board or None."""
    b = board.copy(stack=False)
    n = 1 if rnd.random() < 0.7 else 2
    cands = [sq for sq, p in b.piece_map().items() if p.piece_type != chess.KING]
    if len(cands) <= n + 2:
        return None
    # weight by piece value so queens/rooks are removed as often as pawns (flat over squares would
    # remove pawns ~60% of the time and the net would still not see big imbalances)
    if rnd.random() < 0.3:
        # "trade-like" perturbation: one piece off EACH side, so the result covers small/medium
        # material differences (a knight vs a bishop, a rook vs two pawns...) and not only big ones.
        for color in (chess.WHITE, chess.BLACK):
            own = [sq for sq in cands if b.color_at(sq) == color]
            if own:
                b.remove_piece_at(rnd.choice(own))
        b.castling_rights = b.clean_castling_rights()
        b.ep_square = None
        return b if (b.is_valid() and not b.is_game_over()) else None
    kinds = {}
    for sq in cands:
        kinds.setdefault(b.piece_type_at(sq), []).append(sq)
    for _ in range(n):
        pt = rnd.choice(list(kinds))
        sq = rnd.choice(kinds[pt])
        kinds[pt].remove(sq)
        if not kinds[pt]:
            del kinds[pt]
        b.remove_piece_at(sq)
        if not kinds:
            break
    b.castling_rights = b.clean_castling_rights()
    b.ep_square = None
    if not b.is_valid() or b.is_game_over():
        return None
    return b


def worker(args):
    wid, sf, minutes, mt_ms, out, seed, src_path, min_pieces = args
    rnd = random.Random(seed * 1000 + wid)
    data = open(src_path, "rb").read()
    n_rec = len(data) // REC.size
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    eng.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": False})
    t_end = time.time() + minutes * 60
    f = open(out, "ab")
    done = tried = 0
    while time.time() < t_end:
        i = rnd.randrange(n_rec)
        board, _ = unpack(data[i * REC.size:(i + 1) * REC.size])
        tried += 1
        if chess.popcount(board.occupied) < min_pieces:
            continue
        pb = perturb(board, rnd)
        if pb is None or pb.is_check():
            continue
        t = max(MIN_MS, int(mt_ms * rnd.uniform(1.0, 1.25))) / 1000.0
        info = eng.analyse(pb, chess.engine.Limit(time=t))
        pv = info.get("pv")
        if not pv or "score" not in info:
            continue
        mv = pv[0]
        if pb.is_capture(mv) or mv.promotion is not None:
            continue
        s = info["score"].pov(pb.turn)
        cp = (32000 if s.mate() > 0 else -32000) if s.is_mate() else s.score()
        cp = max(-3000, min(3000, cp))
        wdl = 2 if cp >= 200 else (0 if cp <= -200 else 1)
        f.write(pack(pb, cp, wdl))
        f.flush()
        done += 1
        if done % 200 == 0:
            print(f"[w{wid}] labelled={done} tried={tried}", flush=True)
    eng.quit()
    f.close()
    return done, tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--src", required=True, help="teacher .bin shard to take real positions from")
    ap.add_argument("--out", default="data/perturbed.bin")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--movetime-ms", type=int, default=500, help="per position, FLOOR 500")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-pieces", type=int, default=10, help="source positions need at least this many pieces")
    a = ap.parse_args()
    if a.movetime_ms < MIN_MS:
        print(f"refusing movetime {a.movetime_ms} ms < {MIN_MS} ms (owner rule: no short games)", file=sys.stderr)
        sys.exit(2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    jobs = [(i, a.stockfish, a.minutes, a.movetime_ms, f"{a.out}.{i}", a.seed, a.src, a.min_pieces)
            for i in range(a.procs)]
    with mp.Pool(a.procs) as pool:
        res = pool.map(worker, jobs)
    with open(a.out, "wb") as out:
        for i in range(a.procs):
            p = f"{a.out}.{i}"
            if os.path.exists(p):
                out.write(open(p, "rb").read())
                os.remove(p)
    print(f"DONE labelled={sum(r[0] for r in res)} tried={sum(r[1] for r in res)} bytes={os.path.getsize(a.out)}")


if __name__ == "__main__":
    main()
