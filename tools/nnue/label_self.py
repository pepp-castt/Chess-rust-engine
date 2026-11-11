#!/usr/bin/env python3
"""Stage-0 data: label positions with THIS engine's own fixed-depth search (no Stockfish).

Why: E measured the first net at -676 Elo vs HCE because it has no sense of what material is
worth (startpos minus a rook: HCE -434, net -159).  Stage 0 teaches the net the engine's own
material/PST/tactical knowledge at volume for free; Stockfish labels (>=500 ms, SF19 MAX) then
refine it in stage 2.  Nothing here calls Stockfish and nothing here is a Stockfish label.

Seeds are 28-byte records in B's format (tools/nnue/train.py REC).  For each seed we optionally
remove 1-2 non-king pieces (material imbalance), then walk the engine's own best-move line for a
few plies (with a little noise) and label every quiet position:

    quiet = side to move not in check, best move not a capture / promotion, no mate score.

Output: the same 28-byte record (occ u64 | 16 nibble bytes | stm | score i16 stm-POV | wdl u8).
wdl is always 1 (unknown) -- train with `--lam 1.0` so the result term is ignored.

    python tools/nnue/label_self.py --engine target/release/chess-rust-engine \
        --seeds 'data/teacher-*.bin' --out data/self-d7.bin --depth 7 --minutes 20 --procs 4
"""
import argparse, glob, multiprocessing as mp, os, random, struct, subprocess, sys, time
import chess

REC = struct.Struct("<Q16sBhB")  # 28 bytes, identical to datagen_movetime.py
assert REC.size == 28


def pack(board: chess.Board, score: int) -> bytes:
    occ, nib = 0, []
    for sq in range(64):
        p = board.piece_at(sq)
        if p:
            occ |= 1 << sq
            nib.append((0 if p.color else 6) + p.piece_type - 1)
    nib += [0] * (32 - len(nib))
    b = bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(16))
    return REC.pack(occ, b, 0 if board.turn else 1, max(-32000, min(32000, score)), 1)


def unpack_board(rec: bytes) -> chess.Board:
    occ, pcs, stm, _score, _wdl = REC.unpack(rec)
    nib = []
    for byte in pcs:
        nib += [byte & 15, byte >> 4]
    bd = chess.Board(None)
    i = 0
    for sq in range(64):
        if occ >> sq & 1:
            n = nib[i]
            i += 1
            bd.set_piece_at(sq, chess.Piece(n % 6 + 1, n < 6))
    bd.turn = chess.WHITE if stm == 0 else chess.BLACK
    return bd


class Engine:
    def __init__(self, path, hash_mb=16, nnue=False):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci"); self.wait("uciok")
        self.send(f"setoption name Hash value {hash_mb}")
        self.send(f"setoption name UseNNUE value {'true' if nnue else 'false'}")
        self.send("isready"); self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def wait(self, tok):
        for line in self.p.stdout:
            if line.startswith(tok):
                return line
        raise RuntimeError("engine died")

    def search(self, fen, depth):
        """-> (score_cp or None if mate, bestmove uci)"""
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        score = None
        for line in self.p.stdout:
            if line.startswith("info") and " score " in line:
                t = line.split()
                k = t.index("score")
                score = None if t[k + 1] == "mate" else int(t[k + 2])
            elif line.startswith("bestmove"):
                return score, line.split()[1]
        raise RuntimeError("engine died")

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        self.p.kill()


def perturb(board: chess.Board, rnd: random.Random) -> chess.Board:
    """Remove 1-2 non-king pieces; keep the position legal (opponent not in check)."""
    for _ in range(8):
        b = board.copy()
        cand = [sq for sq, p in b.piece_map().items() if p.piece_type != chess.KING]
        if len(cand) < 6:
            return board
        for sq in rnd.sample(cand, rnd.choice((1, 1, 2))):
            b.remove_piece_at(sq)
        b.clear_stack()
        if b.is_valid() and not b.is_check():
            return b
    return board


def worker(a):
    wid, args, seeds = a
    rnd = random.Random(args.seed * 1000 + wid)
    eng = Engine(args.engine)
    out = open(f"{args.out}.part{wid}", "wb")
    t_end = time.time() + args.minutes * 60
    n = seen = 0
    try:
        while time.time() < t_end:
            b = unpack_board(rnd.choice(seeds))
            if not b.is_valid():
                continue
            if rnd.random() < args.perturb_p:
                b = perturb(b, rnd)
            big = 0
            for ply in range(args.walk):
                if time.time() >= t_end or b.is_game_over(claim_draw=False):
                    break
                score, bm = eng.search(b.fen(), args.depth)
                if score is None:  # mate score: not a useful regression target
                    break
                mv = chess.Move.from_uci(bm)
                seen += 1
                quiet = not b.is_check() and not b.is_capture(mv) and mv.promotion is None
                if quiet and abs(score) <= args.max_cp and b.halfmove_clock < 80:
                    out.write(pack(b, score)); n += 1
                big = big + 1 if abs(score) > 1500 else 0
                if big >= 3:
                    break
                if rnd.random() < args.noise:  # keep walks from collapsing onto one line
                    mv = rnd.choice(list(b.legal_moves))
                b.push(mv)
            if seen % 500 < args.walk:
                out.flush()
    finally:
        out.close(); eng.close()
    return n, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated globs of 28-byte record files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--walk", type=int, default=10, help="max plies walked per seed")
    ap.add_argument("--perturb-p", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--max-cp", type=int, default=2500)
    ap.add_argument("--max-seeds", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.depth < 6:
        sys.exit("depth < 6 labels are too shallow to be a useful teacher; refuse")
    files = sorted({f for g in args.seeds.split(",") for f in glob.glob(g.strip())})
    seeds = []
    for f in files:
        raw = open(f, "rb").read()
        seeds += [raw[i:i + 28] for i in range(0, len(raw) - 27, 28)]
    random.Random(args.seed).shuffle(seeds)
    seeds = seeds[: args.max_seeds]
    print(f"{len(files)} files, {len(seeds)} seeds, depth {args.depth}, {args.procs} procs, "
          f"{args.minutes} min", flush=True)
    with mp.Pool(args.procs) as pool:
        res = pool.map(worker, [(i, args, seeds) for i in range(args.procs)])
    tot = sum(r[0] for r in res)
    with open(args.out, "wb") as fo:
        for i in range(args.procs):
            p = f"{args.out}.part{i}"
            fo.write(open(p, "rb").read()); os.remove(p)
    print(f"labelled {tot} quiet positions of {sum(r[1] for r in res)} searched "
          f"-> {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)


if __name__ == "__main__":
    main()
