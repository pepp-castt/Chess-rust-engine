#!/usr/bin/env python3
"""Train/inference parity gate for the CRN1 net (768 -> 128)x2 -> 1, 8 buckets, SCReLU.

Why this exists: every net trained so far had good held-out loss and still lost to the HCE.
"Good val loss, bad play" has a short list of causes; the cheapest to rule out is a
disagreement between what the trainer optimises and what `src/nnue.rs` computes (feature
index, perspective, bucket, quantisation, clamping, sign). This script rules it in or out
with data, in seconds, with no training and no Stockfish.

  1. Writes a random CRN1 net (fixed seed, realistic weight scale) in the trainer's byte layout.
  2. Builds N positions (random legal playouts; both sides to move, all 8 material buckets).
  3. Computes a float reference forward pass from the *documented* feature convention using the
     dequantised weights (so any remaining difference is logic, not quantisation).
  4. Asks the real engine (`CRE_NNUE_FILE=<net>`, UCI `eval`) for its NNUE score.
  5. Fails if |engine - reference| exceeds a few cp on any position.
  6. Optional (`--trainer`, needs torch): also runs tools/nnue/train.py (`decode` + `Net`), so
     trainer == reference == engine.

Usage: python3 tools/nnue/parity_check.py --engine target/release/chess-rust-engine [--n 400] [--trainer]
Exit: 0 parity, 1 mismatch, 2 harness problem.
"""
import argparse, os, random, struct, subprocess, sys, tempfile
import numpy as np
import chess

H, FEATS, BUCKETS = 128, 768, 8
QA, QB, SCALE = 255, 64, 400


def write_random_net(path, seed):
    rng = np.random.default_rng(seed)
    ft_w = np.clip(np.round(rng.normal(0, 0.10, (FEATS, H)) * QA), -505, 505).astype("<i2")
    ft_b = np.clip(np.round(rng.normal(0.1, 0.30, H) * QA), -505, 505).astype("<i2")
    out_w = np.clip(np.round(rng.normal(0, 0.12, (BUCKETS, 2 * H)) * QB), -127, 127).astype("<i2")
    out_b = np.round(rng.normal(0, 0.05, BUCKETS) * QA * QB).astype("<i4")
    with open(path, "wb") as f:
        f.write(b"CRN1" + struct.pack("<IIII", 1, H, BUCKETS, FEATS))
        f.write(ft_w.tobytes() + ft_b.tobytes() + out_w.tobytes() + out_b.tobytes())
    return ft_w, ft_b, out_w, out_b


def features(board, persp):
    """Documented convention (src/nnue.rs header): ((piece_color != persp)*6 + type)*64 + (sq ^ 56*persp);
    white = 0, black = 1; P,N,B,R,Q,K = 0..5; a1 = 0."""
    idx = []
    for sq, p in board.piece_map().items():
        c = 0 if p.color else 1
        idx.append(((c != persp) * 6 + (p.piece_type - 1)) * 64 + (sq ^ (56 * persp)))
    return idx


def bucket_of(board):
    return min(max(len(board.piece_map()) - 2, 0) // 4, BUCKETS - 1)


def reference_cp(board, net):
    ft_w, ft_b, out_w, out_b = net
    ft_w = ft_w.astype(np.float64) / QA
    ft_b = ft_b.astype(np.float64) / QA
    out_w = out_w.astype(np.float64) / QB
    out_b = out_b.astype(np.float64) / (QA * QB)
    us = 0 if board.turn else 1
    acc_us = ft_b + ft_w[features(board, us)].sum(0)
    acc_th = ft_b + ft_w[features(board, us ^ 1)].sum(0)
    x = np.concatenate([np.clip(acc_us, 0, 1) ** 2, np.clip(acc_th, 0, 1) ** 2])
    b = bucket_of(board)
    return float((x * out_w[b]).sum() + out_b[b]) * SCALE


def make_positions(n, seed):
    rnd = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        b = chess.Board()
        for _ in range(rnd.randrange(4, 200)):
            mv = list(b.legal_moves)
            if not mv or b.is_game_over():
                break
            caps = [m for m in mv if b.is_capture(m)]  # bias to captures: reach low-piece buckets
            b.push(rnd.choice(caps) if caps and rnd.random() < 0.35 else rnd.choice(mv))
            if rnd.random() < 0.06 and not b.is_game_over():
                key = b.fen()
                if key not in seen:
                    seen.add(key)
                    out.append(b.copy())
    return out[:n]


def engine_cps(engine, net_path, boards):
    cmds = ["uci", "isready"]
    for b in boards:
        cmds += [f"position fen {b.fen()}", "eval"]
    cmds.append("quit")
    env = dict(os.environ, CRE_NNUE_FILE=net_path)
    r = subprocess.run([engine], input="\n".join(cmds) + "\n", capture_output=True, text=True, env=env, timeout=120)
    err = [l for l in r.stdout.splitlines() + r.stderr.splitlines() if "NNUE file" in l]
    if err:
        print("engine could not load the net:", err[0])
        sys.exit(2)
    vals = []
    for line in r.stdout.splitlines():
        if line.startswith("info string eval_hce"):
            tok = line.split()
            vals.append(None if tok[-1] == "none" else int(tok[-1]))
    if len(vals) != len(boards) or any(v is None for v in vals):
        print(f"harness problem: got {len(vals)} eval lines for {len(boards)} positions")
        sys.exit(2)
    return vals


def trainer_cps(net_path, boards):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch, train  # noqa: E402

    recs = np.zeros(len(boards), train.REC)
    for i, b in enumerate(boards):
        occ, nib = 0, []
        for sq in range(64):
            p = b.piece_at(sq)
            if p is not None:
                occ |= 1 << sq
                nib.append((0 if p.color else 6) + p.piece_type - 1)
        nib += [0] * (32 - len(nib))
        recs[i]["occ"] = occ
        recs[i]["pcs"] = np.array([nib[2 * j] | (nib[2 * j + 1] << 4) for j in range(16)], np.uint8)
        recs[i]["stm"] = 0 if b.turn else 1
    A, B, K, _, _ = train.decode(recs)
    net = train.Net()
    train.load_net(net, net_path)
    net.eval()
    with torch.no_grad():
        o = net(torch.from_numpy(A).long(), torch.from_numpy(B).long(), torch.from_numpy(K))
    return (o.numpy() * SCALE).tolist()


def trainer_decode_cps(net, boards):
    """Torch-free check of the trainer's data path: pull the *real* `decode()` (and REC / constants)
    out of tools/nnue/train.py with `ast`, run it on packed records, then apply the numpy forward
    pass to the feature indices it produced. Catches a decode / perspective / bucket / nibble bug
    in the trainer without needing torch. (Net.forward itself is 6 lines and covered by --trainer.)"""
    import ast
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")).read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "decode":
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("REC",) for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Tuple) for t in node.targets):
            keep.append(node)  # H, FEATS, BUCKETS = ... / QA, QB, SCALE = ...
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "train.py", "exec"), ns)
    REC = ns["REC"]
    recs = np.zeros(len(boards), REC)
    for i, b in enumerate(boards):
        occ, nib = 0, []
        for sq in range(64):
            p = b.piece_at(sq)
            if p is not None:
                occ |= 1 << sq
                nib.append((0 if p.color else 6) + p.piece_type - 1)
        nib += [0] * (32 - len(nib))
        recs[i]["occ"] = occ
        recs[i]["pcs"] = np.array([nib[2 * j] | (nib[2 * j + 1] << 4) for j in range(16)], np.uint8)
        recs[i]["stm"] = 0 if b.turn else 1
    A, B, K, _, _ = ns["decode"](recs)
    ft_w, ft_b, out_w, out_b = net
    ft = np.vstack([ft_w.astype(np.float64) / QA, np.zeros((1, H))])  # row FEATS = padding
    fb, ow, ob = ft_b / QA, out_w / QB, out_b / (QA * QB)
    res = []
    for i in range(len(boards)):
        au = np.clip(ft[A[i].astype(int)].sum(0) + fb, 0, 1) ** 2
        at = np.clip(ft[B[i].astype(int)].sum(0) + fb, 0, 1) ** 2
        k = int(K[i])
        res.append(float((np.concatenate([au, at]) * ow[k]).sum() + ob[k]) * SCALE)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="target/release/chess-rust-engine")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=3.0, help="max |engine - reference| in cp (integer rounding only)")
    ap.add_argument("--trainer", action="store_true", help="also compare tools/nnue/train.py forward (needs torch)")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "random.nnue")
        net = write_random_net(path, a.seed)
        boards = make_positions(a.n, a.seed)
        eng = engine_cps(a.engine, path, boards)
        ref = [reference_cp(b, net) for b in boards]
        trn = trainer_cps(path, boards) if a.trainer else None
        dec = trainer_decode_cps(net, boards)

    cov = np.bincount([bucket_of(b) for b in boards], minlength=BUCKETS)
    stm = sum(1 for b in boards if b.turn)
    print(f"positions={len(boards)} white-to-move={stm} black-to-move={len(boards)-stm} bucket-coverage={cov.tolist()}")
    sat = sum(1 for v in eng if abs(v) >= 3000)
    print(f"engine eval range [{min(eng)}, {max(eng)}] cp, {sat} at the +-3000 clamp, std {np.std(eng):.1f}")
    ok = [(e, r) for e, r in zip(eng, ref) if abs(e) < 3000]
    worst = max(abs(e - r) for e, r in ok)
    bad = [i for i, (e, r) in enumerate(zip(eng, ref)) if abs(e) < 3000 and abs(e - r) > a.tol]
    print(f"engine vs float reference: worst |diff| {worst:.2f} cp, {len(bad)} positions over tol {a.tol}")
    if trn is not None:
        tb = [i for i, (t, r) in enumerate(zip(trn, ref)) if abs(t - r) > 0.5]
        print(f"trainer vs float reference: worst |diff| {max(abs(t - r) for t, r in zip(trn, ref)):.4f} cp, {len(tb)} over 0.5")
        bad += tb
    db = [i for i, (t, r) in enumerate(zip(dec, ref)) if abs(t - r) > 0.5]
    print(f"trainer decode() path vs float reference: worst |diff| {max(abs(t - r) for t, r in zip(dec, ref)):.4f} cp, {len(db)} over 0.5")
    bad += db
    if np.std(eng) < 20:
        print("degenerate test net (eval variance too small to detect anything)")
        sys.exit(2)
    if bad:
        for i in bad[:8]:
            print(f"  MISMATCH engine={eng[i]} ref={ref[i]:.1f}" + (f" trainer={trn[i]:.1f}" if trn else "") + f"  {boards[i].fen()}")
        sys.exit(1)
    print("PARITY OK")


if __name__ == "__main__":
    main()
