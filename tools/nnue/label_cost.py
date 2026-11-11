#!/usr/bin/env python3
"""Label cost vs label quality for Stockfish 19 MAX (Agent P, EXP-P1). Reproducible.

  run    : label every FEN at depth 8/10/12/14/16, 500 ms and 2000 ms (the reference), 1 thread each.
  report : MSE in sigmoid(cp/400) space vs the 2000 ms label, cost, and positions per core-hour.

  python3 tools/nnue/label_cost.py run    --stockfish ./stockfish --fens experiments/EXP-P1-data/fens300.txt --out /tmp/lc.json
  python3 tools/nnue/label_cost.py report --data experiments/EXP-P1-data/labelcost.json

CAVEAT (stated in the report): `t500` and `t2000` are the same engine on the same position, so
t500-vs-t2000 understates the true noise of a 500 ms label (shared search path). Depth labels
d8..d16 are shorter searches of the same iterative-deepening run family, so they are correlated
with the reference too. Treat every row as a LOWER bound on label noise.
"""
import argparse, json, math, multiprocessing as mp, statistics as st, time

NAMES = ["d8", "d10", "d12", "d14", "d16", "t500", "t2000"]


def work(args):
    import chess, chess.engine
    sf, chunk = args
    lims = [("d8", chess.engine.Limit(depth=8)), ("d10", chess.engine.Limit(depth=10)), ("d12", chess.engine.Limit(depth=12)),
            ("d14", chess.engine.Limit(depth=14)), ("d16", chess.engine.Limit(depth=16)),
            ("t500", chess.engine.Limit(time=0.5)), ("t2000", chess.engine.Limit(time=2.0))]
    e = chess.engine.SimpleEngine.popen_uci(sf)
    e.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": False})
    out = []
    for fen in chunk:
        b = chess.Board(fen)
        row = {"fen": fen}
        for name, lim in lims:
            t = time.time()
            i = e.analyse(b, lim)
            row[name] = [i["score"].relative.score(mate_score=3000), time.time() - t, i.get("nodes", 0)]
        out.append(row)
    e.quit()
    return out


def run(a):
    fens = [l.strip() for l in open(a.fens) if l.strip()]
    chunks = [(a.stockfish, fens[i :: a.procs]) for i in range(a.procs)]
    with mp.Pool(a.procs) as p:
        res = sum(p.map(work, chunks), [])
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out, len(res))


def sig(c):
    return 1 / (1 + math.exp(-max(-3000, min(3000, c)) / 400))


def report(a):
    R = json.load(open(a.data))
    ref = lambda r: sig(r["t2000"][0])
    var = st.pstdev([ref(r) for r in R]) ** 2
    print(f"N={len(R)} positions; variance of sigmoid(label) = {var:.4f}")
    print(f"{'label':6} {'ms/pos':>7} {'kN/pos':>7} {'MSE vs t2000':>13} {'MSE/var':>8} {'pos/core-hour':>14}")
    for n in NAMES:
        ms = st.mean(r[n][1] for r in R) * 1000
        mse = st.mean((sig(r[n][0]) - ref(r)) ** 2 for r in R)
        print(f"{n:6} {ms:7.0f} {st.mean(r[n][2] for r in R)/1000:7.0f} {mse:13.6f} {mse/var:8.4f} {int(3.6e6/ms):14,}")
    print("\nDisagreement filter (keep iff |sig(cheap)-sig(deeper)| < thr); does cheap disagreement flag bad labels?")
    for lo, hi in (("d8", "d12"), ("d10", "d14")):
        for thr in (1.0, 0.10, 0.05, 0.03):
            kept = [r for r in R if abs(sig(r[lo][0]) - sig(r[hi][0])) < thr]
            mse = st.mean((sig(r[hi][0]) - ref(r)) ** 2 for r in kept)
            print(f"  {lo}/{hi} thr {thr:4}: keep {100*len(kept)/len(R):5.1f}%  {hi} MSE on kept {mse:.5f}")
    flips = sum(1 for r in R if r["d12"][0] * r["t2000"][0] < 0 and min(abs(r["d12"][0]), abs(r["t2000"][0])) > 100)
    print(f"\nd12 vs t2000 sign flips beyond +-100 cp: {flips}/{len(R)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--stockfish", required=True); r.add_argument("--fens", required=True)
    r.add_argument("--out", required=True); r.add_argument("--procs", type=int, default=4)
    p = sub.add_parser("report"); p.add_argument("--data", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
