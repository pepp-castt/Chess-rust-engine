#!/usr/bin/env python3
"""Tactical-accuracy harness (Agent D).

  build : FEN list -> EPD suite whose `bm` is the move Stockfish 19 (MAX, MultiPV=2, >=1 s) prefers
          over the runner-up by >= --gap centipawns.  Filtering for a *unique* best move keeps
          the suite from rewarding lucky or arbitrary choices ("don't blindly imitate SF moves":
          we only keep positions where the alternative is clearly worse).
          Score perspective is side-to-move; mate scores are mapped to +-100000.
  run   : play each EPD position with `go movetime|nodes` and report solve rate + mean time.
          Emits JSON so runs are comparable commit to commit (tactical accuracy is a tracked metric).

  python tools/tactics.py build --stockfish ./stockfish --fens seeds.txt --out tactics.epd --movetime-ms 2000
  python tools/tactics.py run --engine ./target/release/nextgen --epd tactics.epd --movetime-ms 1000 --json out.json
"""
import argparse, json, sys, time
import chess, chess.engine



def build(a):
    if a.movetime_ms < 1000:
        sys.exit("refusing movetime < 1000 ms for labels (no shallow teacher labels)")
    eng = chess.engine.SimpleEngine.popen_uci(a.stockfish)
    eng.configure({"Threads": a.threads, "Hash": a.hash, "MultiPV": 2})
    ident = eng.id.get("name", "?")
    kept = seen = 0
    with open(a.out, "w") as out:
        out.write(f"; built by tools/tactics.py teacher={ident} movetime_ms={a.movetime_ms} gap>={a.gap} multipv=2 threads={a.threads}\n")
        for line in open(a.fens):
            fen = line.split(";")[0].strip()
            if not fen:
                continue
            try:
                b = chess.Board(fen)
            except ValueError:
                continue
            if not b.is_valid() or b.is_game_over():
                continue
            seen += 1
            infos = eng.analyse(b, chess.engine.Limit(time=a.movetime_ms / 1000), multipv=2)
            if len(infos) < 2:
                continue  # single legal move: not a puzzle
            s = [i["score"].relative.score(mate_score=100000) for i in infos]
            if s[0] - s[1] >= a.gap and s[0] > -300:
                bm = b.san(infos[0]["pv"][0])
                out.write(f'{b.epd()} bm {bm}; id "t{kept}"; c0 "sf={s[0]} second={s[1]}";\n')
                kept += 1
            if a.max and kept >= a.max:
                break
    eng.quit()
    print(f"seen={seen} kept={kept} teacher={ident}")


def run(a):
    eng = chess.engine.SimpleEngine.popen_uci(a.engine)
    for kv in a.option or []:
        k, v = kv.split("=", 1)
        eng.configure({k: int(v) if v.lstrip("-").isdigit() else (v == "true" if v in ("true", "false") else v)})
    lim = chess.engine.Limit(nodes=a.nodes) if a.nodes else chess.engine.Limit(time=a.movetime_ms / 1000)
    solved = total = 0
    rows = []
    t_all = time.time()
    for line in open(a.epd):
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        b = chess.Board()
        ops = b.set_epd(line)
        bms = ops.get("bm", [])
        ams = ops.get("am", [])
        if not bms and not ams:
            continue
        t0 = time.time()
        mv = eng.play(b, lim).move
        ok = (mv in bms) if bms else (mv not in ams)
        total += 1
        solved += ok
        rows.append({"id": ops.get("id"), "move": b.san(mv), "bm": [b.san(m) for m in bms], "ok": bool(ok), "s": round(time.time() - t0, 3)})
        if a.verbose:
            print(("OK  " if ok else "MISS"), ops.get("id"), b.san(mv), "bm", [b.san(m) for m in bms])
    ident = eng.id.get("name", "?")
    eng.quit()
    res = {"engine": ident, "epd": a.epd, "limit": f"nodes {a.nodes}" if a.nodes else f"movetime {a.movetime_ms} ms",
           "options": a.option or [], "solved": solved, "total": total,
           "rate": round(solved / total, 4) if total else None, "wall_s": round(time.time() - t_all, 1), "rows": rows}
    print(f"{ident}: solved {solved}/{total} ({res['rate']}) at {res['limit']}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("--stockfish", required=True); p.add_argument("--fens", required=True); p.add_argument("--out", required=True)
    p.add_argument("--movetime-ms", type=int, default=2000); p.add_argument("--gap", type=int, default=150)
    p.add_argument("--threads", type=int, default=1); p.add_argument("--hash", type=int, default=64); p.add_argument("--max", type=int, default=0)
    p.set_defaults(f=build)
    p = sp.add_parser("run")
    p.add_argument("--engine", required=True); p.add_argument("--epd", required=True)
    p.add_argument("--movetime-ms", type=int, default=1000); p.add_argument("--nodes", type=int, default=0)
    p.add_argument("--option", action="append", help="UCI option k=v (repeatable)"); p.add_argument("--json"); p.add_argument("--verbose", action="store_true")
    p.set_defaults(f=run)
    a = ap.parse_args()
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
