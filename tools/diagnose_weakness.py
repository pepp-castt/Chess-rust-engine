#!/usr/bin/env python3
"""Diagnose WHY NextGen lost: reconstruct, label with Stockfish 19, attribute, export.

Pipeline per costly mistake (mission section 4):
  1. Reconstruct the position (FEN + game history is kept in the PGN; the replay below
     starts from the FEN, so TT/history effects are reported as `history-variance`).
  2. Ask SF19 (full strength, MultiPV) for candidate moves and scores, >= 0.5 s/position.
  3. Attribute the mistake using OUR engine:
        movegen         engine's bestmove is illegal / engine crashed
        history-variance replaying at the same time budget already finds an acceptable move
        search-depth    an acceptable move appears within +2 plies of the depth we reached
        search-pruning  an acceptable move appears only at +3..+8 plies (ordering / pruning)
        eval            never found AND static eval prefers the move we played over SF's move
        search-horizon  never found, static eval agrees with SF (tactic beyond +8 plies)
  4. Emit measurable artefacts: JSONL, EPD regression suite (bm = all moves within
     --tol cp of SF's best), a markdown report, and provenance.json.

Positions where SF already says the game is decided (|cp| >= --max-abs) are NOT mined:
we learn from decisions, not from lost positions. Analysis of a game stops at the first
such position.

NOT a trainer: output is a diagnosis + hard-position set. Do not blindly imitate single SF
moves; use the classes to choose WHICH subsystem to change, then SPRT the change.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.engine
import chess.pgn

MIN_MOVETIME = 0.5
CLAMP = 1000  # cp, so a mate score does not dominate averages


def clamp(x: int) -> int:
    return max(-CLAMP, min(CLAMP, x))


def pov_cp(score: chess.engine.PovScore, color: bool) -> int:
    return clamp(score.pov(color).score(mate_score=10000))


def phase_of(board: chess.Board) -> str:
    pieces = sum(len(board.pieces(t, c)) for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) for c in (True, False))
    if board.fullmove_number <= 12:
        return "opening"
    return "endgame" if pieces <= 6 else "middlegame"


COMMENT_RE = re.compile(r"^\s*(M-?\d+|[+-]\d+\.\d+)/(\d+)\s+([\d.]+)s")


def parse_comment(c: str):
    """`+0.35/12 0.90s` (play_max.py / cutechess style) -> (depth, seconds)."""
    m = COMMENT_RE.match(c or "")
    if not m:
        return None, None
    return int(m.group(2)), float(m.group(3))


def static_eval(ours_path: str, fen: str, move: chess.Move | None) -> int | None:
    """Static eval (cp, side-to-move POV) via the engine's `eval` command, after `move`."""
    b = chess.Board(fen)
    if move is not None:
        b.push(move)
    try:
        out = subprocess.run(
            [ours_path], input=f"position fen {b.fen()}\neval\nquit\n", capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        return None
    m = re.search(r"eval\s+(-?\d+)\s*cp", out)
    return int(m.group(1)) if m else None


def stable_solve(ladder: dict, acceptable: list):
    """Smallest depth d such that the engine's move is acceptable at d AND every deeper rung.

    A best-of-N 'ever acceptable' test counts move flip-flopping as convergence and
    over-attributes to search depth (measured: it said 51% search/variance while 6x more
    time solved only +7 pts). Stable convergence is the honest criterion."""
    items = sorted((int(k), v) for k, v in ladder.items())
    solve = None
    for d, m in reversed(items):
        if m in acceptable:
            solve = d
        else:
            break
    return solve


def classify(row: dict) -> str:
    if not row.get("legal", True):
        return "movegen"
    if row.get("replay_acceptable"):
        return "history-variance"
    solve = row.get("solve_depth")
    if solve is not None and solve - row["orig_depth"] <= 2:
        return "search-depth"
    if solve is not None:
        return "search-pruning"
    if row.get("static_prefers_played"):
        return "eval"
    return "search-horizon"


def run_diag(a) -> int:
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.movetime < MIN_MOVETIME:
        print(f"refusing --movetime {a.movetime} < {MIN_MOVETIME}s (shallow labels teach junk)", file=sys.stderr)
        return 2

    sf = chess.engine.SimpleEngine.popen_uci(a.stockfish)
    sf.configure({"Threads": a.threads, "Hash": a.hash, "UCI_LimitStrength": False})
    sf_name = sf.id.get("name", "?")
    ours = chess.engine.SimpleEngine.popen_uci(a.ours)
    ours.configure({"Hash": 64, "Threads": 1})
    ours_name = ours.id.get("name", "?")

    rows = []
    games = positions = 0
    t_start = time.time()
    for pgn_path in a.pgn:
        with open(pgn_path) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                w, b = g.headers.get("White", ""), g.headers.get("Black", "")
                names = tuple(a.name)
                if any(n.lower() in w.lower() for n in names):
                    us = chess.WHITE
                elif any(n.lower() in b.lower() for n in names):
                    us = chess.BLACK
                else:
                    continue
                res = g.headers.get("Result", "*")
                lost = (res == "0-1" and us == chess.WHITE) or (res == "1-0" and us == chess.BLACK)
                if not lost and not a.all_games:
                    continue
                games += 1
                gid = f"{Path(pgn_path).name}#{g.headers.get('Round', games)}"
                board = g.board()
                ply0 = len(board.move_stack)
                for node in g.mainline():
                    mv = node.move
                    ply = len(board.move_stack)
                    if board.turn == us and ply - ply0 >= a.skip_plies:
                        positions += 1
                        infos = sf.analyse(board, chess.engine.Limit(time=a.movetime), multipv=a.multipv)
                        cands = {}
                        for inf in infos:
                            if inf.get("pv"):
                                cands[inf["pv"][0]] = pov_cp(inf["score"], us)
                        best_mv = infos[0]["pv"][0]
                        best_cp = cands[best_mv]
                        if abs(best_cp) >= a.max_abs:
                            break  # already decided: stop mining this game
                        if mv in cands:
                            played_cp = cands[mv]
                        else:
                            alt = sf.analyse(board, chess.engine.Limit(time=a.movetime), root_moves=[mv])
                            played_cp = pov_cp(alt["score"], us)
                        loss = best_cp - played_cp
                        if loss >= a.min_loss and mv != best_mv:
                            rows.append(diagnose_one(a, ours, board.copy(), mv, best_mv, best_cp, played_cp, loss, cands, gid, ply, node.comment))
                    board.push(mv)
    sf.quit()
    ours.quit()

    prov = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "teacher": sf_name,
        "teacher_settings": {"UCI_LimitStrength": False, "Threads": a.threads, "Hash": a.hash, "movetime_s": a.movetime, "multipv": a.multipv},
        "engine": ours_name,
        "engine_rev": a.ours_rev,
        "pgn_inputs": [str(p) for p in a.pgn],
        "mining": {"min_loss_cp": a.min_loss, "max_abs_cp": a.max_abs, "tol_cp": a.tol, "skip_plies": a.skip_plies},
        "games_analysed": games,
        "our_positions_labelled": positions,
        "mistakes": len(rows),
        "host": {"cpus": os.cpu_count(), "machine": platform.machine(), "python": platform.python_version()},
        "wall_s": round(time.time() - t_start, 1),
        "note": "score perspective: cp from OUR side; replay starts from FEN (no repetition history)",
    }
    write_outputs(out_dir, prov, rows)
    print(f"games={games} positions={positions} mistakes={len(rows)} -> {out_dir}")
    return 0


def write_outputs(out_dir: Path, prov: dict, rows: list):
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    with open(out_dir / "diagnosis.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    with open(out_dir / "hard_suite.epd", "w") as fo:
        for i, r in enumerate(rows):
            b = chess.Board(r["fen"])
            bms = [chess.Move.from_uci(u) for u in r["acceptable"]]
            tag = "class=%s loss=%s phase=%s" % (r["class"], r["loss_cp"], r["phase"])
            fo.write(b.epd(bm=bms, am=[chess.Move.from_uci(r["played"])], id="nextgen-%04d" % (i + 1), c0=tag) + "\n")
    with open(out_dir / "hard_positions.tsv", "w") as fo:
        fo.write("fen\tsf_best\tsf_cp_pov_us\tclass\n")
        for r in rows:
            fo.write(f'{r["fen"]}\t{r["best"]}\t{r["best_cp"]}\t{r["class"]}\n')
    write_report(out_dir / "report.md", prov, rows)


def diagnose_one(a, ours, board, played, best, best_cp, played_cp, loss, cands, gid, ply, comment):
    fen = board.fen()
    acceptable = [m for m, cp in cands.items() if cp >= best_cp - a.tol]
    orig_depth, orig_s = parse_comment(comment)
    orig_depth = orig_depth or 8
    budget = max(0.05, orig_s or 0.2)
    row = {
        "game": gid,
        "ply": ply,
        "fen": fen,
        "played": played.uci(),
        "best": best.uci(),
        "best_cp": best_cp,
        "played_cp": played_cp,
        "loss_cp": loss,
        "acceptable": [m.uci() for m in acceptable],
        "phase": phase_of(board),
        "tactical": bool(board.is_capture(best) or board.gives_check(best) or best.promotion or loss >= 300),
        "orig_depth": orig_depth,
        "orig_time_s": orig_s,
    }
    try:
        # (a) same budget replay
        r = ours.play(board, chess.engine.Limit(time=budget))
        row["replay_move"] = r.move.uci() if r.move else None
        legal = r.move is not None and r.move in board.legal_moves
        row["replay_acceptable"] = bool(legal and r.move in acceptable)
        # (b) depth ladder
        ladder = {}
        for d in range(orig_depth, orig_depth + a.extra_plies + 1, 2):
            rr = ours.play(board, chess.engine.Limit(depth=d, time=a.depth_cap_s), info=chess.engine.INFO_SCORE)
            mv = rr.move
            if mv is not None and mv not in board.legal_moves:
                legal = False
            ladder[d] = mv.uci() if mv else None
        row["ladder"] = ladder
        row["any_solve_depth"] = min((d for d, m in ladder.items() if m in row["acceptable"]), default=None)
        row["solve_depth"] = stable_solve(ladder, row["acceptable"])
        row["legal"] = legal
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
        row.update({"legal": False, "error": str(e)})
        row["class"] = "movegen"
        return row

    # (c) static eval agreement (our POV after each move; engine reports side-to-move POV)
    e_best = static_eval(a.ours, fen, best)
    e_play = static_eval(a.ours, fen, played)
    row["static_after_best_us"] = -e_best if e_best is not None else None
    row["static_after_played_us"] = -e_play if e_play is not None else None
    prefers_played = (
        e_best is not None and e_play is not None and (-e_play) >= (-e_best)
    )
    row["static_prefers_played"] = bool(prefers_played)

    row["class"] = classify(row)
    return row


ADVICE = {
    "movegen": "Perft/legality bug. Fix before anything else (tests/perft.rs + this FEN).",
    "history-variance": "Same budget finds an acceptable move from a clean state: look at TT/history pollution, time management (soft/hard limits), instability handling.",
    "search-depth": "Engine finds it 1-2 plies deeper: NPS, time allocation, or too-aggressive stopping when the score drops (extend time on fail-low).",
    "search-pruning": "Needs 3-8 more plies: ordering / LMR / NMP / futility are cutting the line. Candidate for SPRT on reduction tweaks + corrhist.",
    "eval": "Static eval prefers the played move even after deeper search: train/tune the evaluation (NNUE data should include these positions).",
    "search-horizon": "Eval agrees with SF but the tactic is beyond +8 plies: extensions (singular/check), quiescence, SEE.",
}


def write_report(path: Path, prov: dict, rows: list[dict]):
    n = len(rows)
    byc = Counter(r["class"] for r in rows)
    lossc = defaultdict(list)
    for r in rows:
        lossc[r["class"]].append(r["loss_cp"])
    byp = Counter((r["phase"], r["class"]) for r in rows)
    L = []
    L.append("# NextGen weakness diagnosis\n")
    L.append(f"- teacher: **{prov['teacher']}** (full strength, {prov['teacher_settings']['movetime_s']} s/position, MultiPV {prov['teacher_settings']['multipv']})")
    L.append(f"- engine: {prov['engine']} @ `{prov['engine_rev']}`")
    L.append(f"- games analysed: {prov['games_analysed']}, our positions labelled: {prov['our_positions_labelled']}, costly mistakes (>= {prov['mining']['min_loss_cp']} cp): **{n}**")
    L.append(f"- host: {prov['host']['cpus']} cpus {prov['host']['machine']}, wall {prov['wall_s']} s\n")
    L.append("## Attribution\n")
    L.append("| class | n | share | mean loss (cp) | what to do |")
    L.append("|---|---|---|---|---|")
    for c, k in byc.most_common():
        L.append(f"| {c} | {k} | {100 * k / max(1, n):.0f}% | {sum(lossc[c]) / k:.0f} | {ADVICE[c]} |")
    L.append("\n## By phase\n")
    L.append("| phase | class | n |")
    L.append("|---|---|---|")
    for (p, c), k in sorted(byp.items()):
        L.append(f"| {p} | {c} | {k} |")
    L.append("\n## Ten costliest\n")
    L.append("| loss | class | phase | played | SF best | solve depth (orig) | FEN |")
    L.append("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["loss_cp"])[:10]:
        L.append(f"| {r['loss_cp']} | {r['class']} | {r['phase']} | {r['played']} | {r['best']} | {r.get('solve_depth')} ({r['orig_depth']}) | `{r['fen']}` |")
    L.append("\n> Small-n warning: classes are hypotheses. A change is kept only if the EPD suite AND an SPRT agree.\n")
    path.write_text("\n".join(L))


def reclassify(d: str) -> int:
    """Re-run attribution on stored diagnosis.jsonl (no Stockfish needed)."""
    out_dir = Path(d)
    rows = [json.loads(l) for l in open(out_dir / "diagnosis.jsonl")]
    for r in rows:
        r["solve_depth"] = stable_solve(r.get("ladder", {}), r["acceptable"])
        r["class"] = classify(r)
    prov = json.loads((out_dir / "provenance.json").read_text())
    prov["mistakes"] = len(rows)
    prov["classifier"] = "stable-solve v2 (acceptable at depth d and all deeper rungs)"
    write_outputs(out_dir, prov, rows)
    print(f"reclassified {len(rows)} -> {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", nargs="*")
    ap.add_argument("--reclassify", default="", help="DIR with diagnosis.jsonl: recompute classes offline")
    ap.add_argument("--stockfish", default="")
    ap.add_argument("--ours", default="target/release/chess-rust-engine")
    ap.add_argument("--ours-rev", default="unknown")
    ap.add_argument("--name", action="append", default=None, help="substring(s) of our engine's PGN name")
    ap.add_argument("--movetime", type=float, default=1.0, help="SF19 s/position (floor 0.5)")
    ap.add_argument("--multipv", type=int, default=4)
    ap.add_argument("--min-loss", type=int, default=80)
    ap.add_argument("--max-abs", type=int, default=600)
    ap.add_argument("--tol", type=int, default=30, help="moves within this many cp of SF's best are acceptable")
    ap.add_argument("--skip-plies", type=int, default=0, help="plies after the book position to skip")
    ap.add_argument("--extra-plies", type=int, default=8)
    ap.add_argument("--depth-cap-s", type=float, default=6.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--all-games", action="store_true")
    ap.add_argument("--out-dir", default="results/diagnosis")
    a = ap.parse_args()
    if a.reclassify:
        return reclassify(a.reclassify)
    if not a.pgn or not a.stockfish:
        ap.error("pgn files and --stockfish are required unless --reclassify is used")
    a.name = a.name or ["NextGen", "Chess-rust-engine", "Rust"]
    return run_diag(a)


if __name__ == "__main__":
    raise SystemExit(main())
