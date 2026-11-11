#!/usr/bin/env python3
"""SPSA tuner for Chess-rust-engine search parameters (src/params.rs).

Each iteration k:
  1. draw delta_i in {-1,+1}
  2. theta+ = clip(theta + c_k*delta), theta- = clip(theta - c_k*delta)
  3. play `--rounds` rounds (2 games each, colours swapped, same opening) of
     engine(theta+) vs engine(theta-) with fastchess, opponent-independent
  4. result = (W - L) / games  in [-1, 1]  (from theta+'s point of view)
  5. theta_i += c_k * R_k * result * delta_i        (fishtest/OpenBench form)

Gains follow Spall: c_k = c_end * (N/(k+1))^gamma, a_k = a_end*((A+N)/(A+k+1))^alpha,
R_k = a_k / c_k^2, with a_end = r_end * c_end^2.

Important honesty notes:
* The SPSA match score is the *training signal*, not the verdict. Tuned values
  must still pass `sprt.yml` against main before they are merged.
* Few games per iteration is fine (SPSA is noisy-gradient tolerant); thousands
  of iterations are not optional. Use `--average-last` for the reported values.
* Time control floor is 5+0.05, same as sprt.yml. Nothing at ~100 ms/move.

Usage:
  spsa.py --engine ./engine --book tools/openings_book.epd --iterations 400 --rounds 4
  spsa.py --simulate --iterations 3000          # optimiser self-test, no engine needed
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass

ALPHA, GAMMA = 0.602, 0.101
R_MULT = 1.0  # set from --r-mult in main()


@dataclass
class Param:
    name: str
    default: float
    lo: float
    hi: float
    c_end: float
    r_end: float


def parse_spec(lines: list[str]) -> list[Param]:
    out = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 7 or parts[1] != "int":
            continue
        out.append(Param(parts[0], *(float(x) for x in parts[2:])))
    return out


def engine_spec(engine: str) -> list[Param]:
    r = subprocess.run([engine], input="params\nquit\n", capture_output=True, text=True, timeout=30)
    spec = parse_spec(r.stdout.splitlines())
    if not spec:
        sys.exit(f"{engine} printed no SPSA spec for `params` (needs src/params.rs)")
    return spec


def check_tc(tc: str) -> None:
    base, _, inc = tc.partition("+")
    b = float(base.split("/")[-1])
    i = float(inc or 0)
    if b < 5 or (b + 40 * i) < 7:
        sys.exit(f"refusing tc {tc}: floor is 5+0.05 (no ~100 ms games)")


def clip(v: float, p: Param) -> float:
    return min(p.hi, max(p.lo, v))


def gains(p: Param, k: int, n: int, a_big: float) -> tuple[float, float]:
    """Return (c_k, R_k) for parameter p at iteration k (0-based) of n."""
    c_k = p.c_end * (n / (k + 1)) ** GAMMA
    a_end = p.r_end * R_MULT * p.c_end**2
    a_k = a_end * ((a_big + n) / (a_big + k + 1)) ** ALPHA
    return c_k, a_k / c_k**2


# fastchess prints, per report (last one wins):
#   Results of plus vs minus (5+0.05, ...):
#   Games: 4, Wins: 0, Losses: 4, Draws: 0, Points: 0.0 (0.00 %)
FC_HEAD = re.compile(r"Results of (\S+) vs (\S+)")
FC_SCORE = re.compile(r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)")


def play_match(args, plus: dict[str, int], minus: dict[str, int], seed: int) -> tuple[int, int, int]:
    """Return (wins, losses, draws) for `plus` vs `minus`."""

    def opts(d: dict[str, int]) -> list[str]:
        return [f"option.{k}={v}" for k, v in d.items()]

    cmd = [
        args.fastchess,
        "-engine", f"cmd={args.engine}", "name=plus", "option.Hash=16", "option.Threads=1", *opts(plus),
        "-engine", f"cmd={args.engine}", "name=minus", "option.Hash=16", "option.Threads=1", *opts(minus),
        "-each", f"tc={args.tc}",
        "-rounds", str(args.rounds), "-games", "2", "-repeat", "-concurrency", str(args.concurrency),
        "-openings", f"file={args.book}", "format=epd", "order=random",
        "-srand", str(seed),
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=4", "score=800", "twosided=true",
        "-maxmoves", "300", "-recover",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.match_timeout)
    out = r.stdout
    heads = list(FC_HEAD.finditer(out))
    scores = list(FC_SCORE.finditer(out))
    if not heads or not scores:
        raise RuntimeError("could not parse fastchess output:\n" + out[-1500:] + r.stderr[-500:])
    first = heads[-1].group(1)
    _, w, l, d = (int(x) for x in scores[-1].groups())
    return (w, l, d) if first == "plus" else (l, w, d)


def simulate_match(spec, plus, minus, rng, optimum):
    """Synthetic game outcome from a hidden objective (optimiser self-test)."""

    def loss(d):
        return sum(((d[p.name] - optimum[p.name]) / (p.hi - p.lo)) ** 2 for p in spec)

    diff = loss(minus) - loss(plus)  # positive => plus better
    p_win = 1 / (1 + math.exp(-diff * 600))
    w = l = d = 0
    for _ in range(16):
        x = rng.random()
        if x < 0.3:
            d += 1
        elif x < 0.3 + 0.7 * p_win:
            w += 1
        else:
            l += 1
    return w, l, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="./engine")
    ap.add_argument("--fastchess", default="fastchess")
    ap.add_argument("--book", default="tools/openings_book.epd")
    ap.add_argument("--tc", default="5+0.05")
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=4, help="rounds per iteration (2 games each)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--state", default="spsa_state.json", help="resume file")
    ap.add_argument("--only", default="", help="comma list of param names to tune (default: all)")
    ap.add_argument("--average-last", type=float, default=0.25, help="Polyak-average the last fraction of thetas")
    ap.add_argument("--r-mult", type=float, default=8.0,
                    help="multiply the spec's r_end. fishtest's 0.002 assumes 100k+ game pairs; with a few "
                         "thousand CI games a larger step is needed or theta barely moves")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--match-timeout", type=int, default=900)
    ap.add_argument("--time-budget-min", type=float, default=0, help="stop gracefully after this many minutes")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--out", default="spsa_result.json")
    args = ap.parse_args()

    global R_MULT
    R_MULT = args.r_mult
    rng = random.Random(args.seed)
    if args.simulate:
        # Synthetic spec identical in shape to the real one.
        spec = [Param(f"P{i}", 50, 0, 100, 4, 0.002) for i in range(8)]
        optimum = {p.name: rng.uniform(25, 75) for p in spec}
    else:
        check_tc(args.tc)
        spec = engine_spec(args.engine)
        optimum = {}
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        spec = [p for p in spec if p.name in keep]
        if not spec:
            sys.exit("--only matched nothing")

    n = args.iterations
    a_big = 0.1 * n
    theta = {p.name: p.default for p in spec}
    history: list[dict[str, float]] = []
    k0 = 0
    total = [0, 0, 0]
    if os.path.exists(args.state) and not args.simulate:
        st = json.load(open(args.state))
        if st.get("names") == [p.name for p in spec] and st.get("iterations") == n:
            theta, history, k0, total = st["theta"], st["history"], st["k"], st["total"]
            print(f"resumed at iteration {k0}/{n}", flush=True)
        else:
            print("state file does not match this spec/iteration count; starting fresh", flush=True)

    t_start = time.time()
    for k in range(k0, n):
        if args.time_budget_min and (time.time() - t_start) / 60 > args.time_budget_min:
            print(f"time budget reached at iteration {k}; saving state for resume", flush=True)
            break
        delta = {p.name: rng.choice((-1, 1)) for p in spec}
        ck = {p.name: gains(p, k, n, a_big) for p in spec}
        plus = {p.name: int(round(clip(theta[p.name] + ck[p.name][0] * delta[p.name], p))) for p in spec}
        minus = {p.name: int(round(clip(theta[p.name] - ck[p.name][0] * delta[p.name], p))) for p in spec}
        if args.simulate:
            w, l, d = simulate_match(spec, plus, minus, rng, optimum)
        else:
            try:
                w, l, d = play_match(args, plus, minus, args.seed * 100003 + k)
            except Exception as e:  # a crashed batch must not poison theta
                print(f"iter {k}: match failed ({e}); skipped", flush=True)
                continue
        games = w + l + d
        result = (w - l) / games if games else 0.0
        for p in spec:
            c_k, r_k = ck[p.name]
            theta[p.name] = clip(theta[p.name] + c_k * r_k * result * delta[p.name], p)
        history.append(dict(theta))
        total = [total[0] + w, total[1] + l, total[2] + d]
        if k % 10 == 0 or k == n - 1:
            print(f"iter {k+1}/{n} plus {w}-{l}-{d}  elapsed {(time.time()-t_start)/60:.1f}m", flush=True)
        if not args.simulate:
            json.dump(
                {"names": [p.name for p in spec], "iterations": n, "k": k + 1, "theta": theta,
                 "history": history, "total": total},
                open(args.state, "w"),
            )
    else:
        k = n  # loop completed

    # Polyak average of the tail: much less noisy than the last theta.
    tail = history[max(0, int(len(history) * (1 - args.average_last))):] or [theta]
    avg = {p.name: sum(h[p.name] for h in tail) / len(tail) for p in spec}
    final = {p.name: int(round(clip(avg[p.name], p))) for p in spec}
    result = {
        "iterations_done": len(history),
        "games_total": sum(total),
        "plus_minus_wld": total,
        "tuned": final,
        "defaults": {p.name: int(p.default) for p in spec},
        "changed": {p.name: final[p.name] for p in spec if final[p.name] != int(p.default)},
        "uci_options": " ".join(f"option.{k}={v}" for k, v in final.items()),
        "note": "SPSA signal only. Gate with sprt.yml vs main before merging.",
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    if args.simulate:
        err0 = sum(abs(p.default - optimum[p.name]) for p in spec) / len(spec)
        err1 = sum(abs(final[p.name] - optimum[p.name]) for p in spec) / len(spec)
        print(f"SIMULATION mean |theta-opt|: start {err0:.2f} -> end {err1:.2f}")
        sys.exit(0 if err1 < 0.5 * err0 else 1)


if __name__ == "__main__":
    main()
