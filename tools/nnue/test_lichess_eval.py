#!/usr/bin/env python3
"""Tests for lichess_eval_to_bin.py. Needs python-chess + zstandard. Run:  python tools/nnue/test_lichess_eval.py
Uses LICHESS_SAMPLE=<jsonl> if set, else streams the first ~3 MB of the real database."""
import json, os, struct, sys, urllib.request
import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lichess_eval_to_bin as L


def sample_lines(n=4000):
    p = os.environ.get("LICHESS_SAMPLE")
    if p:
        out = []
        for ln in open(p, "rb"):
            out.append(ln)
            if len(out) >= n:
                break
        return out
    import zstandard
    req = urllib.request.Request(L.URL, headers={"Range": "bytes=0-6000000"})
    d = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(urllib.request.urlopen(req, timeout=120))
    buf = b""
    try:
        while len(buf) < 3_000_000:
            c = d.read(1 << 20)
            if not c:
                break
            buf += c
    except Exception:
        pass
    return buf.split(b"\n")[:-1][:n]


def ref_pack(board, score, wdl):  # == tools/nnue/datagen_movetime.py::pack
    occ, nib = 0, []
    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None:
            occ |= 1 << sq
            nib.append((0 if p.color else 6) + p.piece_type - 1)
    nib += [0] * (32 - len(nib))
    b = bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(16))
    return L.REC.pack(occ, b, 0 if board.turn else 1, score, wdl)


def test_pov_and_pack(lines):
    n = 0
    for ln in lines:
        rec, why, _, _ = L.convert_line(ln, 0, 10 ** 6, True)
        if rec is None:
            continue
        j = json.loads(ln)
        e = max(j["evals"], key=lambda x: (x["depth"], x["knodes"]))
        cp = e["pvs"][0]["cp"]
        bd = chess.Board(j["fen"])
        want = cp if bd.turn else -cp  # Lichess is white POV -> stm POV
        assert rec == ref_pack(bd, want, 1), (j["fen"], cp)
        n += 1
    assert n > 1000, n
    print(f"ok pov+pack identical to python-chess reference on {n} records")


def test_filters(lines):
    chk = noisy = legal_bad = 0
    for ln in lines:
        j = json.loads(ln)
        e = max(j["evals"], key=lambda x: (x["depth"], x["knodes"]))
        pv = e["pvs"][0]
        if "cp" not in pv:
            continue
        bd = chess.Board(j["fen"])
        bd.castling_rights = 0  # features ignore castling; only validate the piece layout
        rec, why, _, _ = L.convert_line(ln, 0, 10 ** 6, False)
        if not bd.is_valid():
            assert why == "illegal", (j["fen"], why)
            legal_bad += 1
            continue
        if bd.is_check():
            assert why == "in_check", (j["fen"], why)
            chk += 1
            continue
        mv = chess.Move.from_uci(pv["line"].split()[0]) if pv.get("line") else None
        if mv is not None:
            cap = bd.is_capture(mv) or mv.promotion is not None
            if cap:
                assert why.startswith("noisy"), (j["fen"], pv["line"][:12], why)
                noisy += 1
                continue
        assert why == "ok", (j["fen"], why)
    print(f"ok filters agree with python-chess: in_check={chk} noisy={noisy} illegal={legal_bad}")


def test_black_to_move_flip():
    j = {"fen": "4k3/8/8/8/8/8/4P3/4K3 b - -", "evals": [{"pvs": [{"cp": 150, "line": "e8d8"}], "knodes": 1, "depth": 30}]}
    rec, why, _, _ = L.convert_line(json.dumps(j).encode(), 20, 2000, False)
    assert why == "ok"
    assert L.REC.unpack(rec)[2:] == (1, -150, 1), rec  # stm=black, -150 stm POV
    j["fen"] = j["fen"].replace(" b ", " w ")
    rec, _, _, _ = L.convert_line(json.dumps(j).encode().replace(b"e8d8", b"e1d1"), 20, 2000, False)
    assert L.REC.unpack(rec)[2:] == (0, 150, 1)
    print("ok black-to-move flip: white-POV +150 -> -150 for black, +150 for white")


def test_overfull_board_rejected():
    """Regression: the audit found 33/36/62-piece rows that overflow the 32 nibble slots."""
    fen = "rnbqkbnr/pppppppp/pppppppp/8/8/PPPPPPPP/PPPPPPPP/RNBQKBNR w - -"
    j = {"fen": fen, "evals": [{"pvs": [{"cp": 10, "line": "e2e4"}], "knodes": 1, "depth": 30}]}
    assert L.convert_line(json.dumps(j).encode(), 20, 2000, True)[1] == "illegal"
    print("ok over-full boards rejected")


def test_lichess_is_white_pov(lines):
    """Evidence for the perspective claim: play out mating PVs; mate>0 must mean WHITE mates."""
    seen = 0
    for ln in lines:
        if b'"mate"' not in ln:
            continue
        j = json.loads(ln)
        for e in j["evals"]:
            for pv in e["pvs"]:
                if "mate" not in pv:
                    continue
                bd = chess.Board(j["fen"])
                try:
                    for m in pv["line"].split():
                        bd.push_uci(m)
                except Exception:
                    continue
                if not bd.is_checkmate():
                    continue
                white_mates = not bd.turn
                assert (pv["mate"] > 0) == white_mates, (j["fen"], pv)
                seen += 1
    assert seen >= 20, f"too few mating PVs to prove anything ({seen})"
    print(f"ok Lichess scores are WHITE-POV ({seen} mating lines verified)")


if __name__ == "__main__":
    ls = sample_lines()
    print(f"{len(ls)} sample lines")
    test_black_to_move_flip()
    test_overfull_board_rejected()
    test_pov_and_pack(ls)
    test_filters(ls)
    test_lichess_is_white_pov(ls)
    print("ALL OK")
