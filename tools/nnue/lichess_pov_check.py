#!/usr/bin/env python3
"""Empirically decide whether `cp` in Lichess/chess-position-evaluations is White-POV or STM-POV.

For every row we know the material balance (white - black, P=1 N=B=3 R=5 Q=9) and who is to move.
Positions with a large material edge are decided by material; the sign of `cp` must agree with
material *from White's view* if cp is White-POV, or from the mover's view if cp is STM-POV.
Prints agreement rates for both hypotheses on rows with |material| >= 5 and |cp| >= 150.
"""
import sys
import pyarrow.parquet as pq

V = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "p": -1, "n": -3, "b": -3, "r": -5, "q": -9}
pf = pq.ParquetFile(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 400000
n = white_ok = stm_ok = 0
for batch in pf.iter_batches(batch_size=1 << 16, columns=["fen", "cp"]):
    for fen, cp in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
        if cp is None or abs(cp) < 150:
            continue
        board, stm = fen.split(" ")[:2]
        mat = sum(V.get(c, 0) for c in board)
        if abs(mat) < 5:
            continue
        n += 1
        white_ok += (cp > 0) == (mat > 0)
        mover_mat = mat if stm == "w" else -mat
        stm_ok += (cp > 0) == (mover_mat > 0)
        if n >= limit:
            break
    if n >= limit:
        break
print(f"rows tested={n}  agree if cp is WHITE-POV: {white_ok/n:.4f}   agree if cp is STM-POV: {stm_ok/n:.4f}")
print("=> cp is", "WHITE-POV (use --pov white)" if white_ok > stm_ok else "STM-POV (use --pov stm)")
